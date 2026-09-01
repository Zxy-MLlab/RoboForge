from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from openhands.sdk import ImageContent, TextContent
from openhands.sdk.event import ActionEvent
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
)

from .models import ExperimentEvidence
from .service import ExperimentService, ProtocolError
from .store import CorruptStore


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class ObserveAction(Action):
    pass


class RunControllerAction(Action):
    intent: str = Field(
        min_length=1,
        description=(
            "The factual hypothesis or intended behavioral change tested by this "
            "physical experiment. The Harness captures it as provenance."
        ),
    )
    assets_used: list[str] = Field(default_factory=list,
        description="Asset IDs actually read and semantically used in this Controller revision.")


class InspectTrialAction(Action):
    ref: str | None = Field(
        default=None,
        description="Experiment reference. Omit to inspect the latest experiment.",
    )


class CompareTrialsAction(Action):
    first_ref: str = Field(description="Earlier experiment reference.")
    second_ref: str = Field(description="Later experiment reference.")


class ExperimentObservation(Observation):
    operation: str
    result: dict[str, Any] = Field(default_factory=dict)


def _request_id(
    conversation: "LocalConversation | None",
    action: Action,
    tool_name: str,
) -> str:
    if conversation is not None:
        for event in reversed(conversation.state.active_branch()):
            if not isinstance(event, ActionEvent) or event.tool_name != tool_name:
                continue
            if event.action is action:
                return f"openhands:{conversation.id}:{event.tool_call_id}"
        for event in reversed(conversation.state.active_branch()):
            if (
                isinstance(event, ActionEvent)
                and event.tool_name == tool_name
                and event.action == action
            ):
                return f"openhands:{conversation.id}:{event.tool_call_id}"
    return f"direct:{tool_name}:{uuid.uuid4()}"


def _content_for_evidence(
    service: ExperimentService,
    evidence: ExperimentEvidence,
) -> list[TextContent | ImageContent]:
    body = evidence.public_dict()
    content: list[TextContent | ImageContent] = [
        TextContent(text=json.dumps(body, sort_keys=True, indent=2))
    ]
    for handle in evidence.artifacts:
        if not handle.media_type.startswith("image/"):
            continue
        data = service.read_artifact(handle)
        encoded = base64.b64encode(data).decode("ascii")
        content.append(
            ImageContent(
                image_urls=[f"data:{handle.media_type};base64,{encoded}"],
            )
        )
    return content


def _error(operation: str, exc: Exception) -> ExperimentObservation:
    return ExperimentObservation.from_text(
        f"{type(exc).__name__}: {exc}",
        is_error=True,
        operation=operation,
        result={},
    )


class ObserveExecutor(ToolExecutor[ObserveAction, ExperimentObservation]):
    def __init__(self, service: ExperimentService):
        self.service = service

    def __call__(
        self,
        action: ObserveAction,
        conversation: "LocalConversation | None" = None,
    ) -> ExperimentObservation:
        try:
            evidence = self.service.observe(
                request_id=_request_id(conversation, action, "observe")
            )
            return ExperimentObservation(
                operation="observe",
                result=evidence.public_dict(),
                content=_content_for_evidence(self.service, evidence),
            )
        except (ProtocolError, CorruptStore) as exc:
            return _error("observe", exc)


class RunControllerExecutor(
    ToolExecutor[RunControllerAction, ExperimentObservation]
):
    def __init__(self, service: ExperimentService, controller_path: str | Path, asset_library=None):
        self.service = service
        self.controller_path = Path(controller_path).resolve()
        self.asset_library = asset_library

    def __call__(
        self,
        action: RunControllerAction,
        conversation: "LocalConversation | None" = None,
    ) -> ExperimentObservation:
        try:
            if action.assets_used:
                if self.asset_library is None: raise ProtocolError("asset provenance is unavailable")
                session_id = str(conversation.id) if conversation is not None else None
                unread = [item for item in action.assets_used
                    if not self.asset_library.was_read(item, session_id=session_id)]
                if unread: raise ProtocolError(f"assets_used were not read: {unread}")
            evidence = self.service.run_controller(
                request_id=_request_id(conversation, action, "run_controller"),
                controller_path=self.controller_path,
                intent=action.intent,
                assets_used=action.assets_used,
            )
            return ExperimentObservation(
                operation="run_controller",
                result=evidence.public_dict(),
                content=_content_for_evidence(self.service, evidence),
            )
        except (ProtocolError, CorruptStore) as exc:
            return _error("run_controller", exc)


class InspectTrialExecutor(ToolExecutor[InspectTrialAction, ExperimentObservation]):
    def __init__(self, service: ExperimentService):
        self.service = service

    def __call__(
        self,
        action: InspectTrialAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ExperimentObservation:
        try:
            ref = action.ref or self.service.status()["latest_evidence"]
            if not ref:
                raise ProtocolError("no experiment is available to inspect")
            evidence = self.service.inspect_trial(ref)
            return ExperimentObservation(
                operation="inspect_trial",
                result=evidence.public_dict(),
                content=_content_for_evidence(self.service, evidence),
            )
        except (ProtocolError, CorruptStore) as exc:
            return _error("inspect_trial", exc)


class CompareTrialsExecutor(ToolExecutor[CompareTrialsAction, ExperimentObservation]):
    def __init__(self, service: ExperimentService):
        self.service = service

    def __call__(
        self,
        action: CompareTrialsAction,
        conversation: "LocalConversation | None" = None,  # noqa: ARG002
    ) -> ExperimentObservation:
        try:
            comparison = self.service.compare_trials(
                action.first_ref,
                action.second_ref,
            )
            return ExperimentObservation.from_text(
                json.dumps(comparison, sort_keys=True, indent=2),
                operation="compare_trials",
                result=comparison,
            )
        except (ProtocolError, CorruptStore) as exc:
            return _error("compare_trials", exc)


class ObserveTool(ToolDefinition[ObserveAction, ExperimentObservation]):
    name = "observe"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=("embodied:adapter",), declared=True)

    @classmethod
    def create(cls, service: ExperimentService, **_: Any) -> Sequence["ObserveTool"]:
        return [
            cls(
                description=(
                    "Observe the current public embodied state. This is a bounded, "
                    "read-only diagnostic and does not consume a physical trial."
                ),
                action_type=ObserveAction,
                observation_type=ExperimentObservation,
                annotations=ToolAnnotations(
                    title="observe",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=ObserveExecutor(service),
            )
        ]


class RunControllerTool(
    ToolDefinition[RunControllerAction, ExperimentObservation]
):
    name = "run_controller"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=("embodied:adapter",), declared=True)

    @classmethod
    def create(
        cls,
        service: ExperimentService,
        controller_path: str | Path,
        asset_library=None,
        **_: Any,
    ) -> Sequence["RunControllerTool"]:
        return [
            cls(
                description=(
                    "Run the current Controller as one physical experiment. The "
                    "Harness snapshots it, consumes one trial, resets to S0, executes "
                    "through the physical safety boundary, and returns durable public "
                    "evidence. Use observe for read-only diagnosis."
                ),
                action_type=RunControllerAction,
                observation_type=ExperimentObservation,
                annotations=ToolAnnotations(
                    title="run_controller",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=RunControllerExecutor(service, controller_path, asset_library),
            )
        ]


class InspectTrialTool(
    ToolDefinition[InspectTrialAction, ExperimentObservation]
):
    name = "inspect_trial"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=("embodied:evidence",), declared=True)

    @classmethod
    def create(
        cls,
        service: ExperimentService,
        **_: Any,
    ) -> Sequence["InspectTrialTool"]:
        return [
            cls(
                description=(
                    "Inspect one committed diagnostic or physical experiment as a "
                    "coherent public Controller/state/action/verifier/artifact view."
                ),
                action_type=InspectTrialAction,
                observation_type=ExperimentObservation,
                annotations=ToolAnnotations(
                    title="inspect_trial",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=InspectTrialExecutor(service),
            )
        ]


class CompareTrialsTool(
    ToolDefinition[CompareTrialsAction, ExperimentObservation]
):
    name = "compare_trials"

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=("embodied:evidence",), declared=True)

    @classmethod
    def create(
        cls,
        service: ExperimentService,
        **_: Any,
    ) -> Sequence["CompareTrialsTool"]:
        return [
            cls(
                description=(
                    "Compare two committed experiments and return aligned factual "
                    "differences. It provides no task strategy or recommendation."
                ),
                action_type=CompareTrialsAction,
                observation_type=ExperimentObservation,
                annotations=ToolAnnotations(
                    title="compare_trials",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=CompareTrialsExecutor(service),
            )
        ]


def create_embodied_tools(
    service: ExperimentService,
    controller_path: str | Path,
    asset_library=None,
) -> list[ToolDefinition[Any, Any]]:
    return [
        *ObserveTool.create(service),
        *RunControllerTool.create(service, controller_path, asset_library=asset_library),
        *InspectTrialTool.create(service),
        *CompareTrialsTool.create(service),
    ]
