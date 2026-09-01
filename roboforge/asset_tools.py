"""OpenHands tools for progressive embodied asset discovery and persistence."""
from __future__ import annotations
import json
from collections.abc import Sequence
from typing import Any, Literal
from pydantic import Field
from openhands.sdk.tool import Action, Observation, ToolAnnotations, ToolDefinition, ToolExecutor
from .assets import AssetLibrary
from .capability import CapabilityAcquirer

class AssetObservation(Observation):
    result: Any = None

class SearchAssetsAction(Action):
    query: str = Field(description="Semantic text to match against asset metadata.")
    asset_kind: Literal["experiences", "skills", "capabilities"] | None = None

class ReadAssetAction(Action):
    asset_id: str

class SaveAssetAction(Action):
    asset_kind: Literal["experiences", "skills", "capabilities"]
    name: str; purpose: str; description: str
    applicability: Any = None
    evidence: list[str] = Field(min_length=1, description="Committed experiment references supporting this asset.")
    usage: str = ""
    implementation: Any = None

class AcquireCapabilityAction(Action):
    source_path: str = Field(description=(
        "Path to a NEW .py capability source file inside the current OpenHands workspace. "
        "Do not pass an asset-library JSON path. For an existing capability:// asset, "
        "call read_asset and materialize_capability instead."
    ))
    name: str; purpose: str; description: str
    validation_command: list[str] = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)

class MaterializeCapabilityAction(Action):
    asset_id: str
    destination: str = Field(description="Workspace-relative .py destination.")

class SearchAssetsExecutor(ToolExecutor):
    def __init__(self, library): self.library = library
    def __call__(self, action, conversation=None):
        value = self.library.search(action.query, kind=action.asset_kind)
        return AssetObservation.from_text(json.dumps(value, indent=2), result=value)

class ReadAssetExecutor(ToolExecutor):
    def __init__(self, library): self.library = library
    def __call__(self, action, conversation=None):
        session_id = str(conversation.id) if conversation is not None else None
        try: value = self.library.read(action.asset_id, session_id=session_id)
        except KeyError as exc: return AssetObservation.from_text(str(exc), is_error=True, result=None)
        return AssetObservation.from_text(json.dumps(value, indent=2), result=value)

class SaveAssetExecutor(ToolExecutor):
    def __init__(self, library, service): self.library, self.service = library, service
    def __call__(self, action, conversation=None):
        try:
            for ref in action.evidence: self.service.inspect_trial(ref)
            value = self.library.register(action.asset_kind, name=action.name, purpose=action.purpose,
                description=action.description, applicability=action.applicability,
                evidence=action.evidence, provenance={"source": "openhands-conversation"},
                usage=action.usage, implementation=action.implementation)
        except Exception as exc: return AssetObservation.from_text(f"{type(exc).__name__}: {exc}", is_error=True, result=None)
        return AssetObservation.from_text(json.dumps(value, indent=2), result=value)

def create_asset_tools(library: AssetLibrary, service, workspace: str | None = None) -> list[ToolDefinition[Any, Any]]:
    class SearchAssetsTool(ToolDefinition[SearchAssetsAction, AssetObservation]):
        name = "search_assets"
        @classmethod
        def create(cls, **_: Any) -> Sequence["SearchAssetsTool"]: return []
    class ReadAssetTool(ToolDefinition[ReadAssetAction, AssetObservation]):
        name = "read_asset"
        @classmethod
        def create(cls, **_: Any) -> Sequence["ReadAssetTool"]: return []
    class SaveAssetTool(ToolDefinition[SaveAssetAction, AssetObservation]):
        name = "save_asset"
        @classmethod
        def create(cls, **_: Any) -> Sequence["SaveAssetTool"]: return []
    class AcquireCapabilityTool(ToolDefinition[AcquireCapabilityAction, AssetObservation]):
        name = "acquire_capability"
        @classmethod
        def create(cls, **_: Any) -> Sequence["AcquireCapabilityTool"]: return []
    class MaterializeCapabilityTool(ToolDefinition[MaterializeCapabilityAction, AssetObservation]):
        name = "materialize_capability"
        @classmethod
        def create(cls, **_: Any) -> Sequence["MaterializeCapabilityTool"]: return []
    class AcquireExecutor(ToolExecutor):
        def __init__(self, acquirer): self.acquirer = acquirer
        def __call__(self, action, conversation=None):
            try: value = self.acquirer.acquire(source_path=action.source_path, name=action.name,
                purpose=action.purpose, description=action.description,
                validation_command=action.validation_command, evidence=action.evidence,
                provenance={"source": "agent-acquisition"})
            except Exception as exc: return AssetObservation.from_text(f"{type(exc).__name__}: {exc}", is_error=True, result=None)
            return AssetObservation.from_text(json.dumps(value, indent=2), result=value)
    specs = [(SearchAssetsTool, SearchAssetsAction, SearchAssetsExecutor(library), True, "Search only compact asset metadata."),
             (ReadAssetTool, ReadAssetAction, ReadAssetExecutor(library), True, "Read one selected asset in detail."),
             (SaveAssetTool, SaveAssetAction, SaveAssetExecutor(library, service), False, "Persist an evidence-backed embodied asset.")]
    if workspace:
        acquirer = CapabilityAcquirer(workspace, library)
        specs.append((AcquireCapabilityTool, AcquireCapabilityAction,
            AcquireExecutor(acquirer), False,
            "Validate and register a NEW workspace-local Python capability with immutable provenance. "
            "Existing capability:// assets must be read then materialized, not reacquired from their JSON files."))
        class MaterializeExecutor(ToolExecutor):
            def __call__(self, action, conversation=None):
                session_id = str(conversation.id) if conversation is not None else None
                try: value = acquirer.materialize(action.asset_id, action.destination, session_id=session_id)
                except Exception as exc: return AssetObservation.from_text(f"{type(exc).__name__}: {exc}", is_error=True, result=None)
                return AssetObservation.from_text(json.dumps(value, indent=2), result=value)
        specs.append((MaterializeCapabilityTool, MaterializeCapabilityAction,
            MaterializeExecutor(), False,
            "Materialize one previously read, digest-verified Capability into the Controller workspace."))
    return [cls(description=desc, action_type=action, observation_type=AssetObservation,
        annotations=ToolAnnotations(title=cls.name, readOnlyHint=ro, destructiveHint=not ro,
            idempotentHint=ro, openWorldHint=False), executor=executor) for cls, action, executor, ro, desc in specs]
