"""Canonical LIBERO provider boundary.

This provider adapts the public deployment protocol without importing the
historical AgentLoop or registering any LLM tools.  It snapshots artifacts and
delegates only the explicit RobotDeployment methods needed by the harness.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from ...models import AdapterResult, RawArtifact


class LiberoProvider:
    observation_protocol = "canonical_embodied"

    def __init__(self, deployment: Any, runtime: Any):
        self.deployment = deployment
        self.runtime = runtime

    @property
    def instruction(self) -> str:
        """The task language exposed to the candidate runtime."""
        return str(getattr(self.deployment, "instruction", ""))

    def task_info(self) -> dict[str, Any]:
        """Return only the generated public SDK manual for the Agent."""
        value = getattr(self.deployment, "sdk_index", {})
        return {"instruction": self.instruction,
                "robot_interface": dict(value) if isinstance(value, dict) else {}}

    def dispatch(self, method: str, arguments: dict[str, Any]) -> Any:
        """Dispatch the finite public Robot protocol to the deployment."""
        if method not in {"observe", "act", "use", "verify", "check_observable_condition", "record", "sdk"}:
            raise RuntimeError(f"unsupported candidate operation: {method}")
        return self.deployment.dispatch(method, arguments)

    def project_rpc_output(self, method: str, arguments: dict[str, Any], result: Any) -> Any:
        return self.deployment.project_rpc_output(method, arguments, result)

    def canonical_embodied_state(self) -> Any:
        return self.deployment.canonical_embodied_state()

    def project_public_entities(self, tool_id: str, result: Any) -> Any:
        return self.deployment.project_public_entities(tool_id, result)

    def sdk_consequence(self, method: str) -> str:
        return self.deployment.sdk_consequence(method)

    def capability_consequence(self, tool_id: str) -> str:
        return self.deployment.capability_consequence(tool_id)

    def begin_execution(self, kind: str = "physical_trial") -> None:
        self.deployment.begin_execution(kind)

    def preflight(self, *, controller_path: Path, controller_sha256: str | None = None) -> dict[str, Any]:
        from ...preflight import preflight_controller
        del controller_sha256
        return preflight_controller(
            controller_path,
            capability_contracts=getattr(self.deployment, "capability_contracts", {}),
            sdk_contract=getattr(self.deployment, "robot_sdk_contract", {}),
        )

    def observe(self) -> AdapterResult:
        arguments = {"channel": "rgbd", "request": {}}
        value = self.deployment.dispatch("observe", arguments)
        public = self.deployment.project_rpc_output("observe", arguments, value)
        projected, artifacts = self._project(public)
        return AdapterResult(public=dict(projected), artifacts=artifacts)

    def _project(self, value: Any) -> tuple[Any, tuple[RawArtifact, ...]]:
        artifacts: list[RawArtifact] = []

        def visit(item: Any) -> Any:
            if isinstance(item, dict):
                result: dict[str, Any] = {}
                for key, child in item.items():
                    name = str(key)
                    if name.endswith("_path"):
                        if child is None:
                            continue
                        handle = str(child)
                        resolver = getattr(self.deployment, "resolve_controller_artifact", None)
                        source = resolver(handle) if handle.startswith("artifact://") and callable(resolver) else Path(handle)
                        if not source.is_absolute() or not source.is_file():
                            raise RuntimeError(f"public artifact is unavailable: {name}")
                        artifacts.append(RawArtifact(
                            name=source.name,
                            media_type=mimetypes.guess_type(source.name)[0] or "application/octet-stream",
                            data=source.read_bytes(),
                        ))
                        continue
                    result[name] = visit(child)
                return result
            if isinstance(item, list):
                return [visit(child) for child in item]
            if isinstance(item, str) and Path(item).is_absolute():
                raise RuntimeError("public evidence exposed an unlabelled host path")
            return item

        return visit(value), tuple(artifacts)

    def reset_to_s0(self) -> str:
        self.deployment.reset_case()
        identity = self.deployment.execution_identity()
        generation = identity.get("environment_generation")
        if not isinstance(generation, str) or not generation:
            raise RuntimeError("LIBERO reset did not return an environment generation")
        return generation

    def execute_controller(self, *, controller_path: Path, controller_sha256: str,
                           environment_generation: str,
                           candidate_bundle_digest: str | None = None,
                           candidate_source_root: Path | None = None) -> AdapterResult:
        execution = self.runtime.execute(
            controller_path, self, source_root=candidate_source_root
        )
        report = self.deployment.sensor_report(execution)
        public = self.deployment.agent_evidence(execution, report)
        receipt = self.deployment.verification_receipt(execution)
        receipt = {**dict(receipt), "candidate_bundle_digest": candidate_bundle_digest}
        projected, artifacts = self._project(public)
        return AdapterResult(public=dict(projected), artifacts=artifacts,
                             private_receipt=receipt)

    def validate_receipt(self, receipt: dict[str, Any], *, controller_sha256: str,
                         environment_generation: str,
                         candidate_bundle_digest: str | None = None) -> bool:
        identity = self.deployment.execution_identity()
        return bool(
            receipt.get("verified") is True
            and receipt.get("controller_sha256") == controller_sha256
            and receipt.get("environment_generation") == environment_generation
            and receipt.get("environment_identity") == identity
            and receipt.get("candidate_bundle_digest") == candidate_bundle_digest
        )

    def candidate_runtime_metadata(self) -> dict[str, Any]:
        value = getattr(self.deployment, "candidate_runtime_metadata", None)
        return dict(value() if callable(value) else value or {})


__all__ = ["LiberoProvider"]
