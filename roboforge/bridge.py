"""Legacy compatibility bridge for historical evidence only.

Canonical RPC execution uses :mod:`roboforge.providers.libero` and does not
import this module.  It remains available solely for old evidence readers.
"""
from __future__ import annotations

from pathlib import Path
import mimetypes
from typing import Any, Protocol

from .models import AdapterResult, RawArtifact


class LegacyAdapter(Protocol):
    """The small public subset exposed by a frozen v1 deployment."""

    def begin_execution(self, kind: str = "physical_trial") -> None: ...

    def dispatch(self, method: str, arguments: dict[str, Any]) -> Any: ...

    def reset_case(self) -> Any: ...

    def execution_identity(self) -> dict[str, Any]: ...

    def sensor_report(self, execution: dict[str, Any]) -> dict[str, Any]: ...

    def agent_evidence(
        self, execution: dict[str, Any], sensor_report: dict[str, Any]
    ) -> dict[str, Any]: ...

    def verification_receipt(self, execution: dict[str, Any]) -> dict[str, Any]: ...


class LegacyAdapterBridge:
    """Adapt v1's public deployment hooks without importing task code.

    The bridge deliberately treats the legacy deployment as the authority for
    controller execution, public projection, and receipt creation.  It does not
    expose raw execution payloads or host artifact paths to the spike service.
    """

    observation_protocol = "canonical_embodied"

    def __init__(self, legacy: LegacyAdapter, runtime: Any):
        self.legacy = legacy
        self.runtime = runtime

    def begin_execution(self, kind: str) -> None:
        self.legacy.begin_execution(kind)

    def preflight(
        self,
        *,
        controller_path: Path,
        controller_sha256: str | None = None,
    ) -> dict[str, Any]:
        from .preflight import preflight_controller

        del controller_sha256
        return preflight_controller(
            controller_path,
            capability_contracts=getattr(self.legacy, "capability_contracts", {}),
            sdk_contract=getattr(self.legacy, "robot_sdk_contract", {}),
        )

    def observe(self) -> AdapterResult:
        value = self.legacy.dispatch("observe", {"channel": "rgbd", "request": {}})
        public = self.legacy.project_rpc_output(
            "observe", {"channel": "rgbd", "request": {}}, value
        ) if hasattr(self.legacy, "project_rpc_output") else value
        if not isinstance(public, dict):
            raise TypeError("legacy observe projection must be a mapping")
        projected, artifacts = self._project_evidence(public)
        return AdapterResult(public=dict(projected), artifacts=artifacts)

    def _project_evidence(self, value: Any) -> tuple[Any, tuple[RawArtifact, ...]]:
        """Remove legacy paths while snapshotting files as opaque artifacts."""
        artifacts: list[RawArtifact] = []

        def visit(item: Any, key: str = "") -> Any:
            if isinstance(item, dict):
                result = {}
                for child_key, child in item.items():
                    name = str(child_key)
                    if name.endswith("_path"):
                        if child is None:
                            continue
                        handle = str(child)
                        resolver = getattr(self.legacy, "resolve_controller_artifact", None)
                        if handle.startswith("artifact://") and callable(resolver):
                            source = resolver(handle)
                        else:
                            source = Path(handle)
                        if not source.is_absolute() or not source.is_file():
                            raise RuntimeError(f"legacy artifact path is unavailable: {name}")
                        data = source.read_bytes()
                        media_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
                        artifacts.append(RawArtifact(name=source.name, media_type=media_type, data=data))
                        continue
                    result[name] = visit(child, name)
                return result
            if isinstance(item, list):
                return [visit(child, key) for child in item]
            if isinstance(item, str) and Path(item).is_absolute():
                raise RuntimeError(f"legacy evidence exposes an unlabelled host path: {key}")
            return item

        return visit(value), tuple(artifacts)

    def reset_to_s0(self) -> str:
        self.legacy.reset_case()
        identity = self.legacy.execution_identity()
        generation = identity.get("environment_generation")
        if not isinstance(generation, str) or not generation:
            raise RuntimeError("legacy Adapter reset did not provide a generation")
        return generation

    def execute_controller(
        self,
        *,
        controller_path: Path,
        controller_sha256: str,
        environment_generation: str,
        candidate_bundle_digest: str | None = None,
        candidate_source_root: Path | None = None,
    ) -> AdapterResult:
        execution = self.runtime.execute(
            controller_path, self.legacy, source_root=candidate_source_root
        )
        if not isinstance(execution, dict):
            raise TypeError("legacy Controller runtime must return a mapping")
        report = self.legacy.sensor_report(execution)
        public = self.legacy.agent_evidence(execution, report)
        if not isinstance(public, dict):
            raise TypeError("legacy public evidence must be a mapping")
        receipt = self.legacy.verification_receipt(execution)
        if not isinstance(receipt, dict):
            raise TypeError("legacy verification receipt must be a mapping")
        # The service performs the final binding check. These values are kept
        # private in AdapterResult and never enter the public projection.
        receipt = {**receipt, "candidate_bundle_digest": candidate_bundle_digest}
        projected, artifacts = self._project_evidence(public)
        return AdapterResult(
            public=dict(projected), artifacts=artifacts, private_receipt=dict(receipt)
        )

    def validate_receipt(
        self,
        receipt: dict[str, Any],
        *,
        controller_sha256: str,
        environment_generation: str,
        candidate_bundle_digest: str | None = None,
    ) -> bool:
        identity = self.legacy.execution_identity()
        return bool(
            receipt.get("verified") is True
            and receipt.get("controller_sha256") == controller_sha256
            and receipt.get("environment_generation") == environment_generation
            and receipt.get("environment_identity") == identity
            and receipt.get("candidate_bundle_digest") == candidate_bundle_digest
        )

    def candidate_runtime_metadata(self) -> dict[str, Any]:
        value = getattr(self.legacy, "candidate_runtime_metadata", None)
        return dict(value() if callable(value) else value or {})


__all__ = ["LegacyAdapter", "LegacyAdapterBridge"]
