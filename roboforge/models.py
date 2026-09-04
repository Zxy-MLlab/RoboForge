from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ExecutionKind = Literal["diagnostic", "physical_trial"]


@dataclass(frozen=True)
class RawArtifact:
    name: str
    media_type: str
    data: bytes


@dataclass(frozen=True)
class ArtifactHandle:
    uri: str
    sha256: str
    media_type: str
    name: str
    size_bytes: int


@dataclass(frozen=True)
class AdapterResult:
    public: dict[str, Any]
    artifacts: tuple[RawArtifact, ...] = ()
    private_receipt: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentEvidence:
    schema_version: int
    ref: str
    execution_kind: ExecutionKind
    request_id: str
    diagnostic_index: int | None
    physical_trial_index: int | None
    environment_generation: str | None
    controller_sha256: str | None
    intent: str | None
    public: dict[str, Any]
    assets_used: tuple[str, ...] = ()
    artifacts: tuple[ArtifactHandle, ...] = ()
    physical_verification: dict[str, bool] | None = None
    execution_error: str | None = None
    evidence_sha256: str = field(default="")
    candidate_bundle_digest: str | None = None

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["assets_used"] = list(self.assets_used)
        if value["physical_verification"] is None:
            value.pop("physical_verification")
        if value["physical_trial_index"] is None:
            value.pop("physical_trial_index")
        if value["diagnostic_index"] is None:
            value.pop("diagnostic_index")
        if value["environment_generation"] is None:
            value.pop("environment_generation")
        if value["controller_sha256"] is None:
            value.pop("controller_sha256")
        if value["intent"] is None:
            value.pop("intent")
        if value["execution_error"] is None:
            value.pop("execution_error")
        if value["candidate_bundle_digest"] is None:
            value.pop("candidate_bundle_digest")
        return value
