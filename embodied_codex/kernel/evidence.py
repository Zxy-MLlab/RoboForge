"""Explicit evidence views at the model/Harness trust boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentEvidence:
    """Bounded diagnostic evidence that may be sent to the coding model."""

    execution: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    evidence_ref: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {"execution": dict(self.execution),
                  "diagnostics": dict(self.diagnostics)}
        if self.evidence_ref:
            result["evidence_ref"] = self.evidence_ref
        return result

    @classmethod
    def from_execution(cls, execution: Mapping[str, Any],
                       diagnostics: Mapping[str, Any] | None = None,
                       evidence_ref: str | None = None):
        return cls(execution={"completed": execution.get("completed") is True,
                              "error": execution.get("error")},
                   diagnostics=dict(diagnostics or {}), evidence_ref=evidence_ref)


@dataclass(frozen=True)
class HarnessMetadata:
    """Receipt and recovery metadata retained by the Harness, never the model."""

    artifact_uri: str | None
    artifact_sha256: str | None
    controller_sha256: str | None
    execution_key: str | None
    environment_identity: Mapping[str, Any]
    verification_receipt: Mapping[str, Any]
    resume_token: str | None

    @classmethod
    def from_evidence(cls, evidence: Mapping[str, Any]):
        return cls(artifact_uri=evidence.get("artifact_uri"),
                   artifact_sha256=evidence.get("artifact_sha256"),
                   controller_sha256=evidence.get("controller_sha256"),
                   execution_key=evidence.get("execution_key"),
                   environment_identity=dict(evidence.get("environment_identity") or {}),
                   verification_receipt=dict(evidence.get("verification_receipt") or {}),
                   resume_token=evidence.get("resume_token"))

    def as_reference(self, *, summary: Mapping[str, Any]) -> dict[str, Any]:
        return {"artifact_uri": self.artifact_uri,
                "artifact_sha256": self.artifact_sha256,
                "controller_sha256": self.controller_sha256,
                "execution_key": self.execution_key,
                "environment_identity": dict(self.environment_identity),
                "verification_receipt": dict(self.verification_receipt),
                "resume_token": self.resume_token,
                "summary": dict(summary)}


__all__ = ["AgentEvidence", "HarnessMetadata"]
