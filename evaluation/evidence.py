"""Evaluator-only access to immutable execution evidence artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvaluatorEvidence:
    artifact_uri: str
    artifact_sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def load(cls, run_root: str | Path, reference: Mapping[str, Any]):
        uri = str(reference.get("artifact_uri") or "")
        if not uri.startswith("run://"):
            raise ValueError("Evaluator evidence requires a run:// artifact")
        root = Path(run_root).resolve()
        relative = Path(uri.removeprefix("run://"))
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in path.parents:
            raise ValueError("Evaluator evidence escapes the run root")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != reference.get("artifact_sha256"):
            raise ValueError("Evaluator evidence hash mismatch")
        payload = json.loads(data)
        if not isinstance(payload, Mapping):
            raise ValueError("Evaluator evidence must be an object")
        return cls(uri, digest, dict(payload))


__all__ = ["EvaluatorEvidence"]
