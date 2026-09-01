"""Agent-external lifecycle operations over immutable RoboForge records."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .assets import AssetLibrary
from .store import canonical_json


def environment_info() -> dict[str, Any]:
    try:
        import importlib.metadata
        openhands = importlib.metadata.version("openhands-sdk")
    except importlib.metadata.PackageNotFoundError:
        openhands = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "openhands_sdk": openhands,
        "runtime_providers": ["libero"],
        "control_plane": "roboforge",
    }


def load_evidence(reference: str | Path) -> dict[str, Any]:
    path = Path(reference).resolve()
    if path.is_dir():
        candidates = sorted((path / "evidence").glob("*.json"))
        if not candidates:
            candidates = sorted(path.glob("adapter-worker/service/evidence/*.json"))
        if not candidates: raise FileNotFoundError(f"no evidence in {path}")
        path = candidates[-1]
    value = json.loads(path.read_text(encoding="utf-8"))
    recorded = value.pop("evidence_sha256", None)
    actual = hashlib.sha256(canonical_json(value)).hexdigest()
    if recorded != actual: raise ValueError(f"evidence digest mismatch: {path}")
    value["evidence_sha256"] = recorded
    value["evidence_path"] = str(path)
    return value


def replay(reference: str | Path) -> dict[str, Any]:
    value = load_evidence(reference)
    return {"mode": "evidence_only", "physical_action_replayed": False, "evidence": value}


def compare(first: str | Path, second: str | Path) -> dict[str, Any]:
    left, right = load_evidence(first), load_evidence(second)
    ignored = {"evidence_path", "evidence_sha256", "ref", "request_id"}
    keys = sorted((set(left) | set(right)) - ignored)
    changes = [{"field": key, "baseline": left.get(key), "candidate": right.get(key)}
               for key in keys if left.get(key) != right.get(key)]
    return {"baseline": left["ref"], "candidate": right["ref"], "changes": changes,
            "paired_seed": (left.get("public") or {}).get("seed") ==
                           (right.get("public") or {}).get("seed")}


def submit(asset_root: str | Path, asset_id: str, evidence_paths: list[str],
           *, note: str) -> dict[str, Any]:
    evidence = [load_evidence(path) for path in evidence_paths]
    if not all((item.get("physical_verification") or {}).get("verified") is True
               for item in evidence):
        raise ValueError("promotion requires independently verified physical evidence")
    refs = [str(item["ref"]) for item in evidence]
    return AssetLibrary(asset_root).decide_capability(
        asset_id, decision="promoted", evidence=refs, note=note)
