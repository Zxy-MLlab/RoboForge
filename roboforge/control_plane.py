"""Agent-external lifecycle operations over immutable RoboForge records."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import os
from pathlib import Path
from typing import Any

from .assets import AssetLibrary
from .store import canonical_json
from .trust import verify_receipt

def trusted_mode_available() -> bool:
    """Return whether an externally isolated evaluator domain is configured."""
    return os.environ.get("ROBOFORGE_TRUSTED_MODE", "").lower() in {"1", "true", "yes"}


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
    lp, rp = left.get("public") or {}, right.get("public") or {}
    # Missing identity fields are unknown, never an accidental paired match.
    fields = ("task", "initial_state", "seed", "provider_version", "environment_version",
              "eval_protocol", "episode_budget")
    pairing = {field: (lp.get(field), rp.get(field),
                       lp.get(field) is not None and rp.get(field) is not None and lp.get(field) == rp.get(field))
               for field in fields}
    paired = all(item[2] for item in pairing.values())
    return {"baseline": left["ref"], "candidate": right["ref"], "changes": changes,
            "paired": paired, "paired_seed": pairing["seed"][2] if pairing["seed"][0] is not None and pairing["seed"][1] is not None else "unknown",
            "pairing": {key: ("match" if value[2] else "mismatch" if value[0] is not None and value[1] is not None else "unknown") for key, value in pairing.items()}}


def submit(asset_root: str | Path, asset_id: str, evidence_paths: list[str],
           *, note: str, evaluator_key: bytes | None = None,
           require_trusted_mode: bool = False) -> dict[str, Any]:
    evidence = [load_evidence(path) for path in evidence_paths]
    if not all((item.get("physical_verification") or {}).get("verified") is True
               for item in evidence):
        raise ValueError("promotion requires independently verified physical evidence")
    if evaluator_key is None:
        raise ValueError("promotion requires an evaluator-only receipt key")
    if require_trusted_mode and not trusted_mode_available():
        raise ValueError("trusted promotion unavailable: evaluator isolation is not configured")
    for item in evidence:
        receipt = item.get("sealed_receipt")
        if not verify_receipt(receipt, evaluator_key):
            raise ValueError("invalid or expired evaluator receipt")
        if receipt.get("trial_id") != item.get("ref"):
            raise ValueError("receipt trial binding mismatch")
        if asset_id not in item.get("assets_used", []):
            raise ValueError("receipt evidence does not prove capability use")
    refs = [str(item["ref"]) for item in evidence]
    return AssetLibrary(asset_root).decide_capability(
        asset_id, decision="promoted", evidence=refs, note=note)
