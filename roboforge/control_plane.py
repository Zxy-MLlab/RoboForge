"""Agent-external lifecycle operations over immutable RoboForge records."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import os, socket
from pathlib import Path
from typing import Any

from .assets import AssetLibrary
from .store import canonical_json
from .trust import verify_receipt


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
    del asset_root, evaluator_key, require_trusted_mode
    socket_path=os.environ.get("ROBOFORGE_PROMOTION_SOCKET")
    token=os.environ.get("ROBOFORGE_PROMOTION_TOKEN")
    if not socket_path or not token: raise RuntimeError("external promotion service is not configured")
    request={"token":token,"asset_id":asset_id,"evidence":evidence_paths,"note":note}
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as connection:
        connection.connect(socket_path); connection.sendall((json.dumps(request)+"\n").encode())
        response=json.loads(connection.makefile("rb").readline())
    if not response.get("ok"): raise PermissionError(response.get("error","promotion failed"))
    return dict(response["result"])
