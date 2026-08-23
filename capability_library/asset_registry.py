"""Structured, append-only Embodied Capability Library.

The registry stores executable tools, procedural skills, models, and
experiences together with provenance and cross-task reuse evidence. It never
stores evaluator-only state as an action input; benchmark outcomes are kept as
auditable evidence attached to an asset, not as hidden policy inputs.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


LIFECYCLE = (
    "discovered",
    "provenance_checked",
    "installed_or_implemented",
    "unit_tested",
    "development_validated",
    "cross_task_reused",
    "frozen_candidate",
)


def _now() -> float:
    return time.time()


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "assets": [], "events": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("asset library must contain a JSON object")
    payload.setdefault("schema_version", 1)
    payload.setdefault("assets", [])
    payload.setdefault("events", [])
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_asset_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return schema errors; no network or benchmark state is consulted."""
    errors: list[str] = []
    for key in ("asset_id", "kind", "name", "status"):
        if not str(manifest.get(key, "")).strip():
            errors.append(f"missing {key}")
    if manifest.get("status") not in set(LIFECYCLE) | {"rejected", "invalidated", "pending"}:
        errors.append("invalid status")
    if not isinstance(manifest.get("source_urls", []), list):
        errors.append("source_urls must be a list")
    if not isinstance(manifest.get("tested_tasks", []), list):
        errors.append("tested_tasks must be a list")
    if not isinstance(manifest.get("reused_tasks", []), list):
        errors.append("reused_tasks must be a list")
    if manifest.get("current_task_data_used") is True:
        errors.append("current_task_data_used must be false for primary assets")
    if manifest.get("privileged_state_used") is True:
        errors.append("privileged_state_used must be false for primary assets")
    return errors


def asset_id_for(kind: str, name: str, version: str = "1") -> str:
    slug = "-".join(str(name).casefold().split())
    return f"{kind}.{slug}.v{version}"


def register_asset(
    manifest: Mapping[str, Any],
    *,
    library_path: str = "capability_library/library.json",
    event: str = "asset_registered",
) -> dict[str, Any]:
    """Insert or update one asset and append a lifecycle event."""
    candidate = dict(manifest)
    candidate.setdefault("source_urls", [])
    candidate.setdefault("tested_tasks", [])
    candidate.setdefault("reused_tasks", [])
    candidate.setdefault("known_failures", [])
    candidate.setdefault("current_task_data_used", False)
    candidate.setdefault("privileged_state_used", False)
    candidate.setdefault("created_at_unix", _now())
    errors = validate_asset_manifest(candidate)
    if errors:
        return {"success": False, "errors": errors}
    path = Path(library_path)
    payload = _read(path)
    assets = [item for item in payload["assets"] if item.get("asset_id") != candidate["asset_id"]]
    assets.append(candidate)
    event_record = {
        "event": event,
        "asset_id": candidate["asset_id"],
        "status": candidate["status"],
        "timestamp_unix": _now(),
        "tested_tasks": candidate["tested_tasks"],
        "reused_tasks": candidate["reused_tasks"],
    }
    payload["assets"] = sorted(assets, key=lambda item: str(item["asset_id"]))
    payload["events"].append(event_record)
    _write(path, payload)
    return {"success": True, "asset_id": candidate["asset_id"], "event": event_record}


def record_asset_reuse(
    asset_id: str,
    task_id: str,
    *,
    outcome: str,
    evidence: str = "",
    library_path: str = "capability_library/library.json",
) -> dict[str, Any]:
    """Record cross-task reuse without modifying the asset's provenance."""
    path = Path(library_path)
    payload = _read(path)
    matches = [item for item in payload["assets"] if item.get("asset_id") == asset_id]
    if not matches:
        return {"success": False, "reason": f"unknown asset: {asset_id}"}
    asset = matches[0]
    reused = list(asset.get("reused_tasks", []))
    if task_id not in reused:
        reused.append(task_id)
    asset["reused_tasks"] = sorted(reused)
    if asset.get("status") == "development_validated":
        asset["status"] = "cross_task_reused"
    payload["events"].append({
        "event": "asset_reused",
        "asset_id": asset_id,
        "task_id": task_id,
        "outcome": str(outcome),
        "evidence": str(evidence),
        "timestamp_unix": _now(),
    })
    _write(path, payload)
    return {"success": True, "asset_id": asset_id, "task_id": task_id, "outcome": outcome}


def find_assets(
    *,
    kind: str | None = None,
    sensor: str | None = None,
    task_id: str | None = None,
    library_path: str = "capability_library/library.json",
) -> list[dict[str, Any]]:
    """Retrieve reusable assets by declared interface metadata."""
    payload = _read(Path(library_path))
    found: list[dict[str, Any]] = []
    for asset in payload["assets"]:
        if kind and asset.get("kind") != kind:
            continue
        if sensor and sensor not in asset.get("sensors", []):
            continue
        if task_id and task_id not in asset.get("tested_tasks", []) + asset.get("reused_tasks", []):
            continue
        found.append(dict(asset))
    return found


__all__ = ["LIFECYCLE", "asset_id_for", "find_assets", "record_asset_reuse", "register_asset", "validate_asset_manifest"]
