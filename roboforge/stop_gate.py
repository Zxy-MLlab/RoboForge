"""OpenHands public Stop-hook gate for an active robot development session.

This module is deliberately not a verifier.  It only prevents a coding-agent
conversation from treating a prose acknowledgement as completion while the
external experiment service still has no successful sealed receipt.
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from .models import ExperimentEvidence
from .service import ExperimentService, ProtocolError


def write_public_status(
    service: ExperimentService,
    workspace: str | Path,
    evidence: ExperimentEvidence | None = None,
) -> Path:
    """Atomically expose the minimal non-privileged completion state."""
    root = Path(workspace).resolve()
    status = service.status()
    latest_ref = status.get("latest_physical_evidence")
    verification = None
    if evidence is not None and evidence.ref == latest_ref:
        verification = evidence.physical_verification
    elif latest_ref:
        verification = service.inspect_trial(str(latest_ref)).physical_verification
    trials = int(status.get("physical_trials", 0))
    maximum = int(status.get("max_trials", 0))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "physical_trials": trials,
        "max_physical_trials": maximum,
        "remaining_physical_trials": max(0, maximum - trials),
        "latest_physical_evidence": latest_ref,
        "latest_verified": bool((verification or {}).get("verified") is True),
    }
    target = (root / ".roboforge" / "campaign-status.json").resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("campaign status escaped workspace") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def stop_decision(status_path: str | Path) -> dict[str, Any]:
    """Return an OpenHands Stop-hook decision from public service state."""
    path = Path(status_path)
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "decision": "deny",
            "reason": f"robot campaign status is unavailable: {type(exc).__name__}",
            "additionalContext": (
                "The robot task is not complete. Use an available coding or "
                "robot-development tool now; do not respond with acknowledgement "
                "or planning text alone."
            ),
        }
    if status.get("latest_verified") is True:
        return {"decision": "allow", "reason": "latest sealed physical receipt is verified"}
    if int(status.get("physical_trials", 0)) >= int(status.get("max_physical_trials", 0)):
        return {"decision": "allow", "reason": "physical trial budget is exhausted"}
    latest = status.get("latest_physical_evidence")
    context = (
        "The robot task has no verified sealed receipt. Continue the same "
        "OpenHands conversation and independently choose a concrete Editor, "
        "Terminal, diagnostic, or physical-trial action."
    )
    if latest:
        context += f" Inspect the public artifacts for {latest} before revising code."
    return {
        "decision": "deny",
        "reason": "robot task is not yet verified and physical budget remains",
        "additionalContext": context,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(stop_decision(args.status), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
