"""OpenHands public Stop-hook gate for an active robot development session.

This module is deliberately not a verifier.  It only prevents a coding-agent
conversation from treating a prose acknowledgement as completion while the
external experiment service still has no successful sealed receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .models import ExperimentEvidence
from .service import ExperimentService, ProtocolError


def write_tool_activity(workspace: str | Path, count: int = 0) -> Path:
    """Persist non-secret session progress consumed by the public Stop hook."""
    root = Path(workspace).resolve()
    target = (root / ".roboforge" / "tool-activity.json").resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("tool activity escaped workspace") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    current = 0
    if target.is_file():
        try:
            current = int(json.loads(target.read_text()).get("tool_calls", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            current = 0
    payload = {"schema_version": 1, "tool_calls": max(current, int(count))}
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)
    return target


def execution_task_stop_decision(activity_path: str | Path, workspace: str | Path) -> dict[str, Any]:
    """Gate Finish using only public workspace progress and campaign state.

    A failed physical trial is useful feedback, but it is not completion.  The
    conversation must continue while budget remains; only a verified receipt
    or an explicitly exhausted physical-trial budget permits Finish.
    """
    try:
        activity = json.loads(Path(activity_path).read_text(encoding="utf-8"))
        count = int(activity.get("tool_calls", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        count = 0
    root = Path(workspace).resolve()
    trial_root = (root / ".roboforge" / "trials").resolve()
    trial_root.relative_to(root)
    records = []
    if trial_root.is_dir():
        for result_path in sorted(trial_root.glob("*/result.json")):
            try:
                value = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict): records.append((result_path, value))
    physical = [item for item in records if item[1].get("physical_trial_consumed") is not False
                and not str(item[1].get("trial_id", "")).startswith("preflight-")]
    status_path = root / ".roboforge" / "campaign-status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        status = {}
    if status.get("latest_verified") is True:
        return {"decision": "allow", "reason": "latest sealed physical receipt is verified"}
    try:
        trials = int(status.get("physical_trials", len(physical)))
        maximum = int(status.get("max_physical_trials", 0))
    except (TypeError, ValueError):
        trials, maximum = len(physical), 0
    if maximum > 0 and trials >= maximum:
        return {"decision": "allow", "reason": "physical trial budget is exhausted"}
    if physical:
        directory = physical[-1][0].parent
        return {"decision": "deny", "reason": "latest physical trial is not verified",
                "additionalContext": (
                    f"Read {directory / 'result.json'}, {directory / 'first_error.json'} and "
                    f"{directory / 'trace.json'}, then continue the same conversation with a "
                    "new hypothesis and ordinary Terminal trial."
                )}
    if records:
        directory = records[-1][0].parent
        return {"decision": "deny", "reason": "only a non-physical trial result exists",
                "additionalContext": (
                    f"Read {directory / 'first_error.json'} and {directory / 'trace.json'}, "
                    "fix the public contract error and rerun the ordinary Terminal trial command."
                )}
    if count > 0:
        return {"decision": "deny", "reason": "tools were used but no local trial result exists",
                "additionalContext": (
                    "Run the current Controller through `python -m roboforge trial ...` in "
                    "Terminal, then inspect the materialized result before finishing."
                )}
    return {
        "decision": "deny",
        "reason": "execution task has no tool activity",
        "additionalContext": (
            "This is an execution task. Use an available public OpenHands coding "
            "tool to inspect the workspace and run the Controller through the ordinary Terminal "
            "trial CLI before deciding whether to finish."
        ),
    }


def campaign_stop_decision(
    ledger_path: str | Path,
    controller_path: str | Path,
    baseline_digest: str,
) -> dict[str, Any]:
    """Reject premature completion without prescribing the next agent action."""
    try:
        ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
        records = [item for item in ledger.get("records", []) if isinstance(item, dict)]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        records = []
    try:
        current = hashlib.sha256(Path(controller_path).read_bytes()).hexdigest()
    except OSError:
        current = ""
    candidate_trials = [
        item for item in records
        if item.get("controller_sha256") == current and item.get("valid_trial") is True
    ]
    if current and current != baseline_digest and candidate_trials:
        return {
            "decision": "allow",
            "reason": "a changed candidate has completed a valid development trial",
        }
    if current == baseline_digest:
        context = (
            "The Controller still matches the initial baseline. Inspect public trial evidence, "
            "make a justified workspace change, and test the changed candidate with the ordinary CLI."
        )
    elif not candidate_trials:
        context = (
            "The Controller has changed, but this exact source digest has no valid development trial. "
            "Run it on an allowed development state and inspect the public result before finishing."
        )
    else:
        context = "Continue autonomous development using public workspace tools."
    return {"decision": "deny", "reason": "development completion evidence is incomplete",
            "additionalContext": context}


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
        "physical_attempts": int(status.get("physical_attempts", trials)),
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status")
    group.add_argument("--tool-activity")
    group.add_argument("--campaign-ledger")
    parser.add_argument("--workspace")
    parser.add_argument("--controller")
    parser.add_argument("--baseline-digest")
    args = parser.parse_args(argv)
    if args.tool_activity and not args.workspace:
        parser.error("--workspace is required with --tool-activity")
    if args.campaign_ledger:
        if not args.controller or not args.baseline_digest:
            parser.error("--controller and --baseline-digest are required with --campaign-ledger")
        decision = campaign_stop_decision(args.campaign_ledger, args.controller, args.baseline_digest)
    else:
        decision = (execution_task_stop_decision(args.tool_activity, args.workspace)
                    if args.tool_activity else stop_decision(args.status))
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
