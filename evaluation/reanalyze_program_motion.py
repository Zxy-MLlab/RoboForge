"""Create an immutable sensor-only motion audit for an existing program run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "capability_library")]

from libero_robot_sdk import summarize_motion_outcome


def reanalyze_rpc_motion(run_report: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct per-command convergence from legal RPC arguments/results."""
    last_eef = None
    outcomes = []
    for event in run_report.get("rpc_events") or ():
        if not isinstance(event, Mapping):
            continue
        result = event.get("result") or {}
        if event.get("method") == "act" and isinstance(result, Mapping):
            action = (event.get("arguments") or {}).get("action") or {}
            target = action.get("target_eef_xyz")
            after = result.get("eef_xyz")
            if last_eef is not None and target is not None and after is not None:
                outcome = summarize_motion_outcome(
                    target, last_eef, after,
                    requested_repeat=int(action.get("repeat", 1)),
                )
                outcome.update({
                    "command_index": len(outcomes) + 1,
                    "rpc_id": event.get("id"),
                    "step_after": result.get("step"),
                    "gripper": action.get("gripper"),
                    "orientation": (
                        action.get("orientation")
                        if isinstance(action.get("orientation"), str)
                        else "quaternion"
                    ),
                    "position_gain": action.get("position_gain"),
                    "max_translation_action": action.get("max_translation_action"),
                })
                outcomes.append(outcome)
        if isinstance(result, Mapping) and result.get("eef_xyz") is not None:
            last_eef = result["eef_xyz"]

    return {
        "protocol": "sensor-only-motion-reanalysis-v1",
        "action_outcomes": outcomes,
        "control_diagnostics": {
            "commands": len(outcomes),
            "targets_reached": sum(bool(item["reached_target"]) for item in outcomes),
            "targets_not_reached": sum(not bool(item["reached_target"]) for item in outcomes),
            "stalled_commands": sum(bool(item["stalled"]) for item in outcomes),
            "max_final_error_m": max(
                (float(item["final_error_m"]) for item in outcomes
                 if item.get("final_error_m") is not None),
                default=None,
            ),
        },
        "source": "legal controller RPC action/proprioception history",
        "evaluator_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = reanalyze_rpc_motion(json.loads(args.run_report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
