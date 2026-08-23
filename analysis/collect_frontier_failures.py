"""Collect valid sealed failures without exposing evaluator state to policy code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(run_root: Path) -> dict:
    failures = []
    for path in sorted(run_root.glob("task_*/state_*/result.json")):
        result = json.loads(path.read_text())
        failures.append(
            {
                "task_index": result.get("task_index"),
                "initial_state_index": result.get("initial_state_index"),
                "success": bool(result.get("success")),
                "claimable": bool(result.get("claimable")),
                "failure_kind": result.get("failure_kind"),
                "policy_track": result.get("policy_track"),
                "sealed_results_consumed_for_iteration": result.get(
                    "sealed_results_consumed_for_iteration"
                ),
            }
        )
    return {
        "run": str(run_root),
        "episodes": len(failures),
        "failures": sum(not item["success"] for item in failures),
        "claimable_results": sum(item["claimable"] for item in failures),
        "sealed_results_consumed_for_iteration": any(
            item["sealed_results_consumed_for_iteration"] for item in failures
        ),
        "failure_records": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(args.run_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
