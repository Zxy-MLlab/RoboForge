#!/usr/bin/env python3
"""Summarize LeRobot LIBERO eval_info.json without changing evaluation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(payload: dict) -> dict:
    failures = []
    tasks = []
    for item in payload.get("per_task", []):
        task_id = int(item["task_id"])
        successes = [bool(value) for value in item["metrics"].get("successes", [])]
        failed_episode_indices = [index for index, value in enumerate(successes) if not value]
        tasks.append(
            {
                "task_group": item["task_group"],
                "task_id": task_id,
                "successes": sum(successes),
                "episodes": len(successes),
                "success_rate": sum(successes) / len(successes) if successes else None,
                "failed_episode_indices": failed_episode_indices,
            }
        )
        failures.extend(
            {
                "task_group": item["task_group"],
                "task_id": task_id,
                "episode_index": index,
                "status": "unresolved",
                "failure_category": "pending_visual_trace_review",
            }
            for index in failed_episode_indices
        )
    return {
        "overall": payload.get("overall", {}),
        "tasks": tasks,
        "unresolved_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_info", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.eval_info.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

