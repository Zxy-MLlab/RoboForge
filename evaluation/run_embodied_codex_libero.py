"""Canonical end-to-end Embodied Codex entry point for LIBERO.

One command owns the public workflow: autonomous sensor-only development,
capability acquisition, frozen Task Skill creation, deterministic unseen-state
validation, and post-batch sealed scoring.  Evaluator results are never passed
back into development or used to modify a Skill.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/data/zxy/envs/vla-report/bin/python"


def _task_list(value: str) -> list[int]:
    tasks: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(item) for item in token.split("-", 1))
            tasks.extend(range(start, end + 1))
        else:
            tasks.append(int(token))
    tasks = list(dict.fromkeys(tasks))
    if not tasks or any(task < 0 or task > 9 for task in tasks):
        raise argparse.ArgumentTypeError("tasks must be LIBERO-Spatial selectors 0..9")
    return tasks


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def development_command(
    *, task: int, state: int, seed: int, max_rounds: int,
    max_turns_per_round: int, max_turns_acquisition: int,
    acquisition_after_same_failure: int, output: Path,
    controllers: Path, stage_nodes: Path, capabilities: Path, task_skills: Path,
    force_acquisition: bool,
) -> list[str]:
    command = [
        PYTHON, str(ROOT / "evaluation" / "run_autonomous_evolution.py"),
        "--task", str(task), "--state", str(state), "--seed", str(seed),
        "--max-rounds", str(max_rounds),
        "--max-turns-per-round", str(max_turns_per_round),
        "--max-turns-acquisition", str(max_turns_acquisition),
        "--acquisition-after-same-failure", str(acquisition_after_same_failure),
        "--output", str(output),
        "--controller-workspace", str(controllers),
        "--controller-interface", "graph",
        "--stage-node-workspace", str(stage_nodes),
        "--capability-workspace", str(capabilities),
        "--task-skill-workspace", str(task_skills),
    ]
    if force_acquisition:
        command.append("--force-acquisition-next-round")
    return command


def validation_command(
    *, skill_id: str, task: int, count: int, seed: int,
    task_skills: Path, output: Path,
) -> list[str]:
    return [
        PYTHON, str(ROOT / "evaluation" / "run_task_skill_validation.py"),
        "--skill-workspace", str(task_skills), "--skill-id", skill_id,
        "--suite", "libero_spatial", "--task", str(task),
        "--count", str(count), "--seed", str(seed), "--output", str(output),
    ]


def _development_status(task_dir: Path) -> dict[str, Any]:
    report_path = task_dir / "development" / "report.json"
    state_path = task_dir / "development" / "evolution_state.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    return {
        "status": report.get("status") or state.get("status") or "process_failed",
        "rounds": len((report or state).get("rounds") or []),
        "authoring_failures": len((report or state).get("authoring_failures") or []),
        "task_skill_candidate": report.get("task_skill_candidate"),
        "last_sensor_evidence": (
            ((report or state).get("rounds") or [{}])[-1].get("sensor_evidence") or {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=_task_list, default=[4])
    parser.add_argument("--development-state", type=int, default=23)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--max-turns-per-round", type=int, default=22)
    parser.add_argument("--max-turns-acquisition", type=int, default=24)
    parser.add_argument("--acquisition-after-same-failure", type=int, default=2)
    parser.add_argument("--unseen-state-count", type=int, default=3)
    parser.add_argument("--force-acquisition-next-round", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capability-workspace", type=Path,
        default=ROOT / "capability_library" / "acquired_tools",
    )
    parser.add_argument(
        "--task-skill-workspace", type=Path,
        default=ROOT / "capability_library" / "task_skills",
    )
    args = parser.parse_args()
    if not os.environ.get("APEX_API_KEY"):
        raise OSError("APEX_API_KEY is required for autonomous development")
    if args.unseen_state_count < 3:
        raise ValueError("at least three unseen states are required")
    args.output.mkdir(parents=True, exist_ok=True)
    args.capability_workspace.mkdir(parents=True, exist_ok=True)
    args.task_skill_workspace.mkdir(parents=True, exist_ok=True)

    campaign = {
        "protocol": "embodied-codex-libero-campaign-v1",
        "model": "gpt-5.6-sol",
        "suite": "libero_spatial",
        "tasks": args.tasks,
        "development_state": args.development_state,
        "seed": args.seed,
        "agent_owns_controller_code": True,
        "generic_capability_acquisition": True,
        "evaluator_results_consumed_for_iteration": False,
        "task_results": [],
    }
    _write_json(args.output / "campaign.json", campaign)

    for task in args.tasks:
        task_dir = args.output / f"task_{task:02d}"
        development = task_dir / "development"
        controllers = task_dir / "controllers"
        stage_nodes = task_dir / "stage_nodes"
        command = development_command(
            task=task, state=args.development_state, seed=args.seed,
            max_rounds=args.max_rounds,
            max_turns_per_round=args.max_turns_per_round,
            max_turns_acquisition=args.max_turns_acquisition,
            acquisition_after_same_failure=args.acquisition_after_same_failure,
            output=development, controllers=controllers, stage_nodes=stage_nodes,
            capabilities=args.capability_workspace,
            task_skills=args.task_skill_workspace,
            force_acquisition=args.force_acquisition_next_round,
        )
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
        status = _development_status(task_dir)
        task_result: dict[str, Any] = {
            "task": task, "development_process_returncode": completed.returncode,
            **status,
        }
        candidate = status.get("task_skill_candidate") or {}
        if status["status"] == "sensor_success" and candidate.get("skill_id"):
            validation_output = task_dir / "unseen_validation"
            if not (validation_output / "summary.json").is_file():
                validated = subprocess.run(validation_command(
                    skill_id=str(candidate["skill_id"]), task=task,
                    count=args.unseen_state_count, seed=args.seed,
                    task_skills=args.task_skill_workspace,
                    output=validation_output,
                ), cwd=ROOT, env=os.environ.copy())
                task_result["validation_process_returncode"] = validated.returncode
            summary_path = validation_output / "summary.json"
            if summary_path.is_file():
                task_result["unseen_validation"] = json.loads(summary_path.read_text())
        else:
            frontier = {
                "task": task,
                "development_status": status["status"],
                "rounds": status["rounds"],
                "last_sensor_evidence": status["last_sensor_evidence"],
                "evaluator_used": False,
            }
            _write_json(task_dir / "frontier_failure.json", frontier)
        campaign["task_results"].append(task_result)
        _write_json(args.output / "campaign.json", campaign)

    campaign["tasks_with_development_sensor_success"] = sum(
        item.get("status") == "sensor_success" for item in campaign["task_results"]
    )
    campaign["tasks_sensor_validated"] = sum(
        (item.get("unseen_validation") or {}).get("skill_status") == "sensor_validated"
        for item in campaign["task_results"]
    )
    _write_json(args.output / "summary.json", campaign)
    print(json.dumps(campaign, indent=2))


if __name__ == "__main__":
    main()
