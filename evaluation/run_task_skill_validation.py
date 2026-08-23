"""Validate one frozen embodied Task Skill on predeclared unseen LIBERO states.

The complete batch is selected and hash-sealed before execution.  Sensor
evidence may promote the immutable program to ``sensor_validated``; evaluator
files are opened only after every frozen execution has finished and can never
change the Skill or its actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "capability_library")]

from libero_robot_sdk import (  # noqa: E402
    execute_libero_graph_skill_sealed,
    execute_libero_program_sealed,
    libero_task_state_count,
)
from task_skill_workspace import TaskSkillWorkspace  # noqa: E402
from graph_task_skill_workspace import GraphTaskSkillWorkspace  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _select_states(
    *, total: int, count: int, excluded: set[int], program_sha256: str,
) -> list[int]:
    candidates = [state for state in range(total) if state not in excluded]
    ranked = sorted(
        candidates,
        key=lambda state: hashlib.sha256(
            f"{program_sha256}:libero-unseen-state:{state}".encode()
        ).hexdigest(),
    )
    if len(ranked) < count:
        raise ValueError(
            f"requested {count} unseen states but only {len(ranked)} are available"
        )
    return ranked[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-workspace", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--states", default=None,
        help="Optional comma-separated predeclared states; automatic hash selection is default.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 3:
        raise ValueError("Task Skill validation requires at least three unseen states")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"validation output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    skills: Any = TaskSkillWorkspace(args.skill_workspace)
    graph_skill = False
    try:
        inspected = skills.inspect(args.skill_id)
    except (ValueError, KeyError, FileNotFoundError):
        skills = GraphTaskSkillWorkspace(args.skill_workspace)
        inspected = skills.inspect(args.skill_id)
        graph_skill = True
    manifest = inspected["manifest"]
    controller_sha256 = str(
        manifest.get("program_sha256")
        or manifest.get("graph_manifest_sha256")
        or ""
    )
    if len(controller_sha256) != 64:
        raise ValueError("Task Skill has no immutable controller hash")
    development = manifest.get("development_context") or {}
    task = args.task if args.task is not None else development.get("task_selector")
    if task is None:
        raise ValueError("--task is required when the Skill has no development task selector")
    task = int(task)
    total_states = libero_task_state_count(args.suite, task)
    excluded = set()
    if (
        development.get("environment") == args.suite
        and int(development.get("task_selector", -1)) == task
        and development.get("state_selector") is not None
    ):
        excluded.add(int(development["state_selector"]))
    if args.states:
        states = [int(item.strip()) for item in args.states.split(",") if item.strip()]
        if len(states) < 3 or len(states) != len(set(states)):
            raise ValueError("--states must name at least three distinct states")
        if any(state < 0 or state >= total_states for state in states):
            raise ValueError("a declared state is outside the LIBERO state range")
        if excluded.intersection(states):
            raise ValueError("development states cannot be used for unseen validation")
    else:
        states = _select_states(
            total=total_states, count=args.count, excluded=excluded,
            program_sha256=controller_sha256,
        )

    plan = {
        "protocol": "embodied-task-skill-unseen-validation-v1",
        "skill_id": args.skill_id,
        "controller_interface": "graph" if graph_skill else "program",
        "controller_sha256": controller_sha256,
        "suite": args.suite,
        "task": task,
        "seed": args.seed,
        "states": states,
        "development_states_excluded": sorted(excluded),
        "selection": "sha256-ranked-before-execution" if not args.states else "predeclared",
        "sensor_results_consumed_for_skill_validation": True,
        "evaluator_results_consumed_for_iteration": False,
    }
    plan_bytes = (json.dumps(plan, indent=2) + "\n").encode()
    (args.output / "validation_plan.json").write_bytes(plan_bytes)
    _write_json(args.output / "seal.json", {
        "validation_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "controller_sha256": controller_sha256,
        "dependency_hashes": manifest.get("dependencies") or [],
        "evaluator_opened_during_execution": False,
    })

    sensor_results = []
    evaluator_paths = []
    capability_workspace = Path(inspected["capability_workspace"])
    for index, state in enumerate(states, 1):
        run_dir = args.output / "runs" / f"episode_{index:03d}_state_{state:03d}"
        execute_sealed = (
            execute_libero_graph_skill_sealed
            if graph_skill else execute_libero_program_sealed
        )
        report = execute_sealed(
            skills, args.skill_id, suite=args.suite, task=task, state=state,
            seed=args.seed, output=run_dir,
            capability_workspace=capability_workspace,
        )
        evidence = dict(report["sensor_evidence"])
        sensor_results.append({
            "state": state, "seed": args.seed, "run_dir": str(run_dir),
            "sensor_evidence": evidence,
        })
        evaluator_paths.append(Path(report["evaluator_result_path"]))

    # Persist the full sensor batch before any evaluator-only result is opened.
    _write_json(args.output / "sensor_results.json", sensor_results)
    validation_updates = []
    for item in sensor_results:
        state = int(item["state"])
        evidence = dict(item["sensor_evidence"])
        evidence["artifacts"] = {
            **(evidence.get("artifacts") or {}), "validation_run": item["run_dir"],
        }
        validation_updates.append(skills.record_unseen_validation(
            args.skill_id, environment=args.suite,
            state_key=f"task-{task}:state-{state}:seed-{args.seed}",
            sensor_evidence=evidence,
        ))
    _write_json(args.output / "sensor_validation_updates.json", validation_updates)

    # Scoring barrier: frozen executions and sensor validation are now complete.
    scored = []
    for item, evaluator_path in zip(sensor_results, evaluator_paths):
        evaluator = json.loads(evaluator_path.read_text())
        scored.append({
            "state": item["state"], "seed": item["seed"],
            "sensor_verified": item["sensor_evidence"].get("sensor_only_conclusion")
                == "sensor_verification_passed",
            "evaluator_success": bool(evaluator.get("success")),
            "evaluator_calls": int(evaluator.get("evaluator_calls", 0)),
        })
    summary = {
        "protocol": plan["protocol"], "skill_id": args.skill_id,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "episodes": len(scored),
        "sensor_verified": sum(bool(row["sensor_verified"]) for row in scored),
        "evaluator_successes": sum(bool(row["evaluator_success"]) for row in scored),
        "skill_status": validation_updates[-1]["status"],
        "evaluator_results_consumed_for_iteration": False,
        "results": scored,
    }
    _write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
