"""Prove that independent controller source drives real LIBERO action traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "capability_library"), str(ROOT / "evaluation")]

from controller_program_workspace import ControllerProgramWorkspace
from libero_robot_sdk import execute_libero_program


PROGRAM = '''def run(robot):
    first = robot.observe()
    target = list(first["eef_xyz"])
    target[0] = target[0] + DIRECTION * 0.03
    outcomes = []
    for index in range(3):
        result = robot.act({
            "target_eef_xyz": target,
            "gripper": -1,
            "orientation": "topdown",
            "position_gain": 0.35,
            "max_translation_action": 0.30,
            "repeat": 2,
        })
        outcomes.append(result)
        robot.record({"phase": "source_owned_motion", "index": index, "error_m": result["error_m"]})
    return {"initial_eef_xyz": first["eef_xyz"], "final_eef_xyz": outcomes[-1]["eef_xyz"]}
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=5)
    parser.add_argument("--state", type=int, default=22)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    workspace = ControllerProgramWorkspace(
        args.output / "programs", python="/data/zxy/envs/vla-report/bin/python",
        timeout_sec=120, max_rpc_calls=100,
    )
    reports = []
    for label, direction in (("positive", 1.0), ("negative", -1.0)):
        source = f"DIRECTION = {direction}\n" + PROGRAM
        created = workspace.create(f"smoke_motion_{label}", source)
        reports.append(execute_libero_program(
            workspace, created["program_id"], suite="libero_spatial",
            task=args.task, state=args.state, seed=args.seed,
            output=args.output / f"execution_{label}",
        ))
    traces = []
    for report in reports:
        trace = json.loads(Path(report["runtime_trace"]).read_text())
        acts = [event for event in trace["rpc_events"] if event["method"] == "act"]
        traces.append([event["arguments"]["action"]["target_eef_xyz"][0] for event in acts])
    result = {
        "protocol": "libero-independent-program-smoke-v1",
        "program_ids": [report["program_id"] for report in reports],
        "x_targets": traces,
        "different_source_changed_real_action_sequence": bool(
            traces[0] and traces[1] and traces[0] != traces[1]
        ),
        "evaluator_called": False,
        "reports": reports,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
