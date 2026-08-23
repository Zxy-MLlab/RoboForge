"""Evaluator-blind smoke test for persistent multi-node LIBERO graph execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "capability_library"), str(ROOT / "evaluation")]

from controller_graph_workspace import ControllerGraphWorkspace
from stage_node_workspace import StageNodeWorkspace
from libero_robot_sdk import execute_libero_graph


PYTHON = "/data/zxy/envs/vla-report/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, default=4)
    parser.add_argument("--state", type=int, default=23)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    nodes = StageNodeWorkspace(
        args.output / "stage_nodes", python=PYTHON, timeout_sec=120,
    )
    first = nodes.create(
        name="smoke_first_observation", stage_kind="observation",
        description="Acquire the first public sensor frame.", requires=[],
        provides_by_outcome={"observed": ["first_frame", "first_step"]},
        source='''def run_stage(robot, context):
    frame = robot.observe()
    return {"outcome": "observed", "updates": {
        "first_frame": frame["frame_id"], "first_step": frame["step"]}}
''',
    )["node_id"]
    second = nodes.create(
        name="smoke_second_observation", stage_kind="observation",
        description="Acquire a second frame through the same live adapter.",
        requires=["first_frame", "first_step"],
        provides_by_outcome={"observed_again": ["second_frame", "second_step"]},
        source='''def run_stage(robot, context):
    frame = robot.observe()
    robot.record({"first_frame": context["first_frame"],
                  "second_frame": frame["frame_id"]})
    return {"outcome": "observed_again", "updates": {
        "second_frame": frame["frame_id"], "second_step": frame["step"]}}
''',
    )["node_id"]
    graphs = ControllerGraphWorkspace(args.output / "controller_graphs", nodes=nodes)
    graph_id = graphs.create(
        name="libero_persistent_adapter_smoke",
        description="Two independent Stage Nodes observe one persistent episode.",
        entry_node="first",
        bindings={"first": first, "second": second},
        edges=[
            {"from": "first", "outcome": "observed", "to": "second"},
            {"from": "second", "outcome": "observed_again", "to": "$success"},
        ],
    )["graph_id"]
    result = execute_libero_graph(
        graphs, graph_id, suite="libero_spatial", task=args.task,
        state=args.state, seed=args.seed, output=args.output / "episode",
    )
    trace = json.loads((args.output / "episode" / "trace.json").read_text())
    observations = [row for row in trace if row.get("event") == "observe"]
    node_trace = result["sensor_evidence"]["controller_graph"]["node_trace"]
    passed = bool(
        result["execution_completed"]
        and [row.get("alias") for row in node_trace] == ["first", "second"]
        and [row.get("frame_id") for row in observations] == [
            "frame-0001", "frame-0002",
        ]
        and result["sensor_evidence"].get("controller_interface") == "graph"
        and result.get("evaluator_hidden") is True
    )
    report = {
        "protocol": "libero-persistent-graph-smoke-v1",
        "passed": passed,
        "graph_id": graph_id,
        "node_trace": node_trace,
        "observation_frame_ids": [row.get("frame_id") for row in observations],
        "sensor_only_conclusion": result["sensor_evidence"].get(
            "sensor_only_conclusion"
        ),
        "evaluator_called": False,
    }
    (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
