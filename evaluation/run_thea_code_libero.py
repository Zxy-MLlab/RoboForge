"""Execute one frozen code-controller episode through Thea's tool registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "capability_library" / "tools"))

from harness.tools.registry import ToolRegistry
from frontier_registrar import make_frontier_registrar
from rgbd_pick_place import make_thea_rgbd_pick_place_tool
from openvla_general_policy import make_openvla_general_tool
from thea_simulation.adapters.libero import LiberoEpisode, project_libero_observation
from thea_simulation.runtime import SimulationRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--state", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy", choices=("classical", "openvla_base"), default="classical")
    parser.add_argument("--openvla-model", default="/data/zxy/cache/models--openvla--openvla-7b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    episode = LiberoEpisode.from_suite(
        args.suite,
        args.task,
        initial_state_id=args.state,
        seed=args.seed,
        env_kwargs={
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_depths": True,
            "ignore_done": True,
            "horizon": 1200,
        },
    )
    policy_tool = (
        make_openvla_general_tool(model_path=args.openvla_model, device="cuda:0")
        if args.policy == "openvla_base"
        else make_thea_rgbd_pick_place_tool()
    )
    runtime = SimulationRuntime(
        episode,
        observation_projector=project_libero_observation,
        tools=(policy_tool,),
    )
    registry = ToolRegistry()
    runtime.register_tools(registry)
    # These tools are available to an LLM-driven Harness. The deterministic
    # controller below does not call them; exposing them here keeps the
    # simulation adapter identical for autonomous capability-acquisition runs.
    make_frontier_registrar(
        (f"{args.suite}:task_{args.task}",),
        ledger_path=str(args.output / "capability_acquisition.jsonl"),
        state_path=str(args.output / "self_evolution_state.json"),
    )(registry)
    try:
        runtime.reset(seed=args.seed)
        tool_name = "execute_language_policy" if args.policy == "openvla_base" else "rgbd_code_pick_place"
        result = registry.call_tool(tool_name, {})
        evidence = runtime.capture(result["run_id"]) if result.get("run_id") else None
        report = {
            "protocol": "harness-acquired-task-zero-shot-v2",
            "harness": "Thea SimulationRuntime + ToolRegistry",
            "suite": args.suite,
            "task_index": args.task,
            "initial_state_index": args.state,
            "success": bool(evidence.task_success) if evidence is not None else False,
            "terminated": bool(evidence.terminated) if evidence is not None else False,
            "truncated": bool(evidence.truncated) if evidence is not None else False,
            "tool_result": result,
            "failure_kind": (
                None
                if evidence is not None
                else str(result.get("kind") or "tool_execution_failure")
            ),
            "learned_models_used": (
                ["openvla/openvla-7b"] if args.policy == "openvla_base" else []
            ),
            "policy_track": (
                "task_disjoint_transfer" if args.policy == "openvla_base" else "classical_diagnostic"
            ),
            "capability_acquisition_tools_registered": [
                "search_public_embodied_resources",
                "record_capability_acquisition_event",
                "check_asset_provenance",
                "self_evolve_from_failure",
            ],
            "claimable": not (args.task == 0 and args.state == 0),
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
