"""One persistent-model worker for the frozen OpenVLA sealed evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "capability_library"),
    str(ROOT / "capability_library" / "tools"),
    str(ROOT / "Thea"),
    str(ROOT / "Thea" / "simulation"),
]

from frontier_registrar import make_frontier_registrar
from harness.tools.registry import ToolRegistry
from openvla_general_policy import make_openvla_general_tool
from thea_simulation.adapters.libero import LiberoEpisode, project_libero_observation
from thea_simulation.runtime import SimulationRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    policy_tool = make_openvla_general_tool(device=args.device, max_actions_per_call=10)
    completed = 0
    for job in jobs:
        task, state = int(job[0]), int(job[1])
        episode_dir = args.output / f"task_{task:02d}" / f"state_{state:02d}"
        result_file = episode_dir / "result.json"
        if result_file.exists():
            completed += 1
            continue
        episode_dir.mkdir(parents=True, exist_ok=True)
        report: dict[str, object] = {
            "protocol": "harness-acquired-task-zero-shot-v2",
            "harness": "Thea SimulationRuntime + persistent OpenVLA Tool",
            "suite": args.suite,
            "task_index": task,
            "initial_state_index": state,
            "policy": "openvla_base",
            "policy_track": "task_disjoint_transfer",
            "learned_models_used": ["openvla/openvla-7b"],
            "claimable": True,
            "sealed_results_consumed_for_iteration": False,
        }
        runtime = None
        try:
            episode = LiberoEpisode.from_suite(
                args.suite,
                task,
                initial_state_id=state,
                seed=args.seed,
                env_kwargs={
                    "camera_heights": 256,
                    "camera_widths": 256,
                    "camera_depths": True,
                    "ignore_done": True,
                    "horizon": 1200,
                },
            )
            runtime = SimulationRuntime(
                episode,
                observation_projector=project_libero_observation,
                tools=(policy_tool,),
            )
            registry = ToolRegistry()
            runtime.register_tools(registry)
            make_frontier_registrar(
                (f"{args.suite}:task_{task}",),
                ledger_path=str(episode_dir / "capability_acquisition.jsonl"),
                state_path=str(episode_dir / "self_evolution_state.json"),
            )(registry)
            runtime.reset(seed=args.seed)
            result = registry.call_tool("execute_language_policy", {})
            evidence = runtime.capture(result["run_id"])
            report.update(
                {
                    "success": bool(evidence.task_success),
                    "terminated": bool(evidence.terminated),
                    "truncated": bool(evidence.truncated),
                    "tool_result": result,
                    "failure_kind": None if evidence.task_success else "manipulation_execution",
                }
            )
        except Exception as exc:
            report.update(
                {
                    "success": False,
                    "integration_error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if runtime is not None:
                runtime.close()
        result_file.write_text(json.dumps(report, indent=2) + "\n")
        completed += 1
        if completed % 5 == 0:
            print(f"device={args.device} completed={completed}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
