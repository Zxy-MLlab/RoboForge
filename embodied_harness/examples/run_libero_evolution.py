"""Run the standalone Harness on one evaluator-blind LIBERO deployment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from embodied_harness.adapters.libero import LiberoAdapter, LiberoEpisode
from embodied_harness.capabilities import OpenVocabularyRGBD
from embodied_harness.evolution import EvolutionEngine
from embodied_harness.model import OpenAICompatibleModel


TOOL_NAME = "open_vocab_rgbd_grounded_sam"


class LiberoFactory:
    def __init__(self, *, episode, root, perception, tool_id):
        self.episode = episode; self.root = Path(root)
        self.perception = perception; self.tool_id = tool_id
        episode_root = self.root / "episodes"
        existing = [
            int(path.name.split("_")[-1]) for path in episode_root.glob("episode_[0-9]*")
            if path.name.split("_")[-1].isdigit()
        ]
        self.count = max(existing, default=0)

    def __call__(self):
        self.count += 1
        return LiberoAdapter(
            episode=self.episode,
            artifact_dir=self.root / "episodes" / f"episode_{self.count:03d}",
            capabilities={self.tool_id: self.perception.detect},
            verifiers={"visual_support_relation": self.perception.verify_support_relation},
            verifier_tool_ids={"visual_support_relation": self.tool_id},
        )


def task_language(config_path: str, suite_name: str, task_index: int) -> str:
    os.environ["LIBERO_CONFIG_PATH"] = str(Path(config_path).resolve())
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name]()
    return str(suite.get_task(task_index).language)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--base-url", default="https://api.apexin.ai/v1")
    parser.add_argument("--config", default="config/standalone_libero")
    args = parser.parse_args()
    key = os.environ.get("APEX_API_KEY")
    if not key: raise SystemExit("APEX_API_KEY is not set")
    run_root = Path(args.run_dir).resolve()
    instruction = task_language(args.config, args.suite, args.task)
    perception = OpenVocabularyRGBD(
        groundingdino_root="third_party/GroundingDINO",
        groundingdino_config="third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        groundingdino_checkpoint="/data/zxy/GroundingDINO/models/groundingdino_swint_ogc.pth",
        sam_root="third_party/segment-anything",
        sam_checkpoint="checkpoints/sam_vit_b_01ec64.pth", device=args.device,
    )
    model = OpenAICompatibleModel(
        api_key=key, base_url=args.base_url, model=args.model,
        reasoning_effort=args.reasoning_effort, max_tokens=8000,
    )
    episode = LiberoEpisode(
        suite=args.suite, task_index=args.task, initial_state_index=args.state,
        seed=args.seed, image_size=256, horizon=1200,
        config_path=args.config,
    )
    placeholder_factory = lambda: None
    engine = EvolutionEngine(
        root=run_root, model=model, adapter_factory=placeholder_factory,
        available_initial_fields={"task_instruction", "deployment_contract"},
        python="/data/zxy/envs/vla-report/bin/python", max_agent_turns=50,
        deployment_guidance={
            "benchmark": "LIBERO with OSC_POSE Panda; no benchmark signal is visible",
            "sense": "adapter.sense('rgbd', {}) returns frame paths, calibration, and proprioception",
            "perception_tool": {
                "tool_id": f"{TOOL_NAME}:v001",
                "call": "adapter.use(tool_id, {'frame': frame, 'queries': ['black bowl','plate','ramekin']})",
                "result": "receipt.result.detections maps each query to scored instances; each projected instance has world_xyz, world_bounds_10_90, point_ref, mask_path",
                "selection": "use language relations and live instance geometry; never choose by task/state ID",
            },
            "actions": {
                "move_to_point": {"required": ["type='move_to_point'", "target_ref"],
                                  "optional": ["offset xyz", "gripper -1 open / +1 close", "max_steps", "tolerance_m"]},
                "gripper": {"type": "gripper", "command": "open or close", "repeat": "1..40"},
                "settle": {"type": "settle", "steps": "1..60", "gripper": "-1 or +1"},
                "grasp_geometry": "Panda top-down wrist usually approaches 0.15 m above the RGB-D surface point, contacts with a 0.06..0.10 m positive-z wrist offset, closes, then lifts; all positions remain point_ref-relative",
                "place_geometry": "move above target point_ref, descend with a positive-z wrist offset, open, retreat, then re-observe",
            },
            "final_verifier": {
                "name": "visual_support_relation",
                "call": "adapter.verify(name, {'frame': fresh_rgbd_frame, 'object_query':'black bowl', 'source_ref': selected_bowl_point_ref, 'target_ref': selected_plate_point_ref})",
                "required": "source_ref and target_ref must be independent pre-action GroundingDINO+SAM references; same-mask cross-label matching is forbidden",
                "constraint": "must be in a checkpoint-only node; success edge comes directly from verified",
            },
        },
    )
    registered = engine.tools.register_deployment_tool(
        name=TOOL_NAME,
        description="Task-disjoint GroundingDINO + SAM open-vocabulary RGB-D instances",
        implementation_path=Path(__file__).parents[1] / "capabilities/open_vocab_rgbd.py",
        input_schema={"frame": "adapter RGB-D report", "queries": ["text"]},
        output_schema={"detections": "query -> projected instances"},
        provenance=perception.provenance,
        validation_evidence={
            "kind": "public_checkpoint_and_interface_validation",
            "current_task_examples_used_for_training": False,
            "passed": True,
        },
    )
    expected = registered["tool_id"]
    engine.deployment_guidance["perception_tool"]["tool_id"] = expected
    engine.deployment_guidance["perception_tool"]["call"] = (
        f"adapter.use('{expected}', {{'frame': frame, "
        "'queries': ['black bowl','plate','ramekin']})"
    )
    engine.adapter_factory = LiberoFactory(
        episode=episode, root=run_root, perception=perception, tool_id=expected,
    )
    state = engine.run(
        task=instruction, skill_name=f"libero_spatial_task_{args.task}_skill",
        max_rounds=args.max_rounds,
    )
    print(json.dumps({
        "status": state["status"], "instruction": instruction,
        "rounds": state["rounds"], "skill": state.get("skill"),
    }, indent=2))
    return 0 if state["status"] == "sensor_success" else 2


if __name__ == "__main__": raise SystemExit(main())
