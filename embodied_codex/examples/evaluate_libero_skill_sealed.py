"""Sealed LIBERO scoring for an immutable Embodied Codex Skill.

All declared controllers terminate before any benchmark predicate is opened.
Scores are evaluator-only artifacts and cannot feed back into evolution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from embodied_codex.deployments import LiberoDeployment, LiberoEpisode
from embodied_codex.examples.evaluate_libero_skill import (
    _load_class, _load_function, _sensor_success, inspect_skill,
)
from embodied_codex.runtime import ControllerRuntime


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def _capabilities(frozen_tools, perception, *, api_key, base_url, model,
                  reasoning_effort):
    capabilities = {}; graspnet = None
    perception_item = next((item for item in frozen_tools.values()
                            if item["manifest"].get("name") ==
                            "open_vocab_rgbd_grounded_sam"), None)
    for tool_id, item in frozen_tools.items():
        manifest = item["manifest"]
        if not manifest.get("execution_owned_by_deployment"):
            capabilities[tool_id] = _load_function(item["folder"] / "tool.py")
            continue
        name = manifest.get("name")
        if name == "open_vocab_rgbd_grounded_sam":
            capabilities[tool_id] = perception.detect
        elif name == "graspnet_rgbd_6dof":
            if graspnet is None:
                if perception_item is None:
                    raise RuntimeError("frozen Skill lacks perception dependency")
                graspnet_class=_load_class(
                    item["folder"] / "tool.py", "GraspNetRGBD",
                    relative_modules={
                        "open_vocab_rgbd": perception_item["folder"] / "tool.py"
                    })
                graspnet = graspnet_class(
                    backend_script="capability_library/tools/graspnet_rgbd_grasp.py",
                    checkpoint="checkpoints/graspnet-checkpoint-rs.tar",
                    python="/data/zxy/envs/vla-report/bin/python")
            capabilities[tool_id] = graspnet.infer
        elif name == "vlm_visual_relation_grounder":
            grounder_class=_load_class(
                item["folder"] / "tool.py", "VLMVisualRelationGrounder")
            grounder=grounder_class(api_key=api_key,base_url=base_url,
                                    model=model,reasoning_effort=reasoning_effort)
            capabilities[tool_id]=grounder.select
        else:
            raise RuntimeError(f"LIBERO Adapter cannot bind deployment Tool: {name}")
    return capabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--states", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--base-url", default="https://api.apexin.ai/v1")
    parser.add_argument("--config", default="config/standalone_libero")
    args = parser.parse_args()
    if len(set(args.states)) != len(args.states):
        raise ValueError("sealed states must be unique")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"sealed output must be new and empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    controller, manifest, frozen_tools = inspect_skill(args.skill_dir)
    skill_manifest = Path(args.skill_dir).resolve() / "manifest.json"
    seal = {
        "protocol": "embodied-codex-libero-sealed-skill-evaluation-v1",
        "skill_id": manifest.get("skill_id"),
        "controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "skill_manifest_sha256": hashlib.sha256(skill_manifest.read_bytes()).hexdigest(),
        "suite": args.suite, "task": args.task, "states": list(args.states),
        "episodes": len(args.states),
        "evaluator_opened_during_execution": False,
        "results_consumed_for_iteration": False,
    }
    _write(output / "seal.json", seal)

    perception_item = next((item for item in frozen_tools.values()
                            if item["manifest"].get("name")=="open_vocab_rgbd_grounded_sam"),None)
    if perception_item is None:raise RuntimeError("frozen Skill lacks perception Tool")
    perception_class=_load_class(perception_item["folder"]/"tool.py","OpenVocabularyRGBD")
    perception = perception_class(
        groundingdino_root="third_party/GroundingDINO",
        groundingdino_config=("third_party/GroundingDINO/groundingdino/config/"
                              "GroundingDINO_SwinT_OGC.py"),
        groundingdino_checkpoint="/data/zxy/GroundingDINO/models/groundingdino_swint_ogc.pth",
        sam_root="third_party/segment-anything",
        sam_checkpoint="checkpoints/sam_vit_b_01ec64.pth", device=args.device)
    capabilities = _capabilities(
        frozen_tools, perception, api_key=os.environ.get("APEX_API_KEY", ""),
        base_url=args.base_url, model=args.model,
        reasoning_effort=args.reasoning_effort)
    runtime = ControllerRuntime(python="/data/zxy/envs/vla-report/bin/python")
    pending = []; sensor_rows = []; scored = []
    try:
        # Execution phase: the evaluator remains unopened for the whole batch.
        for state in args.states:
            episode_dir = output / "runs" / f"state_{state:03d}"
            deployment = LiberoDeployment(
                episode=LiberoEpisode(args.suite, args.task, state,
                                      config_path=args.config),
                artifact_dir=episode_dir, capabilities=capabilities,
                verifiers={"visual_attachment": perception.verify_attachment,
                           "visual_support_relation": perception.verify_support_relation})
            pending.append((state, deployment, episode_dir))
            execution = runtime.execute(controller, deployment)
            report = dict(deployment.sensor_report(execution))
            deployment.seal_controller_execution()
            row = {"state": state,
                   "sensor_success": _sensor_success(execution, report),
                   "execution": execution, "sensor_report": report}
            _write(episode_dir / "skill_execution.json", row)
            sensor_rows.append(row)
        _write(output / "sensor_results.json", [
            {"state": row["state"], "sensor_success": row["sensor_success"]}
            for row in sensor_rows])

        # Scoring phase: all controller processes have ended and robot I/O is sealed.
        for (state, deployment, episode_dir), sensor in zip(pending, sensor_rows):
            success = deployment._sealed_check_once()
            _write(episode_dir / "_evaluator_only" / "result.json", {
                "success": success, "evaluator_calls": 1,
                "opened_after_entire_batch_execution": True,
                "visible_to_controller": False, "fed_back_to_evolution": False})
            scored.append({"state": state, "sensor_success": sensor["sensor_success"],
                           "evaluator_success": success})
    finally:
        for _state, deployment, _episode_dir in pending:
            deployment.close()

    summary = {**seal,
        "sensor_successes": sum(row["sensor_success"] for row in scored),
        "evaluator_successes": sum(row["evaluator_success"] for row in scored),
        "evaluator_calls": len(scored), "fed_back_to_evolution": False,
        "results": scored}
    _write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["evaluator_successes"] == len(scored) else 2


if __name__ == "__main__":
    raise SystemExit(main())
