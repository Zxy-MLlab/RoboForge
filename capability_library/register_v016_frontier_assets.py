"""Register the v016 LIBERO frontier assets with explicit provenance."""

from __future__ import annotations

import json
from pathlib import Path

from asset_registry import register_asset


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "capability_library" / "library.json"
RUN_ROOT = ROOT / "runs" / "coding_harness" / "libero_spatial_v016_state6_full"
CONTROLLER = ROOT / "runs" / "coding_harness" / "controllers_frozen_v14" / "generic_rgbd_closed_loop_pick_place_v016" / "v001"
TASKS = [f"libero_spatial:task_{index}" for index in range(10)]


def _asset(**fields):
    fields.setdefault("source_urls", [])
    fields.setdefault("tested_tasks", TASKS)
    fields.setdefault("reused_tasks", [])
    fields.setdefault("known_failures", [])
    fields.setdefault("sensors", ["RGB", "RGB-D", "proprioception", "language"])
    fields.setdefault("current_task_data_used", False)
    fields.setdefault("privileged_state_used", False)
    return fields


def main() -> None:
    summary = json.loads((RUN_ROOT / "summary.json").read_text())
    assets = [
        _asset(
            asset_id="tool.instance-grounding-relation-projection.v1",
            kind="tool", name="Instance grounding relation-region projection", version="1",
            status="development_validated",
            source_urls=["local:capability_library/tools/instance_grounding.py"],
            sensors=["RGB", "RGB-D", "language"],
            input_schema=["live detector regions", "language relation", "RGB-D geometry"],
            output_schema=["movable source region", "relation consistency audit"],
            implementation=str(ROOT / "capability_library/tools/instance_grounding.py"),
            evidence=str(RUN_ROOT / "sensor_results.json"),
            known_failures=["ambiguous or occluded visual instances"],
        ),
        _asset(
            asset_id="tool.graspnet-orientation-compatible-fallback.v1",
            kind="tool", name="GraspNet strict-plus-calibrated fallback candidate pool", version="1",
            status="development_validated",
            source_urls=[
                "https://github.com/graspnet/graspnet-baseline",
                "local:capability_library/tools/closed_loop_recovery.py",
            ],
            sensors=["RGB-D", "proprioception"],
            input_schema=["public GraspNet candidates", "calibrated robot orientation"],
            output_schema=["deduplicated ranked retry pool", "pool provenance"],
            implementation=str(ROOT / "capability_library/tools/closed_loop_recovery.py"),
            evidence=str(RUN_ROOT / "runs/task6_state6_seed7/result.json"),
            known_failures=["fallback may still fail on support/transfer geometry"],
        ),
        _asset(
            asset_id="tool.sensor-source-relocalization-after-contact.v1",
            kind="tool", name="RGB-D/SAM source relocalization after failed contact", version="1",
            status="development_validated",
            source_urls=["local:/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py"],
            input_schema=["fresh RGB-D", "open-vocabulary detector", "SAM box prompt", "action history"],
            output_schema=["updated source estimate", "attachment evidence"],
            implementation="/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py",
            evidence=str(RUN_ROOT / "sensor_results.json"),
            known_failures=["source can remain visually occluded or unreachable"],
        ),
        _asset(
            asset_id="tool.language-query-local-fallback.v1",
            kind="tool", name="Generic local language noun-query fallback", version="1",
            status="unit_tested",
            source_urls=["local:/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py"],
            sensors=["language"],
            input_schema=["natural-language instruction"],
            output_schema=["concrete visual noun phrases", "fallback audit"],
            implementation="/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py",
            evidence=str(ROOT / "runs/coding_harness/task0_state3_v014_authoring/report.json"),
            known_failures=["lexical fallback is less expressive than the external model"],
        ),
        _asset(
            asset_id="skill.autonomous-closed-loop-grasp-place-recovery.v2",
            kind="skill", name="Autonomous closed-loop grasp-place recovery", version="2",
            status="cross_task_reused",
            source_urls=["local:capability_library/skills/autonomous-closed-loop-grasp-place-recovery/SKILL.md"],
            implementation=str(ROOT / "capability_library/skills/autonomous-closed-loop-grasp-place-recovery/SKILL.md"),
            evidence=str(RUN_ROOT / "seal.json"),
            tested_tasks=TASKS, reused_tasks=TASKS,
            known_failures=["placement support geometry", "ambiguous visual instance identity"],
        ),
        _asset(
            asset_id="skill.visual-articulated-drawer-open-and-retrieve.v1",
            kind="skill", name="Visual articulated drawer open and retrieve", version="1",
            status="cross_task_reused",
            source_urls=["local:capability_library/skills/visual-articulated-drawer-open-and-retrieve/SKILL.md"],
            implementation=str(ROOT / "capability_library/skills/visual-articulated-drawer-open-and-retrieve/SKILL.md"),
            evidence=str(RUN_ROOT / "runs/task4_state6_seed7/agent_observation.json"),
            tested_tasks=["libero_spatial:task_4"], reused_tasks=["libero_spatial:task_4"],
            known_failures=["small observed handle displacement"],
        ),
        _asset(
            asset_id="experience.frozen-libero-spatial-v016-state6-frontier.v1",
            kind="experience", name="Frozen LIBERO-Spatial v016 state-6 frontier", version="1",
            status="frozen_candidate", source_urls=[], tested_tasks=TASKS, reused_tasks=[],
            evidence=str(RUN_ROOT / "summary.json"), run_root=str(RUN_ROOT),
            controller_id="generic_rgbd_closed_loop_pick_place_v016:v001",
            state=6, episodes=10, evaluator_successes=summary["evaluator_successes"],
            sensor_verified=summary["sensor_verified"],
            results_consumed_for_iteration=summary["results_consumed_for_iteration"],
            protocol="sealed-generated-controller-transfer-v1; evaluator after full batch",
            known_failures=["tasks 1, 3, 5, 6, 8 evaluator unresolved"],
        ),
    ]

    for task in range(10):
        run = RUN_ROOT / "runs" / f"task{task}_state6_seed7"
        result = json.loads((run / "result.json").read_text())
        scorer = json.loads((run / "_evaluator_only" / "result.json").read_text())
        sensor = bool(result.get("attachment_verified") and (result.get("placement_verification") or {}).get("verified"))
        evaluator = bool(scorer.get("success"))
        if evaluator:
            assets.append(_asset(
                asset_id=f"experience.libero-spatial-task{task}-v016-state6-success.v1",
                kind="experience", name=f"LIBERO-Spatial task {task} v016 state 6 successful rollout", version="1",
                status="development_validated", tested_tasks=[f"libero_spatial:task_{task}"],
                reused_tasks=[], evidence=str(run / "result.json"), result=str(run / "_evaluator_only/result.json"),
                trajectory=str(run / "trajectory.hdf5") if (run / "trajectory.hdf5").is_file() else None,
                visual_evidence=[str(run / "current_rgb.png"), str(run / "final_rgb.png"), str(run / "rollout.mp4")],
                evaluator_calls=scorer.get("evaluator_calls"), sensor_verified=sensor,
            ))
        else:
            reason = "sensor_and_evaluator_failure"
            if sensor:
                reason = "sensor_verification_passed_but_evaluator_failed"
            elif not result.get("attachment_verified"):
                reason = "attachment_not_verified"
            else:
                reason = "placement_not_verified"
            assets.append(_asset(
                asset_id=f"frontier.failure-libero-spatial-task{task}-v016-state6.v1",
                kind="frontier_failure", name=f"LIBERO-Spatial task {task} v016 state 6 unresolved frontier", version="1",
                status="development_validated", tested_tasks=[f"libero_spatial:task_{task}"],
                evidence=str(run / "result.json"), result=str(run / "_evaluator_only/result.json"),
                failure_reason=reason, sensor_verified=sensor, evaluator_success=evaluator,
                visual_evidence=[str(run / "current_rgb.png"), str(run / "final_rgb.png"), str(run / "rollout.mp4")],
                known_failures=[reason],
            ))

    for asset in assets:
        asset.pop("trajectory", None) if asset.get("trajectory") is None else None
        result = register_asset(asset, library_path=str(LIBRARY))
        if not result.get("success"):
            raise RuntimeError(result)
        print(result["asset_id"])


if __name__ == "__main__":
    main()
