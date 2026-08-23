"""Register the v017 sealed LIBERO-Spatial frontier and failures."""

from __future__ import annotations

import json
from pathlib import Path

from asset_registry import register_asset


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "capability_library" / "library.json"
RUN_ROOT = ROOT / "runs" / "coding_harness" / "libero_spatial_v017_state8_full"
TASKS = [f"libero_spatial:task_{i}" for i in range(10)]


def base(**fields):
    fields.setdefault("source_urls", [])
    fields.setdefault("tested_tasks", TASKS)
    fields.setdefault("reused_tasks", [])
    fields.setdefault("known_failures", [])
    fields.setdefault("sensors", ["RGB", "RGB-D", "proprioception", "language"])
    fields.setdefault("current_task_data_used", False)
    fields.setdefault("privileged_state_used", False)
    return fields


def main():
    summary = json.loads((RUN_ROOT / "summary.json").read_text())
    assets = [base(
        asset_id="tool.language-query-local-fallback.v2",
        kind="tool", name="Generic local language noun-query fallback after provider failure", version="2",
        status="unit_tested", sensors=["language"],
        source_urls=["local:/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py"],
        implementation="/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py",
        input_schema=["natural-language instruction"], output_schema=["concrete noun phrases", "fallback audit"],
        evidence=str(ROOT / "capability_library/tools/test_language_query_fallback.py"),
        known_failures=["lexical fallback cannot replace full VLM relation reasoning"],
    ), base(
        asset_id="experience.frozen-libero-spatial-v017-state8-frontier.v1",
        kind="experience", name="Frozen LIBERO-Spatial v017 state-8 frontier", version="1",
        status="frozen_candidate", tested_tasks=TASKS, evidence=str(RUN_ROOT / "summary.json"),
        run_root=str(RUN_ROOT), controller_id="generic_rgbd_closed_loop_pick_place_v017:v001",
        state=8, episodes=summary["episodes"], sensor_verified=summary["sensor_verified"],
        evaluator_successes=summary["evaluator_successes"],
        results_consumed_for_iteration=summary["results_consumed_for_iteration"],
        protocol="sealed-generated-controller-transfer-v1; evaluator after full batch",
        known_failures=["tasks 0,1,4,5,7,9 unresolved in state 8", "state variance versus v016"],
    )]
    for task in range(10):
        run = RUN_ROOT / "runs" / f"task{task}_state8_seed7"
        result = json.loads((run / "result.json").read_text())
        scorer = json.loads((run / "_evaluator_only" / "result.json").read_text())
        sensor = bool(result.get("attachment_verified") and (result.get("placement_verification") or {}).get("verified"))
        evaluator = bool(scorer.get("success"))
        if evaluator:
            asset = base(
                asset_id=f"experience.libero-spatial-task{task}-v017-state8-success.v1",
                kind="experience", name=f"LIBERO-Spatial task {task} v017 state 8 success", version="1",
                status="development_validated", tested_tasks=[f"libero_spatial:task_{task}"],
                evidence=str(run / "result.json"), result=str(run / "_evaluator_only/result.json"),
                trajectory=str(run / "trajectory.hdf5") if (run / "trajectory.hdf5").is_file() else None,
                sensor_verified=sensor, evaluator_calls=scorer.get("evaluator_calls"),
            )
        else:
            reason = "sensor_verification_passed_but_evaluator_failed" if sensor else (
                "attachment_not_verified" if not result.get("attachment_verified") else "placement_not_verified"
            )
            asset = base(
                asset_id=f"frontier.failure-libero-spatial-task{task}-v017-state8.v1",
                kind="frontier_failure", name=f"LIBERO-Spatial task {task} v017 state 8 unresolved", version="1",
                status="development_validated", tested_tasks=[f"libero_spatial:task_{task}"],
                evidence=str(run / "result.json"), result=str(run / "_evaluator_only/result.json"),
                sensor_verified=sensor, evaluator_success=evaluator, failure_reason=reason,
                known_failures=[reason],
            )
        if asset.get("trajectory") is None:
            asset.pop("trajectory", None)
        assets.append(asset)
    for asset in assets:
        out = register_asset(asset, library_path=str(LIBRARY))
        if not out.get("success"):
            raise RuntimeError(out)
        print(out["asset_id"])


if __name__ == "__main__":
    main()
