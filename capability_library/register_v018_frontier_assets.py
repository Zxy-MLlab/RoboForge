"""Register the sealed v018 LIBERO-Spatial result and per-task outcomes."""

from __future__ import annotations

import json
from pathlib import Path

from asset_registry import register_asset

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "capability_library" / "library.json"
RUN_ROOT = ROOT / "runs" / "coding_harness" / "libero_spatial_v018_state10_full"
TASKS = [f"libero_spatial:task_{i}" for i in range(10)]


def base(**fields):
    fields.setdefault("source_urls", [])
    fields.setdefault("tested_tasks", TASKS)
    fields.setdefault("reused_tasks", [])
    fields.setdefault("sensors", ["RGB", "RGB-D", "proprioception", "language"])
    fields.setdefault("current_task_data_used", False)
    fields.setdefault("privileged_state_used", False)
    return fields


def main():
    summary = json.loads((RUN_ROOT / "summary.json").read_text())
    assets = [base(
        asset_id="experience.frozen-libero-spatial-v018-state10-frontier.v1",
        kind="experience", name="Frozen LIBERO-Spatial v018 state-10 frontier", version="1",
        status="frozen_candidate", evidence=str(RUN_ROOT / "summary.json"),
        run_root=str(RUN_ROOT), controller_id=summary["controller_id"], state=10,
        episodes=summary["episodes"], sensor_verified=summary["sensor_verified"],
        evaluator_successes=summary["evaluator_successes"],
        results_consumed_for_iteration=summary["results_consumed_for_iteration"],
        protocol="sealed-generated-controller-transfer-v1; evaluator after full batch",
        known_failures=["tasks 3,5,7,8 unresolved", "sensor verifier is conservative on tasks 1,6,9"],
    )]
    for task in range(10):
        run = RUN_ROOT / "runs" / f"task{task}_state10_seed7"
        result = json.loads((run / "result.json").read_text())
        scorer = json.loads((run / "_evaluator_only" / "result.json").read_text())
        sensor = bool(result.get("attachment_verified") and (result.get("placement_verification") or {}).get("verified"))
        success = bool(scorer.get("success"))
        prefix = "experience" if success else "frontier.failure"
        assets.append(base(
            asset_id=f"{prefix}.libero-spatial-task{task}-v018-state10.v1",
            kind="experience" if success else "frontier_failure",
            name=f"LIBERO-Spatial task {task} v018 state 10 {'success' if success else 'unresolved'}",
            version="1", status="development_validated", tested_tasks=[f"libero_spatial:task_{task}"],
            evidence=str(run / "result.json"), result=str(run / "_evaluator_only/result.json"),
            sensor_verified=sensor, evaluator_success=success,
            evaluator_calls=scorer.get("evaluator_calls"),
            failure_reason=None if success else ("attachment_not_verified" if not result.get("attachment_verified") else "placement_not_verified"),
        ))
    for asset in assets:
        asset = {k: v for k, v in asset.items() if v is not None}
        out = register_asset(asset, library_path=str(LIBRARY))
        if not out.get("success"):
            raise RuntimeError(out)
        print(out["asset_id"])


if __name__ == "__main__":
    main()
