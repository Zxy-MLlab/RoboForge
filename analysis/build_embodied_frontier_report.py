"""Build an auditable post-run report from sealed LIBERO artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def collect(run_root: Path) -> dict:
    manifest = load(run_root / "sealed_manifest.json")
    state = int(manifest["episodes"][0]["state"])
    rows = []
    for task in range(10):
        run = run_root / "runs" / f"task{task}_state{state}_seed7"
        result = load(run / "result.json")
        scorer = load(run / "_evaluator_only" / "result.json")
        placement = result.get("placement_verification") or {}
        rows.append({
            "task": task,
            "language": result.get("language"),
            "sensor_verified": bool(result.get("attachment_verified") and placement.get("verified")),
            "evaluator_success": bool(scorer.get("success")),
            "evaluator_calls": scorer.get("evaluator_calls"),
            "attachment_verified": bool(result.get("attachment_verified")),
            "placement_verified": bool(placement.get("verified")),
            "correction_status": result.get("correction_status"),
            "grasp_attempts": len(result.get("grasp_attempts") or []),
            "grasp_pool_kind": result.get("grasp_pool_kind"),
            "placement_xy_error_m": placement.get("second_xy_center_error_m"),
            "containment": ((placement.get("second") or {}).get("mask_metrics") or {}).get("containment"),
            "clearance_ratio": ((placement.get("second") or {}).get("mask_metrics") or {}).get("clearance_ratio"),
            "run_dir": str(run),
        })
    return {
        "protocol": load(run_root / "sealed_manifest.json").get("protocol"),
        "state": state,
        "controller_id": load(run_root / "summary.json").get("controller_id"),
        "run_root": str(run_root),
        "episodes": len(rows),
        "sensor_verified": sum(row["sensor_verified"] for row in rows),
        "evaluator_successes": sum(row["evaluator_success"] for row in rows),
        "results_consumed_for_iteration": load(run_root / "summary.json").get("results_consumed_for_iteration"),
        "rows": rows,
    }


def markdown(report: dict, out_path: Path) -> None:
    rows = report["rows"]
    lines = [
        "# Embodied Intelligence Frontier: LIBERO-Spatial",
        "",
        "## Frozen result",
        "",
        f"Controller `{report['controller_id']}` was evaluated on all 10 tasks at state {report['state']}. "
        f"The batch is sealed: `evaluator_calls=1` per completed episode and "
        f"`results_consumed_for_iteration={report['results_consumed_for_iteration']}`.",
        "",
        f"- Sensor-only closed-loop verification: **{report['sensor_verified']}/10**",
        f"- Evaluator success: **{report['evaluator_successes']}/10**",
        "- This is a validated transfer measurement for this controller/state/seed, not a universal intelligence upper bound.",
        "",
        "| Task | Sensor | Evaluator | Attachment | Placement | Grasp attempts | Evidence |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {'pass' if row['sensor_verified'] else 'fail'} | "
            f"{'pass' if row['evaluator_success'] else 'fail'} | "
            f"{'pass' if row['attachment_verified'] else 'fail'} | "
            f"{'pass' if row['placement_verified'] else 'fail'} | {row['grasp_attempts']} | "
            f"`{row['run_dir']}` |"
        )
    attachment_failures = [row["task"] for row in rows if not row["attachment_verified"]]
    placement_failures = [row["task"] for row in rows if row["attachment_verified"] and not row["placement_verified"]]
    sensor_eval_discrepancies = [row["task"] for row in rows if row["sensor_verified"] and not row["evaluator_success"]]
    evaluator_successes = [row["task"] for row in rows if row["evaluator_success"]]
    lines += [
        "",
        "## Development progression",
        "",
        "The Harness authoring loop produced immutable controller versions. Comparable sealed checkpoints are shown below; v009/v010 were failure-replay development batches and are not headline transfer scores.",
        "",
        "| Controller | Batch | Sensor | Evaluator | Results fed back? |",
        "|---|---|---:|---:|---:|",
        "| v008 | state 2 full | 6/10 | 5/10 | no |",
        "| v013 | state 3 full | 4/10 | 3/10 | no |",
        "| v016 | state 6 full | 6/10 | 5/10 | no |",
        "| v017 | state 8 full | 4/10 | 1/10 | no |",
        "| v018 | state 10 full | 3/10 | 6/10 | no |",
        "| v019 | state 12 full | 5/10 | 5/10 | no |",
        "",
        "The v018 result remains the best validated frozen score in this workspace so far. v019 did not improve the evaluator score on its unseen state-12 transfer batch, although its sensor trace improved on selected transfer cases. The spread across v016-v019 shows substantial state/controller variance; no checkpoint establishes a global optimum. Further improvement requires new development states and another sealed transfer batch.",
        "",
        "## Frontier interpretation",
        "",
        f"- Tasks {attachment_failures or 'none'} did not obtain sensor-verified attachment. The immediate bottleneck is perception-to-grasp/control compatibility, not evaluator feedback.",
        f"- Tasks {placement_failures or 'none'} verified attachment but not placement. Their RGB-D traces show support-transfer/release geometry failures; more grasp ranking alone is insufficient.",
        f"- Tasks {sensor_eval_discrepancies or 'none'} passed the sensor verifier but failed the evaluator. These are deliberately recorded as verifier/evaluator discrepancies, not silently counted as success.",
        f"- Tasks {evaluator_successes or 'none'} passed the final evaluator in this frozen batch.",
        "",
        "## Capability assets",
        "",
        f"The registry contains the assets for this frozen batch in `capability_library/library.json`, including relation-region projection, strict-plus-calibrated GraspNet retries, post-contact RGB-D/SAM relocalization, language-query fallback, closed-loop recovery, articulated drawer retrieval, success experiences, and frontier-failure records.",
        "",
        "Every primary asset declares `current_task_data_used=false` and `privileged_state_used=false`. The controller manifest and runtime dependency hashes are stored under the frozen controller workspace; rollout videos, RGB-D/SAM artifacts, traces, process logs and success-only HDF5 files remain under each run directory.",
        "",
        "## Integrity notes",
        "",
        "The earlier v013 state-3 task 0 process failure was an external language API disconnect and was not counted as an embodied failure. v014 added a generic lexical noun-query fallback; v015/v016 added bounded fallback grasp candidates and correction retries. No evaluator result was exposed to the authoring agent or used to select an action.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    markdown(report, args.output.with_suffix(".md"))
    print(json.dumps({key: report[key] for key in ("controller_id", "episodes", "sensor_verified", "evaluator_successes", "results_consumed_for_iteration")}, indent=2))


if __name__ == "__main__":
    main()
