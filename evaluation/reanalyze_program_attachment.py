"""Recheck historical attachment claims against an adapter-owned baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from libero_robot_sdk import locked_attachment_verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-trace", type=Path, required=True)
    parser.add_argument("--round-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime_trace.read_text())
    report = json.loads(args.round_report.read_text())
    baseline = None
    for event in runtime.get("rpc_events") or ():
        if event.get("method") != "call_tool":
            continue
        arguments = event.get("arguments") or {}
        result = event.get("result") or {}
        if arguments.get("name") == "select_entities" and (result.get("source") or {}).get("xyz"):
            baseline = result["source"]["xyz"]
            break
    if baseline is None:
        raise RuntimeError("no adapter-selected source baseline in runtime trace")
    rechecks = []
    for evidence in (report.get("sensor_evidence") or {}).get("verifications") or ():
        if evidence.get("kind") != "attachment":
            continue
        hardened = locked_attachment_verification(
            evidence.get("object_xyz"), evidence.get("eef_xyz"), baseline,
            evidence.get("gripper_width"), [evidence.get("object_xyz")],
        )
        rechecks.append({
            "frame_id": evidence.get("frame_id"),
            "original_verified": evidence.get("verified"),
            "hardened_verified": hardened["verified"],
            "source_vacated": hardened["source_vacated"],
            "adapter_owned_source_baseline_xyz": baseline,
            "controller_supplied_previous_xyz": evidence.get("previous_object_xyz"),
            "observed_object_xyz": evidence.get("object_xyz"),
        })
    result = {
        "protocol": "locked-attachment-reanalysis-v1",
        "original_report": str(args.round_report.resolve()),
        "original_attachment_verified": (report.get("sensor_evidence") or {}).get("attachment_verified"),
        "hardened_attachment_verified": any(item["hardened_verified"] for item in rechecks),
        "evaluator_used": False,
        "rechecks": rechecks,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
