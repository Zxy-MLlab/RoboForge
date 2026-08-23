"""Execute a predeclared immutable-controller manifest, then score it once.

All episodes run through ControllerWorkspace and expose sensor evidence only.
The evaluator-only files are read only after every declared episode finishes,
so no result can alter a later controller or action in the sealed batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller_harness import ControllerWorkspace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("protocol") != "sealed-generated-controller-transfer-v1":
        raise ValueError("unsupported sealed manifest protocol")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"sealed output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    sealed_copy = args.output / "sealed_manifest.json"
    sealed_copy.write_bytes(manifest_bytes)

    root = args.manifest.resolve().parents[1]
    workspace_path = root / str(manifest["controller_workspace"])
    workspace = ControllerWorkspace(workspace_path)
    controller_id = str(manifest["controller_id"])
    controller_path = workspace.resolve(controller_id)
    controller_manifest_bytes = (controller_path / "manifest.json").read_bytes()
    seal = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "controller_manifest_sha256": hashlib.sha256(controller_manifest_bytes).hexdigest(),
        "controller_sha256": hashlib.sha256((controller_path / "controller.py").read_bytes()).hexdigest(),
        "episodes": len(manifest["episodes"]),
        "evaluator_opened_during_execution": False,
    }
    authoring_skill = manifest.get("authoring_skill")
    if authoring_skill:
        skill_path = root / str(authoring_skill)
        seal["authoring_skill"] = str(authoring_skill)
        seal["authoring_skill_sha256"] = hashlib.sha256(skill_path.read_bytes()).hexdigest()
    authoring_skills = manifest.get("authoring_skills") or []
    if authoring_skills:
        seal["authoring_skills"] = {
            str(item): hashlib.sha256((root / str(item)).read_bytes()).hexdigest()
            for item in authoring_skills
        }
    (args.output / "seal.json").write_text(json.dumps(seal, indent=2) + "\n")

    sensor_results = []
    for episode in manifest["episodes"]:
        report = workspace.execute(
            controller_id,
            suite=str(manifest["suite"]),
            task=int(episode["task"]),
            state=int(episode["state"]),
            seed=int(episode.get("seed", 7)),
            output_root=args.output / "runs",
        )
        sensor_results.append({
            "task": int(episode["task"]),
            "state": int(episode["state"]),
            "seed": int(episode.get("seed", 7)),
            "execution_success": bool(report["success"]),
            "run_dir": report["run_dir"],
            "sensor_evidence": report["sensor_evidence"],
            "evaluator_hidden": True,
        })
    (args.output / "sensor_results.json").write_text(json.dumps(sensor_results, indent=2) + "\n")

    # Scoring barrier: evaluator files are opened only after the full batch.
    scored = []
    for item in sensor_results:
        scorer_path = Path(item["run_dir"]) / "_evaluator_only" / "result.json"
        scorer = json.loads(scorer_path.read_text()) if scorer_path.is_file() else {"success": False, "missing": True}
        scored.append({
            "task": item["task"], "state": item["state"], "seed": item["seed"],
            "sensor_verified": item["sensor_evidence"].get("sensor_only_conclusion") == "sensor_verification_passed",
            "evaluator_success": bool(scorer.get("success")),
            "evaluator_calls": scorer.get("evaluator_calls"),
        })
    summary = {
        "protocol": manifest["protocol"],
        "controller_id": controller_id,
        "manifest_sha256": seal["manifest_sha256"],
        "episodes": len(scored),
        "sensor_verified": sum(bool(item["sensor_verified"]) for item in scored),
        "evaluator_successes": sum(bool(item["evaluator_success"]) for item in scored),
        "results_consumed_for_iteration": bool(
            manifest.get("results_consumed_for_iteration", False)
        ),
        "results": scored,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
