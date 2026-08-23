"""Execute a hash-frozen controller-program manifest behind a scoring barrier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller_program_workspace import ControllerProgramWorkspace
from libero_robot_sdk import execute_libero_program_sealed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("protocol") != "sealed-controller-program-v1":
        raise ValueError("unsupported sealed program manifest protocol")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"sealed output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sealed_manifest.json").write_bytes(manifest_bytes)

    root = args.manifest.resolve().parents[1]
    workspace = ControllerProgramWorkspace(
        root / manifest["controller_workspace"],
        python="/data/zxy/envs/vla-report/bin/python",
    )
    program_id = str(manifest["program_id"])
    program_dir = workspace.resolve(program_id)
    seal = {
        "protocol": manifest["protocol"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "program_id": program_id,
        "program_sha256": _sha256(program_dir / "program.py"),
        "program_manifest_sha256": _sha256(program_dir / "manifest.json"),
        "episodes": len(manifest["episodes"]),
        "evaluator_opened_during_execution": False,
        "results_consumed_for_iteration": False,
    }
    (args.output / "seal.json").write_text(json.dumps(seal, indent=2) + "\n")

    sensor_results = []
    for index, episode in enumerate(manifest["episodes"], 1):
        run_dir = args.output / "runs" / f"episode_{index:03d}"
        report = execute_libero_program_sealed(
            workspace, program_id,
            suite=str(manifest["suite"]), task=int(episode["task"]),
            state=int(episode["state"]), seed=int(episode.get("seed", 7)),
            output=run_dir,
            capability_workspace=root / manifest["capability_workspace"],
        )
        sensor_results.append({
            "task": int(episode["task"]), "state": int(episode["state"]),
            "seed": int(episode.get("seed", 7)), "run_dir": str(run_dir),
            "sensor_evidence": report["sensor_evidence"],
            "evaluator_result_path": report["evaluator_result_path"],
            "evaluator_hidden_during_execution": True,
        })
    (args.output / "sensor_results.json").write_text(
        json.dumps(sensor_results, indent=2) + "\n"
    )

    # Scoring barrier: only this post-batch phase opens evaluator-only files.
    scored = []
    for item in sensor_results:
        evaluator = json.loads(Path(item["evaluator_result_path"]).read_text())
        scored.append({
            "task": item["task"], "state": item["state"], "seed": item["seed"],
            "sensor_verified": item["sensor_evidence"].get("sensor_only_conclusion")
                == "sensor_verification_passed",
            "evaluator_success": bool(evaluator.get("success")),
            "evaluator_calls": int(evaluator.get("evaluator_calls", 0)),
        })
    summary = {
        "protocol": manifest["protocol"], "program_id": program_id,
        "manifest_sha256": seal["manifest_sha256"], "episodes": len(scored),
        "sensor_verified": sum(bool(row["sensor_verified"]) for row in scored),
        "evaluator_successes": sum(bool(row["evaluator_success"]) for row in scored),
        "results_consumed_for_iteration": False, "results": scored,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
