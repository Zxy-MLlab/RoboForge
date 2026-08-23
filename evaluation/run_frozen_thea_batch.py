"""Run the frozen Thea code controller over a manifest with isolated workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(task, state) for task in range(1, 10) for state in range(50)]

    def run(job: tuple[int, int]) -> dict[str, object]:
        task, state = job
        episode_dir = args.output / f"task_{task:02d}" / f"state_{state:02d}"
        result_file = episode_dir / "result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["MUJOCO_EGL_DEVICE_ID"] = str((task * 50 + state) % args.workers)
        command = [
            sys.executable,
            str(root / "evaluation/run_thea_code_libero.py"),
            "--task", str(task),
            "--state", str(state),
            "--output", str(episode_dir),
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        if completed.returncode == 0 and result_file.exists():
            return json.loads(result_file.read_text())
        episode_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "protocol": "strict-code-zero-shot-v1",
            "task_index": task,
            "initial_state_index": state,
            "success": False,
            "claimable": True,
            "integration_error": completed.stderr[-4000:] or completed.stdout[-4000:],
        }
        result_file.write_text(json.dumps(failure, indent=2) + "\n")
        return failure

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 10 == 0 or index == len(jobs):
                successes = sum(bool(item.get("success")) for item in results)
                print(f"completed={index}/{len(jobs)} success={successes}", flush=True)

    results.sort(key=lambda item: (int(item["task_index"]), int(item["initial_state_index"])))
    per_task = {}
    for task in range(1, 10):
        task_results = [item for item in results if item["task_index"] == task]
        per_task[str(task)] = {
            "successes": sum(bool(item.get("success")) for item in task_results),
            "episodes": len(task_results),
            "integration_errors": sum("integration_error" in item for item in task_results),
        }
    summary = {
        "protocol": "strict-code-zero-shot-v1",
        "claimable": True,
        "episodes": len(results),
        "successes": sum(bool(item.get("success")) for item in results),
        "per_task": per_task,
        "learned_models_used": [],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
