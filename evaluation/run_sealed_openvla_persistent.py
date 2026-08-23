"""Launch persistent OpenVLA workers for one frozen sealed manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="1,2")
    parser.add_argument("--states-per-task", type=int, default=50)
    parser.add_argument("--suite", default="libero_object")
    args = parser.parse_args()
    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [(task, state) for task in range(10) for state in range(args.states_per_task)]
    jobs_dir = args.output / "worker_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    processes = []
    for index, gpu in enumerate(gpu_ids):
        worker_jobs = jobs[index::len(gpu_ids)]
        job_file = jobs_dir / f"worker_{index}.json"
        job_file.write_text(json.dumps(worker_jobs))
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "MUJOCO_EGL_DEVICE_ID": str(gpu),
                "MUJOCO_GL": "egl",
                "LIBERO_CONFIG_PATH": "/data/zxy/embodied_frontier/runtime_home/.libero",
                "PYTHONPATH": ":".join(
                    [
                        "/data/zxy/LIBERO",
                        str(root / "capability_library"),
                        str(root / "capability_library" / "tools"),
                        str(root / "Thea"),
                        str(root / "Thea" / "simulation"),
                        env.get("PYTHONPATH", ""),
                    ]
                ),
            }
        )
        command = [
            sys.executable,
            str(root / "evaluation" / "run_openvla_sealed_worker.py"),
            "--jobs", str(job_file),
            "--output", str(args.output),
            "--device", "cuda:0",
            "--suite", args.suite,
        ]
        processes.append(subprocess.Popen(command, env=env, text=True))
    return_codes = [process.wait() for process in processes]
    results = []
    for task, state in jobs:
        path = args.output / f"task_{task:02d}" / f"state_{state:02d}" / "result.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
    results.sort(key=lambda item: (int(item["task_index"]), int(item["initial_state_index"])))
    per_task = {
        str(task): {
            "successes": sum(bool(item.get("success")) for item in results if int(item.get("task_index", -1)) == task),
            "episodes": sum(int(item.get("task_index", -1)) == task for item in results),
            "integration_errors": sum("integration_error" in item for item in results if int(item.get("task_index", -1)) == task),
        }
        for task in range(10)
    }
    summary = {
        "protocol": "harness-acquired-task-zero-shot-v2",
        "sealed_manifest": "manifests/libero_sealed_eval_v2.json",
        "suite": args.suite,
        "candidate": "openvla/openvla-7b",
        "track": "task_disjoint_transfer",
        "claimable": len(results) == len(jobs) and all(code == 0 for code in return_codes),
        "episodes": len(results),
        "successes": sum(bool(item.get("success")) for item in results),
        "integration_errors": sum("integration_error" in item for item in results),
        "per_task": per_task,
        "worker_return_codes": return_codes,
        "sealed_results_consumed_for_iteration": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["claimable"] else 2)


if __name__ == "__main__":
    main()
