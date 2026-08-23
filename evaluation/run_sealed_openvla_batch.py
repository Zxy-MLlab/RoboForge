"""Run one frozen OpenVLA transfer candidate on the sealed LIBERO-Object surface."""

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
    parser.add_argument("--states-per-task", type=int, default=50)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--policy", choices=("openvla_base",), default="openvla_base")
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    args = parser.parse_args()
    gpu_ids = [int(item.strip()) for item in str(args.gpu_ids).split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one device")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (task, state)
        for task in range(10)
        for state in range(args.states_per_task)
    ]

    def run(job: tuple[int, int]) -> dict[str, object]:
        task, state = job
        episode_dir = args.output / f"task_{task:02d}" / f"state_{state:02d}"
        result_file = episode_dir / "result.json"
        if result_file.exists():
            return json.loads(result_file.read_text())
        env = os.environ.copy()
        env.update(
            {
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
        gpu = gpu_ids[(task * args.states_per_task + state) % len(gpu_ids)]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # robosuite validates this against CUDA_VISIBLE_DEVICES literally; keep
        # the physical device id instead of remapping it to logical zero.
        env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        command = [
            sys.executable,
            str(root / "evaluation" / "run_thea_code_libero.py"),
            "--suite", args.suite,
            "--task", str(task),
            "--state", str(state),
            "--policy", args.policy,
            "--output", str(episode_dir),
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        if result_file.exists():
            result = json.loads(result_file.read_text())
            result["runner_returncode"] = completed.returncode
            if completed.returncode != 0:
                result["runner_stderr_tail"] = completed.stderr[-2000:]
            result_file.write_text(json.dumps(result, indent=2) + "\n")
            return result
        episode_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "protocol": "harness-acquired-task-zero-shot-v2",
            "suite": args.suite,
            "task_index": task,
            "initial_state_index": state,
            "policy": args.policy,
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
                errors = sum("integration_error" in item for item in results)
                print(f"completed={index}/{len(jobs)} success={successes} integration_errors={errors}", flush=True)

    results.sort(key=lambda item: (int(item["task_index"]), int(item["initial_state_index"])))
    per_task = {}
    for task in range(10):
        items = [item for item in results if int(item["task_index"]) == task]
        per_task[str(task)] = {
            "successes": sum(bool(item.get("success")) for item in items),
            "episodes": len(items),
            "integration_errors": sum("integration_error" in item for item in items),
        }
    summary = {
        "protocol": "harness-acquired-task-zero-shot-v2",
        "sealed_manifest": "manifests/libero_sealed_eval_v2.json",
        "suite": args.suite,
        "policy": args.policy,
        "track": "task_disjoint_transfer",
        "claimable": True,
        "episodes": len(results),
        "successes": sum(bool(item.get("success")) for item in results),
        "integration_errors": sum("integration_error" in item for item in results),
        "per_task": per_task,
        "sealed_results_consumed_for_iteration": False,
        "learned_models_used": ["openvla/openvla-7b"],
        "action_normalization": "nyu_franka_play_dataset_converted_externally_to_rlds",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
