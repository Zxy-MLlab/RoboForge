"""Audit the non-privileged LIBERO sensor boundary and save RGB-D previews.

This development utility intentionally exports only camera calibration, RGB-D,
and robot proprioception. Object-state observations and evaluator signals are
listed as rejected keys but are never serialized.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)


CAMERAS = ("agentview", "robot0_eye_in_hand")
ALLOWED_STATE_KEYS = (
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/sensor_audit/libero_spatial_t0_s0")
    )
    args = parser.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_names=list(CAMERAS),
        camera_heights=args.size,
        camera_widths=args.size,
        camera_depths=True,
    )
    try:
        observation = env.reset()
        observation = env.set_init_state(suite.get_task_init_states(args.task)[args.state])
        args.output.mkdir(parents=True, exist_ok=True)

        cameras: dict[str, object] = {}
        for camera in CAMERAS:
            # LIBERO camera observations use MuJoCo's bottom-up row order.
            rgb = np.ascontiguousarray(observation[f"{camera}_image"][::-1])
            normalized_depth = np.ascontiguousarray(
                observation[f"{camera}_depth"][::-1]
            )
            metric_depth = get_real_depth_map(env.sim, normalized_depth)
            cv2.imwrite(
                str(args.output / f"{camera}_rgb.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            depth_preview = np.clip((metric_depth[..., 0] - 0.4) / 1.6, 0, 1)
            cv2.imwrite(
                str(args.output / f"{camera}_depth.png"),
                (255 * (1 - depth_preview)).astype(np.uint8),
            )
            cameras[camera] = {
                "rgb_shape": list(rgb.shape),
                "metric_depth_shape": list(metric_depth.shape),
                "metric_depth_range_m": [
                    float(metric_depth.min()),
                    float(metric_depth.max()),
                ],
                "intrinsic": get_camera_intrinsic_matrix(
                    env.sim, camera, args.size, args.size
                ).tolist(),
                "camera_to_world": get_camera_extrinsic_matrix(
                    env.sim, camera
                ).tolist(),
            }

        allowed_state = {
            key: np.asarray(observation[key]).tolist() for key in ALLOWED_STATE_KEYS
        }
        rejected = sorted(
            key
            for key in observation
            if key not in ALLOWED_STATE_KEYS
            and not any(
                key == f"{camera}_{suffix}"
                for camera in CAMERAS
                for suffix in ("image", "depth")
            )
        )
        audit = {
            "protocol": "strict-code-zero-shot-v1",
            "suite": args.suite,
            "task_index": args.task,
            "initial_state_index": args.state,
            "instruction": task.language,
            "allowed": {
                "language_instruction": task.language,
                "cameras": cameras,
                "proprioception": allowed_state,
            },
            "rejected_observation_keys": rejected,
            "control": {
                "interface": "OSC_POSE delta + gripper",
                "action_dimension": int(env.env.action_dim),
                "action_low": env.env.action_spec[0].tolist(),
                "action_high": env.env.action_spec[1].tolist(),
            },
            "evaluator_signals_exposed_to_controller": False,
        }
        (args.output / "audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
