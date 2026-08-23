"""Render development-only diagnostics for the classical RGB-D segmenter."""

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

from rgbd_pick_place import (
    allowed_observation,
    annotate_regions,
    backproject_rgbd,
    detect_circular_candidates,
    estimate_table_height,
    parse_pick_place_instruction,
    segment_workspace_regions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
    )
    try:
        raw = env.reset()
        raw = env.set_init_state(suite.get_task_init_states(args.task)[args.state])
        obs = allowed_observation(raw)
        rgb = np.ascontiguousarray(obs["agentview_image"][::-1])
        depth = get_real_depth_map(env.sim, np.ascontiguousarray(obs["agentview_depth"][::-1]))
        intrinsic = get_camera_intrinsic_matrix(env.sim, "agentview", 256, 256)
        extrinsic = get_camera_extrinsic_matrix(env.sim, "agentview")
        world = backproject_rgbd(depth, intrinsic, extrinsic)
        table_height = estimate_table_height(world)
        regions, labels = segment_workspace_regions(rgb, world, table_height=table_height)
        circles = detect_circular_candidates(rgb, world)
        args.output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output / "regions.png"), annotate_regions(rgb, regions))
        cv2.imwrite(str(args.output / "mask.png"), ((labels > 0) * 255).astype(np.uint8))
        report = {
            "instruction": task.language,
            "intent": parse_pick_place_instruction(task.language).__dict__,
            "table_height_m": table_height,
            "regions": [region.__dict__ for region in regions],
            "circular_candidates": [candidate.__dict__ for candidate in circles],
            "inputs": "RGB-D, camera calibration, proprioception only",
        }
        (args.output / "regions.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
