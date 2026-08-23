"""Run one real CALVIN development task with evaluator-blind no-op actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import cv2
import numpy as np
from omegaconf import OmegaConf

from calvin_env.envs.play_table_env import get_env
from thea_simulation import CalvinEpisode, project_calvin_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--calvin-conf", type=Path, required=True)
    parser.add_argument("--task", default="turn_on_led")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-output", type=Path)
    args = parser.parse_args()

    task_cfg = OmegaConf.load(
        args.calvin_conf / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    annotations = OmegaConf.load(
        args.calvin_conf / "annotations/new_playtable_validation.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    instruction = str(annotations[args.task][0])
    env = get_env(
        args.dataset_config,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
        show_gui=False,
    )
    evaluator_state = {}

    def success_fn(target_env, _observation, info):
        solved = task_oracle.get_task_info_for_set(
            evaluator_state["start_info"], info, {args.task}
        )
        return bool(solved)

    episode = CalvinEpisode(
        env,
        task_id=f"calvin_dev:{args.task}",
        instruction=instruction,
        success_fn=success_fn,
    )
    actions_executed = 0
    try:
        observation = episode.reset(seed=0)
        if args.frame_output is not None:
            args.frame_output.parent.mkdir(parents=True, exist_ok=True)
            frame = observation["rgb_obs"]["rgb_static"]
            cv2.imwrite(str(args.frame_output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        evaluator_state["start_info"] = env.get_info()
        projected = project_calvin_observation(observation, turn=0)
        transition = None
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        for _ in range(args.steps):
            transition = episode.step(action)
            actions_executed += 1
            if transition.task_success:
                break
        assert transition is not None
        report = {
            "protocol": "harness-acquired-task-zero-shot-v2",
            "benchmark": "CALVIN",
            "surface": "development_only",
            "claimable": False,
            "task": args.task,
            "instruction": instruction,
            "policy": "fixed_hold_pose_diagnostic",
            "learned_models_used": [],
            "camera_names": [visual.name for visual in projected.visuals],
            "actions_executed": actions_executed,
            "task_success": transition.task_success,
            "failure_kind": None if transition.task_success else "missing_task_skill",
            "success_signal_used_for_action_selection": False,
            "privileged_state_used_for_action_selection": False,
            "sealed_results_consumed_for_iteration": False,
        }
    finally:
        episode.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
