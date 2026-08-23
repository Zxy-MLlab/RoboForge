"""Run a real evaluator-blind CALVIN reset/observation/action smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calvin_env.envs.play_table_env import get_env
from thea_simulation import CalvinEpisode, project_calvin_observation


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = args.dataset_config / ".hydra" / "merged_config.yaml"
    env = get_env(
        args.dataset_config,
        obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []},
        show_gui=False,
    )
    episode = CalvinEpisode(
        env,
        task_id="calvin_environment_smoke",
        instruction="Hold the current pose.",
    )
    try:
        observation = episode.reset(seed=0)
        projected = project_calvin_observation(observation, turn=0)
        transition = episode.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
        report = {
            "protocol": "harness-acquired-task-zero-shot-v2",
            "benchmark": "CALVIN",
            "surface": "environment_integration_smoke_not_task_metric",
            "claimable_task_result": False,
            "dataset_config": str(args.dataset_config),
            "merged_config_sha256": sha256(config),
            "reset_succeeded": True,
            "camera_names": [visual.name for visual in projected.visuals],
            "visual_count": len(projected.visuals),
            "action_dtype": "float64",
            "action_shape": [7],
            "action_accepted": True,
            "transition_reward_evaluator_only": transition.reward,
            "task_success_evaluator_only": transition.task_success,
            "agent_observation_contains_reward": "reward" in episode.observe(),
            "agent_observation_contains_success": any(
                key in episode.observe() for key in ("success", "task_success")
            ),
            "sealed_results_consumed_for_iteration": False,
        }
    finally:
        episode.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
