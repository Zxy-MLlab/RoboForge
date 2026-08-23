"""Real-model smoke test for the standalone evolution loop (no benchmark)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from embodied_harness.evolution import EvolutionEngine
from embodied_harness.model import OpenAICompatibleModel


class FakeRobotAdapter:
    """Sensor-only toy robot; episode one is obstructed, later episodes move."""

    def __init__(self, episode: int) -> None:
        self.episode = episode
        self.x = 0.0

    @property
    def initial_context(self):
        return {"task_instruction": "move the end effector one unit to the right"}

    def dispatch(self, method, arguments):
        if method == "instruction":
            return self.initial_context["task_instruction"]
        if method == "sense":
            return {
                "frame_id": f"episode-{self.episode}-rgbd-1",
                "eef_x": self.x, "target_delta_x": 1.0,
                "channels": ["rgb", "depth", "proprioception"],
            }
        if method == "act":
            action = dict(arguments.get("action") or {})
            if self.episode > 1:
                self.x = float(action.get("target_x", action.get("x", 1.0)))
            return {
                "accepted": True, "reached": self.episode > 1,
                "observed_eef_x": self.x,
                "sensor_reason": None if self.episode > 1 else "path_obstructed",
            }
        if method == "verify":
            verifier = str(arguments.get("verifier") or "")
            if verifier in {"scene_visible", "observation_valid", "rgbd_available"}:
                verified = True
            else:
                verified = self.x >= 0.95
            return {"verified": verified, "observed_eef_x": self.x,
                    "verifier": verifier}
        if method == "record":
            return {"recorded": True}
        if method == "use":
            raise RuntimeError("no capability Tools are installed in this toy deployment")
        raise ValueError(f"unsupported method: {method}")

    def sensor_report(self, execution):
        return {
            "sensor_verification_passed": (
                execution.get("graph_outcome") == "success" and self.x >= 0.95
            ),
            "observed_eef_x": self.x,
        }

    def close(self):
        return None


class Factory:
    def __init__(self): self.episodes = 0
    def __call__(self):
        self.episodes += 1
        return FakeRobotAdapter(self.episodes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://api.apexin.ai/v1")
    args = parser.parse_args()
    api_key = os.environ.get("APEX_API_KEY")
    if not api_key:
        raise SystemExit("APEX_API_KEY is not set")
    model = OpenAICompatibleModel(
        api_key=api_key, base_url=args.base_url, model=args.model,
        reasoning_effort="low", max_tokens=6000,
    )
    engine = EvolutionEngine(
        root=Path(args.run_dir), model=model, adapter_factory=Factory(),
        available_initial_fields={"task_instruction"},
        python="/data/zxy/envs/vla-report/bin/python", max_agent_turns=40,
    )
    state = engine.run(
        task="move the end effector one unit right using live sensing; recover from failure",
        skill_name="sensor_guided_move_right", max_rounds=args.max_rounds,
    )
    print(json.dumps({
        "status": state["status"], "round_count": len(state["rounds"]),
        "rounds": state["rounds"], "skill": state.get("skill"),
    }, indent=2))
    return 0 if state["status"] == "sensor_success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
