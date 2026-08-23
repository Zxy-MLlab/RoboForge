from types import SimpleNamespace

from libero_language_policy import make_execute_language_policy


class Episode:
    instruction = "pick the bowl"

    def __init__(self):
        self.actions = []
        self.last_transition = None

    def observe(self):
        return {"agentview_image": "rgb", "reward": "must-not-be-forwarded"}

    def step(self, action):
        self.actions.append(action)
        return SimpleNamespace(
            reward=0.0,
            task_success=False,
            terminated=False,
            truncated=False,
        )


def test_tool_bounds_action_chunk_and_forwards_only_observation_and_language():
    calls = []

    def infer(observation, instruction):
        calls.append((observation, instruction))
        return range(5)

    spec = make_execute_language_policy(infer, max_actions_per_call=2)
    episode = Episode()
    result = spec.executor(episode, {})

    assert calls == [({"agentview_image": "rgb"}, "pick the bowl")]
    assert episode.actions == [0, 1]
    assert result["actions_executed"] == 2
