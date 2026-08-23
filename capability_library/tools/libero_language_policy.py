"""Thea tool adapter for a frozen language-conditioned LIBERO policy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from thea_simulation import SimulationEpisode, action_chunk_tool


PolicyInfer = Callable[[Mapping[str, Any], str], Iterable[Any]]

_EVALUATOR_ONLY_KEYS = frozenset(
    {
        "reward",
        "success",
        "is_success",
        "task_success",
        "done",
        "terminated",
        "truncated",
        "info",
        "task_id",
        "init_state_id",
        "bddl_goal_predicates",
    }
)


def _policy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in observation.items()
        if str(key).lower() not in _EVALUATOR_ONLY_KEYS
    }


def make_execute_language_policy(
    infer: PolicyInfer,
    *,
    max_actions_per_call: int = 50,
):
    """Create a Thea policy tool without exposing evaluator-only state.

    ``infer`` receives only the current raw policy observation and the episode's
    natural-language instruction. It must return a frozen action chunk. Success,
    reward, termination details, task IDs, and simulator internals are not passed
    into policy inference.
    """
    if max_actions_per_call < 1:
        raise ValueError("max_actions_per_call must be positive")

    def policy(episode: SimulationEpisode, arguments: Mapping[str, Any]):
        del arguments
        actions = iter(infer(_policy_observation(episode.observe()), episode.instruction))
        for index, action in enumerate(actions):
            if index >= max_actions_per_call:
                break
            yield action

    return action_chunk_tool(
        name="execute_language_policy",
        description=(
            "Execute one bounded action chunk from the registered frozen "
            "language-conditioned manipulation policy."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        post_condition="The benchmark task success condition is satisfied.",
        policy=policy,
    )


__all__ = ["PolicyInfer", "make_execute_language_policy"]
