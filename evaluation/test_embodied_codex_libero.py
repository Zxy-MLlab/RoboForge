from pathlib import Path

import pytest

from run_embodied_codex_libero import (
    _task_list,
    development_command,
    validation_command,
)
from run_autonomous_evolution import _run_visible_harness_stream


def test_task_range_parser_is_deterministic():
    assert _task_list("3,5-7,5") == [3, 5, 6, 7]
    with pytest.raises(Exception):
        _task_list("10")


def test_canonical_command_exposes_only_complete_program_controller():
    command = development_command(
        task=4, state=23, seed=7, max_rounds=8,
        max_turns_per_round=20, max_turns_acquisition=20,
        acquisition_after_same_failure=2,
        output=Path("run"), controllers=Path("controllers"),
        stage_nodes=Path("stage_nodes"),
        capabilities=Path("tools"), task_skills=Path("skills"),
        force_acquisition=False,
    )
    assert command[command.index("--controller-interface") + 1] == "graph"
    assert "--stage-node-workspace" in command
    assert "--capability-workspace" in command
    assert "--task-skill-workspace" in command


def test_validation_command_requires_three_state_runner_inputs():
    command = validation_command(
        skill_id="learned_transfer:v001", task=5, count=3, seed=7,
        task_skills=Path("skills"), output=Path("validation"),
    )
    assert command[command.index("--count") + 1] == "3"
    assert command[command.index("--skill-id") + 1] == "learned_transfer:v001"


def test_agent_stream_is_persisted_before_session_completion(tmp_path, capsys):
    class FakeHarness:
        def run_stream(self, instruction, **kwargs):
            assert instruction == "work"
            assert kwargs["max_turns"] == 3
            yield {"type": "tool_call", "name": "create_controller_program"}
            yield {
                "type": "tool_result", "name": "create_controller_program",
                "success": False, "result": {"reason": "audit detail"},
            }

    path = tmp_path / "live.jsonl"
    events = _run_visible_harness_stream(
        FakeHarness(), "work", live_trace=path, max_turns=3,
        failure_budget=2, system_prompt_override="system",
    )
    assert len(path.read_text().splitlines()) == len(events) == 2
    output = capsys.readouterr().out
    assert "create_controller_program" in output
    assert "success=false" in output
