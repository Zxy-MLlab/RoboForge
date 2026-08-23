import json

import pytest

from capability_workspace import CapabilityWorkspace
from controller_program_workspace import ControllerProgramWorkspace
from task_skill_workspace import TaskSkillValidationError, TaskSkillWorkspace


PYTHON = "/data/zxy/envs/vla-report/bin/python"


def _workspaces(tmp_path):
    capabilities = CapabilityWorkspace(tmp_path / "tools", python=PYTHON)
    tool = capabilities.create(
        "relative_waypoint",
        "def run(payload):\n"
        "    return {'target': [payload['origin'][i] + payload['delta'][i] for i in range(3)]}\n",
        "Sensor-relative waypoint",
        input_schema={
            "type": "object", "properties": {
                "origin": {"type": "array"}, "delta": {"type": "array"},
            }, "required": ["origin", "delta"], "additionalProperties": False,
        },
        output_schema={
            "type": "object", "properties": {"target": {"type": "array"}},
            "required": ["target"], "additionalProperties": False,
        },
        stage="planning",
    )
    assert capabilities.test(tool["tool_id"], [{
        "input": {"origin": [1, 2, 3], "delta": [0.1, 0, -0.1]},
        "expected": {"target": [1.1, 2, 2.9]},
    }])["success"]
    controllers = ControllerProgramWorkspace(
        tmp_path / "controllers", python=PYTHON,
        capability_workspace=tmp_path / "tools",
    )
    source = '''def run(robot):
    robot.instruction()
    frame = robot.observe()
    plan = robot.call_tool("relative_waypoint:v001", {
        "origin": frame["eef_xyz"], "delta": [0.0, 0.0, 0.1]})
    robot.act({"target_eef_xyz": plan["target"], "gripper": -1})
'''
    program = controllers.create("relative_task_skill", source)
    skills = TaskSkillWorkspace(
        tmp_path / "skills", controller_workspace=controllers,
        capability_workspace=tmp_path / "tools",
    )
    return skills, program


def test_task_skill_freezes_program_and_tool_then_requires_unseen_states(tmp_path):
    skills, program = _workspaces(tmp_path)
    created = skills.create_candidate(
        name="sensor_grounded_transfer",
        description="Ground and transfer a language-selected object",
        semantic_task="language-conditioned transfer",
        program_id=program["program_id"],
        development_evidence={
            "sensor_only_conclusion": "sensor_verification_passed",
            "attachment_verified": True, "placement_verified": True,
        },
        development_context={
            "environment": "libero_spatial",
            "state_key": "task-5:state-23:seed-7",
        },
    )
    manifest_path = (
        tmp_path / "skills" / "sensor_grounded_transfer" / "v001" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "development_candidate"
    assert manifest["literal_absolute_action_targets"] is False
    assert manifest["dependencies"][0]["tool_id"] == "relative_waypoint:v001"
    for index in range(3):
        result = skills.record_unseen_validation(
            created["skill_id"], environment="libero_spatial",
            state_key=f"heldout-{index}",
            sensor_evidence={"sensor_only_conclusion": "sensor_verification_passed"},
        )
    assert result["status"] == "sensor_validated"


def test_task_skill_executes_its_frozen_program_and_dependencies(tmp_path):
    skills, program = _workspaces(tmp_path)
    created = skills.create_candidate(
        name="executable_sensor_skill", description="Executable frozen skill",
        semantic_task="language-conditioned transfer", program_id=program["program_id"],
        development_evidence={
            "sensor_only_conclusion": "sensor_verification_passed",
            "attachment_verified": True, "placement_verified": True,
        },
    )

    def dispatch(method, arguments):
        if method == "instruction":
            return "move the object"
        if method == "observe":
            return {"eef_xyz": [0.1, 0.2, 0.3]}
        if method == "call_tool":
            payload = arguments["arguments"]
            return {"target": [
                payload["origin"][index] + payload["delta"][index]
                for index in range(3)
            ]}
        if method == "act":
            return {"reached_target": True}
        raise AssertionError(method)

    report = skills.execute(created["skill_id"], dispatch)
    assert report["execution_completed"] is True
    inspected = skills.inspect(created["skill_id"])
    assert inspected["capability_workspace"].endswith("/tools")


def test_development_state_cannot_count_as_unseen_validation(tmp_path):
    skills, program = _workspaces(tmp_path)
    created = skills.create_candidate(
        name="state_separated_skill", description="State-separated skill",
        semantic_task="language-conditioned transfer", program_id=program["program_id"],
        development_evidence={
            "sensor_only_conclusion": "sensor_verification_passed",
            "attachment_verified": True, "placement_verified": True,
        },
        development_context={
            "environment": "libero_spatial",
            "state_key": "task-5:state-23:seed-7",
        },
    )
    with pytest.raises(TaskSkillValidationError, match="development state"):
        skills.record_unseen_validation(
            created["skill_id"], environment="libero_spatial",
            state_key="task-5:state-23:seed-7",
            sensor_evidence={"sensor_only_conclusion": "sensor_verification_passed"},
        )


def test_task_skill_rejects_unverified_development_program(tmp_path):
    skills, program = _workspaces(tmp_path)
    with pytest.raises(TaskSkillValidationError, match="sensor_verification_passed"):
        skills.create_candidate(
            name="failed_skill", description="failed", semantic_task="transfer",
            program_id=program["program_id"],
            development_evidence={"sensor_only_conclusion": "attachment_not_verified"},
        )
