import json

import pytest

from stage_node_workspace import (
    StageNodeValidationError,
    StageNodeWorkspace,
    audit_stage_node,
)


PYTHON = "/data/zxy/envs/vla-report/bin/python"


def test_stage_node_is_immutable_typed_and_executable(tmp_path):
    workspace = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON, timeout_sec=5)
    created = workspace.create(
        name="observe_task", stage_kind="observation",
        description="Read live instruction and observation",
        requires=[],
        provides_by_outcome={"observed": ["instruction", "frame_id", "eef_xyz"]},
        source='''def run_stage(robot, context):
    instruction = robot.instruction()
    frame = robot.observe()
    return {"outcome": "observed", "updates": {
        "instruction": instruction,
        "frame_id": frame["frame_id"],
        "eef_xyz": frame["eef_xyz"],
    }}
''',
    )

    def dispatch(method, arguments):
        if method == "instruction":
            return "pick up the bowl"
        if method == "observe":
            return {"frame_id": "frame-1", "eef_xyz": [0.1, 0.2, 0.3]}
        raise AssertionError(method)

    report = workspace.execute(created["node_id"], {}, dispatch)
    assert report["outcome"] == "observed"
    assert report["updates"]["instruction"] == "pick up the bowl"
    inspected = workspace.inspect(created["node_id"])
    assert inspected["manifest"]["protocol"] == "embodied-stage-node-v1"
    assert inspected["manifest"]["requires"] == []


def test_stage_node_context_and_promised_outputs_are_enforced(tmp_path):
    workspace = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON, timeout_sec=5)
    with pytest.raises(StageNodeValidationError, match="updates_mismatch"):
        workspace.create(
            name="bad_output_stage", stage_kind="planning", description="test",
            requires=["frame_id"], provides_by_outcome={"planned": ["target_xyz"]},
            source='''def run_stage(robot, context):
    robot.record({"frame": context["frame_id"]})
    return {"outcome": "planned", "updates": {}}
''',
        )
    created = workspace.create(
        name="missing_input_stage", stage_kind="planning", description="test",
        requires=["frame_id"], provides_by_outcome={"planned": ["target_xyz"]},
        source='''def run_stage(robot, context):
    return {"outcome": "planned", "updates": {"target_xyz": context["frame_id"]}}
''',
    )
    with pytest.raises(StageNodeValidationError, match="missing context"):
        workspace.execute(created["node_id"], {}, lambda method, args: {})


def test_stage_audit_rejects_privilege_and_fixed_absolute_action():
    privileged = '''def run_stage(robot, context):
    return {"outcome": "x", "updates": {"x": robot._env.check_success()}}
'''
    report = audit_stage_node(privileged)
    assert report["eligible"] is False
    assert "check_success" in report["violations"]
    fixed = '''def run_stage(robot, context):
    robot.act({"target_eef_xyz": [0.1, -0.2, 1.0], "gripper": -1})
    return {"outcome": "done", "updates": {}}
'''
    report = audit_stage_node(fixed)
    assert report["eligible"] is False
    assert "literal_absolute_geometry_target" in report["violations"]


def test_stage_node_hash_tampering_is_rejected(tmp_path):
    workspace = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON)
    created = workspace.create(
        name="immutable_stage", stage_kind="verification", description="test",
        requires=[], provides_by_outcome={"done": []},
        source='''def run_stage(robot, context):
    return {"outcome": "done", "updates": {}}
''',
    )
    destination = workspace.resolve(created["node_id"])
    (destination / "stage.py").write_text("def run_stage(robot, context): return {}\n")
    with pytest.raises(StageNodeValidationError, match="hash mismatch"):
        workspace.inspect(created["node_id"])


def test_checkpoint_requires_adapter_owned_visual_verifier(tmp_path):
    workspace = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON)
    with pytest.raises(StageNodeValidationError, match=r"verify_\*"):
        workspace.create(
            name="fake_verification", stage_kind="visual_verification",
            description="fake", requires=[],
            provides_by_outcome={"verified": []},
            checkpoint_outcomes=["verified"],
            source='''def run_stage(robot, context):
    return {"outcome": "verified", "updates": {}}
''',
        )


def test_stage_audit_rejects_tuple_return_before_robot_execution(tmp_path):
    workspace = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON)
    with pytest.raises(StageNodeValidationError, match="return_must_be_object"):
        workspace.create(
            name="tuple_return_stage", stage_kind="observation", description="bad",
            requires=[], provides_by_outcome={"observed": ["frame_id"]},
            source='''def run_stage(robot, context):
    frame = robot.observe()
    return "observed", {"frame_id": frame["frame_id"]}
''',
        )


def test_verification_must_be_independent_and_action_free(tmp_path):
    workspace = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON)
    with pytest.raises(StageNodeValidationError, match="independent verification"):
        workspace.create(
            name="mixed_motion_verify", stage_kind="motion", description="mixed",
            requires=[], provides_by_outcome={"done": []},
            source='''def run_stage(robot, context):
    result = robot.call_tool("verify_attachment", {"frame_id": "frame"})
    return {"outcome": "done", "updates": {}}
''',
        )
    with pytest.raises(StageNodeValidationError, match="cannot execute robot actions"):
        workspace.create(
            name="active_verification", stage_kind="visual_verification",
            description="mixed", requires=[], provides_by_outcome={"verified": []},
            checkpoint_outcomes=["verified"],
            source='''def run_stage(robot, context):
    robot.act({"target_eef_xyz": context["target"], "gripper": -1})
    robot.call_tool("verify_attachment", {"frame_id": "frame"})
    return {"outcome": "verified", "updates": {}}
''',
        )
