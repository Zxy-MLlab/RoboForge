from __future__ import annotations

import hashlib
import json

import pytest

from controller_program_runtime import ControllerProgramRuntime
from controller_program_workspace import (
    ControllerProgramValidationError,
    ControllerProgramWorkspace,
    audit_controller_program,
    protected_prefix_for_stage,
    verified_stage_from_evidence,
)


class _Registry:
    def __init__(self):
        self.tools = {}

    def tool(self, *, name, **metadata):
        def decorate(function):
            self.tools[name] = (function, metadata)
            return function
        return decorate


PYTHON = "/data/zxy/envs/vla-report/bin/python"


PROGRAM = '''def run(robot):
    instruction = robot.instruction()
    history = []
    for attempt in range(2):
        observation = robot.observe()
        detections = robot.call_tool("detect", {"query": "bowl", "frame": observation["frame"]})
        action = [detections["x"] - attempt, 0.0, 0.0]
        result = robot.act(action)
        history.append({"attempt": attempt + 1, "accepted": result["accepted"]})
        robot.record(history[-1])
        if result["accepted"]:
            break
    return {"instruction": instruction, "history": history}
'''


def test_complete_program_owns_loop_and_calls_robot_sdk(tmp_path):
    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON, timeout_sec=5)
    created = workspace.create("closed_loop_pick", PROGRAM, "two sensor-driven attempts")
    calls = []

    def dispatch(method, arguments):
        calls.append((method, arguments))
        if method == "instruction":
            return "pick up the bowl"
        if method == "observe":
            return {"frame": len([item for item in calls if item[0] == "observe"]), "reward": 99}
        if method == "call_tool":
            return {"x": 0.4}
        if method == "act":
            return {"accepted": len([item for item in calls if item[0] == "act"]) >= 2, "done": True}
        return {"recorded": True}

    report = workspace.execute(created["program_id"], dispatch)
    assert report["execution_completed"] is True
    assert [item[0] for item in calls].count("act") == 2
    assert report["result"]["history"][-1]["accepted"] is True
    serialized = json.dumps(report).lower()
    assert "reward" not in serialized
    assert '"done"' not in serialized
    inspected = workspace.inspect(created["program_id"])
    assert inspected["runs"][0]["execution_completed"] is True


def test_program_versions_are_immutable_and_inspectable(tmp_path):
    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON)
    first = workspace.create("pick_program", PROGRAM)
    second = workspace.create("pick_program", PROGRAM.replace("range(2)", "range(3)"))
    assert first["program_id"] == "pick_program:v001"
    assert second["program_id"] == "pick_program:v002"
    inspected = workspace.inspect(first["program_id"])
    assert inspected["source"] == PROGRAM
    assert inspected["manifest"]["sha256"] == hashlib.sha256(PROGRAM.encode()).hexdigest()


def test_program_audit_rejects_privileged_io_and_private_escape(tmp_path):
    unsafe = "import os\ndef run(robot):\n return robot._env.check_success()\n"
    report = audit_controller_program(unsafe)
    assert report["eligible"] is False
    assert "check_success" in report["violations"]
    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON)
    with pytest.raises(ControllerProgramValidationError):
        workspace.create("unsafe_program", unsafe)


def test_program_audit_rejects_literal_task_text_without_live_instruction():
    source = '''def run(robot):
    instruction = "pick up the bowl"
    return {"instruction": instruction}
'''
    report = audit_controller_program(source)
    assert report["eligible"] is False
    assert "missing_robot_instruction_call" in report["violations"]


def test_program_audit_rejects_literal_absolute_action_target():
    source = '''def run(robot):
    robot.instruction()
    robot.act({"target_eef_xyz": [0.12, -0.08, 1.04], "gripper": -1})
'''
    report = audit_controller_program(source)
    assert report["eligible"] is False
    assert "literal_absolute_action_target" in report["violations"]
    live = source.replace(
        'robot.act({"target_eef_xyz": [0.12, -0.08, 1.04], "gripper": -1})',
        'frame = robot.observe()\n    robot.act({"target_eef_xyz": frame["eef_xyz"], "gripper": -1})',
    )
    assert audit_controller_program(live)["eligible"] is True


def test_verified_stage_revision_preserves_exact_sensor_checkpoint_prefix(tmp_path):
    base_source = '''def run(robot):
    robot.instruction()
    frame = robot.observe()
    result = robot.call_tool("verify_landmark_displacement", {"frame_id": frame["frame_id"]})
    if result.get("verified") is not True: return {"phase": "articulation"}
    robot.record({"phase": "suffix"})
'''
    initial = ControllerProgramWorkspace(tmp_path, python=PYTHON)
    base = initial.create("drawer_skill", base_source)
    checkpoint = protected_prefix_for_stage(base_source, "articulation")
    assert checkpoint["through_line"] == 5

    constrained = ControllerProgramWorkspace(
        tmp_path, python=PYTHON,
        required_revision={
            "base_program_id": base["program_id"], "stage": "articulation",
        },
    )
    revised = base_source.replace(
        '    frame = robot.observe()\n',
        '    # formatting and comments are not executable structure\n    frame = robot.observe()\n',
    ) + '    robot.record({"phase": "new_suffix"})\n'
    created = constrained.create("drawer_skill_revision", revised)
    assert created["success"] is True
    manifest = constrained.inspect(created["program_id"])["manifest"]
    assert manifest["revision_constraint"]["stage"] == "articulation"

    changed_prefix = revised.replace(
        'frame = robot.observe()', 'frame = robot.observe()\n    robot.record({"changed": True})', 1,
    )
    with pytest.raises(ControllerProgramValidationError, match="verified_stage_prefix_changed"):
        constrained.create("invalid_revision", changed_prefix)


def test_verified_stage_is_derived_only_from_sensor_evidence():
    assert verified_stage_from_evidence({
        "verifications": [{"kind": "articulation", "verified": True}],
    }) == "articulation"
    assert verified_stage_from_evidence({"attachment_verified": True}) == "attachment"
    assert verified_stage_from_evidence({
        "attachment_verified": True, "placement_verified": True,
    }) == "placement"
    assert verified_stage_from_evidence({"evaluator_success": True}) is None


def test_program_timeout_and_hash_tampering_are_contained(tmp_path):
    module = tmp_path / "program.py"
    module.write_text("def run(robot):\n    while True: pass\n")
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    runtime = ControllerProgramRuntime(python=PYTHON, timeout_sec=.5)
    report = runtime.run(module, expected_sha256=digest, dispatch=lambda method, args: None)
    assert report["execution_completed"] is False
    assert "timed out" in report["error"]
    module.write_text("def run(robot):\n    return {}\n")
    with pytest.raises(Exception, match="hash changed"):
        runtime.run(module, expected_sha256=digest, dispatch=lambda method, args: None)


def test_runtime_executes_typed_stage_entrypoint(tmp_path):
    module = tmp_path / "stage.py"
    module.write_text(
        "def run_stage(robot, context):\n"
        "    text = robot.instruction()\n"
        "    return {'outcome': 'next', 'updates': {'task': text, 'x': context['x']}}\n"
    )
    digest = hashlib.sha256(module.read_bytes()).hexdigest()
    runtime = ControllerProgramRuntime(python=PYTHON, timeout_sec=5)
    report = runtime.run(
        module, expected_sha256=digest,
        dispatch=lambda method, arguments: "move the bowl",
        entrypoint="run_stage", arguments={"x": 3},
    )
    assert report["execution_completed"] is True
    assert report["result"] == {
        "outcome": "next", "updates": {"task": "move the bowl", "x": 3},
    }


def test_registered_executor_owns_environment_lifecycle(tmp_path):
    from controller_program_workspace import register_controller_program_tools

    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON)
    registry = _Registry()
    calls = []
    register_controller_program_tools(
        registry,
        workspace,
        executor=lambda program_id: calls.append(program_id) or {
            "program_id": program_id,
            "sensor_evidence": {"sensor_only_conclusion": "development_run_completed"},
        },
    )
    created = registry.tools["create_controller_program"][0]("pick_program", PROGRAM)
    report = registry.tools["execute_controller_program"][0](created["program_id"])
    assert calls == ["pick_program:v001"]
    assert report["sensor_evidence"]["sensor_only_conclusion"] == "development_run_completed"


def test_registered_creation_returns_actionable_audit_failure(tmp_path):
    from controller_program_workspace import register_controller_program_tools

    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON)
    registry = _Registry()
    register_controller_program_tools(registry, workspace)
    result = registry.tools["create_controller_program"][0](
        "unsafe_program",
        "def run(robot):\n    return robot._env.check_success()\n",
    )
    assert result["success"] is False
    assert result["controller_created"] is False
    assert "check_success" in result["reason"]


def test_registered_inspection_bounds_run_history_for_model_context(tmp_path):
    from controller_program_workspace import register_controller_program_tools

    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON)
    created = workspace.create("pick_program", PROGRAM)
    workspace.execute(created["program_id"], lambda method, arguments: {"accepted": True})
    registry = _Registry()
    register_controller_program_tools(registry, workspace)
    report = registry.tools["inspect_controller_program"][0](created["program_id"])
    assert report["source"] == PROGRAM
    assert report["runs"][0]["rpc_calls"] > 0
    assert "rpc_events" not in report["runs"][0]


def test_program_may_complete_with_implicit_none(tmp_path):
    workspace = ControllerProgramWorkspace(tmp_path, python=PYTHON, timeout_sec=5)
    created = workspace.create(
        "implicit_none_program",
        "def run(robot):\n    robot.instruction()\n    robot.record({'phase': 'complete'})\n",
    )
    report = workspace.execute(created["program_id"], lambda method, arguments: {"ok": True})
    assert report["execution_completed"] is True
    assert report["result"] is None


def test_program_audit_requires_graspnet_approach_vector_consumption():
    source = '''def run(robot):
    robot.instruction()
    frame = robot.observe()
    grasps = robot.call_tool("generate_grasps", {"frame_id": frame["frame_id"]})
    candidate = grasps["grasps"][0]
    robot.record({"translation": candidate["translation_world"]})
'''
    report = audit_controller_program(source)
    assert report["eligible"] is False
    assert "missing_grasp_approach_world_consumption" in report["violations"]
    fixed = source.replace(
        'robot.record({"translation": candidate["translation_world"]})',
        'robot.record({"translation": candidate["translation_world"], "approach": candidate["approach_world"]})',
    )
    assert audit_controller_program(fixed)["eligible"] is True


def test_program_audit_requires_all_tested_tool_inputs_and_outputs(tmp_path):
    capability_root = tmp_path / "capabilities"
    manifest_dir = capability_root / "execution_profile" / "v001"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({
        "tool_id": "execution_profile:v001",
        "status": "unit_tested",
        "compatible_hooks": ["grasp_execution_profile"],
    }))
    workspace = ControllerProgramWorkspace(
        tmp_path / "programs", python=PYTHON,
        capability_workspace=capability_root,
    )
    required_inputs = {
        "attempt": 1, "candidate": {}, "source_xyz": [0, 0, 0],
        "tracked_source_xyz": [0, 0, 0], "current_eef_xyz": [0, 0, 1],
        "previous_failure": None, "default_profile": {},
    }
    source = '''def run(robot):
    robot.instruction()
    profile = robot.call_tool("execution_profile:v001", PAYLOAD)
    robot.record({"gain": profile["position_gain"]})
'''.replace("PAYLOAD", repr(required_inputs))
    with pytest.raises(ControllerProgramValidationError) as exc:
        workspace.create("incomplete_profile", source)
    assert "capability_output_not_consumed:execution_profile:v001:close_steps" in str(exc.value)

    output_fields = list(workspace.capability_contracts()["execution_profile:v001"]["output_fields"])
    consumption = "\n".join(
        f'    robot.record({{"{field}": profile["{field}"]}})'
        for field in output_fields
    )
    complete = source.replace(
        '    robot.record({"gain": profile["position_gain"]})', consumption,
    )
    created = workspace.create("complete_profile", complete)
    assert created["audit"]["eligible"] is True


def test_program_audit_rejects_missing_tested_tool_input(tmp_path):
    capability_root = tmp_path / "capabilities"
    manifest_dir = capability_root / "retry_ranker" / "v001"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(json.dumps({
        "tool_id": "retry_ranker:v001", "status": "unit_tested",
        "compatible_hooks": ["grasp_retry_ranking"],
    }))
    workspace = ControllerProgramWorkspace(
        tmp_path / "programs", python=PYTHON,
        capability_workspace=capability_root,
    )
    source = '''def run(robot):
    robot.instruction()
    ranked = robot.call_tool("retry_ranker:v001", {"candidate_count": 2})
    robot.record({"order": ranked["candidate_indices"]})
'''
    with pytest.raises(ControllerProgramValidationError) as exc:
        workspace.create("missing_ranker_input", source)
    assert "capability_input_not_supplied:retry_ranker:v001:candidates" in str(exc.value)
