from pathlib import Path

import pytest

from capability_workspace import CapabilityWorkspace
from controller_graph_workspace import ControllerGraphWorkspace
from stage_node_workspace import StageNodeWorkspace
import libero_robot_sdk as sdk_module
from libero_robot_sdk import (
    LiberoRobotSDKAdapter,
    LiberoRobotSDKError,
    diagnose_sensor_failure,
    libero_task_instruction,
    locked_attachment_verification,
    mask_support_decision,
    robot_sdk_contract,
    summarize_motion_outcome,
    summarize_phase_motion,
    execute_libero_graph,
)
from runtime_capabilities import HOOK_CONTRACTS


def test_public_task_instruction_exposes_language_without_task_definition():
    instruction = libero_task_instruction("libero_spatial", 4).lower()
    assert "drawer" in instruction
    assert "plate" in instruction
    assert "bddl" not in instruction


def test_locked_baseline_rejects_controller_supplied_false_motion_scenario():
    baseline = [-0.227, 0.221, 0.969]
    static_object = [-0.235, 0.217, 0.980]
    eef = [-0.261, 0.187, 1.071]
    report = locked_attachment_verification(
        static_object, eef, baseline, 0.003,
        [static_object],
    )
    assert report["verified"] is False
    assert report["source_vacated"] is False


def test_locked_baseline_accepts_object_that_moves_with_eef_and_vacates_source():
    baseline = [-0.227, 0.221, 0.969]
    carried = [-0.230, 0.220, 1.090]
    eef = [-0.240, 0.210, 1.145]
    report = locked_attachment_verification(
        carried, eef, baseline, 0.02,
        [carried],
    )
    assert report["verified"] is True
    assert report["source_vacated"] is True


def test_motion_summary_exposes_unreached_target_without_privileged_state():
    report = summarize_motion_outcome(
        [0.2, 0.2, 1.0], [0.0, 0.0, 1.2], [0.05, 0.05, 1.15],
        requested_repeat=15,
    )
    assert report["initial_error_m"] > report["final_error_m"] > 0.025
    assert report["progress_m"] > 0
    assert report["reached_target"] is False
    assert report["stalled"] is False


def test_motion_summary_only_calls_single_step_negligible_progress_stalled():
    report = summarize_motion_outcome(
        [0.2, 0.2, 1.0], [0.0, 0.0, 1.2], [0.00001, 0.00001, 1.19999],
        requested_repeat=1,
    )
    assert report["reached_target"] is False
    assert report["stalled"] is True


def test_near_goal_slow_progress_is_not_mislabeled_stalled():
    report = summarize_motion_outcome(
        [0.0, 0.0, 0.0], [0.012, 0.0, 0.0], [0.011, 0.0, 0.0],
        requested_repeat=20, tolerance_m=0.008,
    )
    assert report["reached_target"] is False
    assert report["progress_m"] > 0.0005
    assert report["stalled"] is False


def _stalled_outcome(phase, command_index):
    return {
        "phase": phase, "command_index": command_index,
        "reached_target": False, "stalled": True,
        "final_error_m": 0.035,
    }


def test_phase_motion_identifies_repeated_contact_unreachability():
    report = summarize_phase_motion([
        _stalled_outcome("contact", index) for index in range(1, 5)
    ])
    assert report["dominant_unreachable_phase"] == "contact"
    assert report["phases"]["contact"]["commands"] == 4
    assert report["phases"]["contact"]["unreachable"] is True


def test_drawer_task_without_articulation_is_not_called_generic_grasp_failure():
    failure, _ = diagnose_sensor_failure(
        instruction="pick up the bowl in the top drawer and place it on the plate",
        execution_completed=True, attachment_verified=False,
        placement_verified=False, verifications=[], action_outcomes=[],
    )
    assert failure == "articulation_not_attempted"


def test_verified_drawer_open_exposes_later_contact_unreachability():
    actions = [
        _stalled_outcome("contact", index) for index in range(1, 5)
    ]
    actions.insert(0, {
        "phase": "drawer_pull", "command_index": 0,
        "reached_target": True, "stalled": False, "final_error_m": 0.005,
    })
    failure, _ = diagnose_sensor_failure(
        instruction="pick up the bowl in the drawer and place it on the plate",
        execution_completed=True, attachment_verified=False,
        placement_verified=False,
        verifications=[{"kind": "articulation", "verified": True}],
        action_outcomes=actions,
    )
    assert failure == "contact_unreachable"


def test_verified_drawer_open_exposes_low_contact_convergence_without_attachment_check():
    actions = [{
        "phase": "contact", "command_index": index,
        "reached_target": index == 1, "stalled": False,
        "final_error_m": 0.03,
    } for index in range(1, 11)]
    failure, diagnostics = diagnose_sensor_failure(
        instruction="pick up the bowl in the drawer and place it on the plate",
        execution_completed=True, attachment_verified=False,
        placement_verified=False,
        verifications=[{"kind": "articulation", "verified": True}],
        action_outcomes=actions,
    )
    assert failure == "contact_convergence_failed"
    assert diagnostics["phases"]["contact"]["reach_rate"] == 0.1


def test_sdk_enforces_and_audits_tested_capability_hook_contract(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    created = workspace.create(
        "typed_ranker",
        "def run(payload):\n    return {'candidate_indices': [0, 1]}\n",
        "Return two bounded candidates",
    )
    assert workspace.test_hook(created["tool_id"], "grasp_retry_ranking")["success"]
    adapter = object.__new__(LiberoRobotSDKAdapter)
    adapter.capability_store = workspace
    adapter.capability_hook_invocations = []
    payload = HOOK_CONTRACTS["grasp_retry_ranking"]["test_inputs"][0]
    result = adapter._call_tool(created["tool_id"], payload)
    assert result == {"candidate_indices": [0, 1]}
    assert adapter.capability_hook_invocations[-1]["applied"] is True
    with pytest.raises(LiberoRobotSDKError, match="contract violation"):
        adapter._call_tool(created["tool_id"], {"candidates": []})
    assert adapter.capability_hook_invocations[-1]["applied"] is False


def test_sdk_executes_tested_generic_schema_capability_without_fixed_hook(tmp_path):
    workspace = CapabilityWorkspace(
        tmp_path / "tools", python="/data/zxy/envs/vla-report/bin/python"
    )
    created = workspace.create(
        "pull_stall_recovery",
        "def run(payload):\n"
        "    return {'retry': payload['progress_m'] < .002, 'gain': .3}\n",
        "Generic articulation recovery",
        input_schema={
            "type": "object", "properties": {"progress_m": {"type": "number"}},
            "required": ["progress_m"], "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"retry": {"type": "boolean"}, "gain": {"type": "number"}},
            "required": ["retry", "gain"], "additionalProperties": False,
        },
        stage="articulation",
    )
    assert workspace.test(created["tool_id"], [{
        "input": {"progress_m": 0.0003},
        "expected": {"retry": True, "gain": 0.3},
    }])["success"]
    adapter = object.__new__(LiberoRobotSDKAdapter)
    adapter.capability_store = workspace
    adapter.capability_hook_invocations = []
    result = adapter._call_tool(created["tool_id"], {"progress_m": 0.0003})
    assert result == {"retry": True, "gain": 0.3}
    assert adapter.capability_hook_invocations[-1] == {
        "hook": "generic_capability", "tool_id": created["tool_id"],
        "applied": True, "stage": "articulation",
    }
    with pytest.raises(LiberoRobotSDKError, match="contract violation"):
        adapter._call_tool(created["tool_id"], {})


def test_mask_support_decision_accepts_contained_stable_footprint():
    assert mask_support_decision(
        {"containment": 0.82, "clearance_ratio": 0.71}, 0.025,
    ) is True
    assert mask_support_decision(
        {"containment": 0.40, "clearance_ratio": 0.71}, 0.025,
    ) is False


def test_sdk_contract_exposes_opaque_support_mask_flow():
    contract = robot_sdk_contract()["call_tool"]
    assert "mask_id" in contract["segment_box"]["returns"]
    assert "target_mask_id" in contract["verify_support_relation"]["arguments"]
    assert "phase" in robot_sdk_contract()["act"]["arguments"]


def test_sdk_contract_exposes_adapter_locked_landmark_displacement_flow():
    contract = robot_sdk_contract()["call_tool"]
    assert "baseline_id" in contract["capture_landmark_baseline"]["returns"]
    verify = contract["verify_landmark_displacement"]
    assert "baseline_id" in verify["arguments"]
    assert verify["returns"]["kind"] == "articulation"


def test_grasp_contract_exposes_sensor_derived_approach_vector_for_rankers():
    grasp = robot_sdk_contract()["call_tool"]["generate_grasps"]["returns"]["grasps"][0]
    assert grasp["approach_world"] == [0.0, 0.0, -1.0]


def test_sensor_conclusion_does_not_let_old_attachment_label_override_verified_grasp():
    adapter = object.__new__(LiberoRobotSDKAdapter)
    adapter.verifications = [{
        "kind": "attachment", "verified": True,
        "object_xyz": [0.1, 0.1, 1.0], "eef_xyz": [0.1, 0.1, 1.1],
    }]
    adapter.trace = [{
        "event": "controller_record", "payload": {"phase": "attachment_check"},
    }]
    adapter.action_outcomes = []
    adapter.capability_hook_invocations = []
    adapter.instruction_text = "pick up the bowl and place it on the plate"
    adapter.step = 20
    adapter.output = Path("/tmp/sensor-conclusion-test")
    evidence = adapter.sensor_evidence({"execution_completed": True, "rpc_events": []})
    assert evidence["attachment_verified"] is True
    assert evidence["sensor_only_conclusion"] == "transport_not_verified"


def test_execute_libero_graph_keeps_one_adapter_alive_across_stage_processes(
    tmp_path, monkeypatch,
):
    """The production graph wrapper must not reset the episode per Stage Node."""
    nodes = StageNodeWorkspace(
        tmp_path / "nodes", python="/data/zxy/envs/vla-report/bin/python",
        timeout_sec=5,
    )
    observe = nodes.create(
        name="persistent_observe", stage_kind="observation", description="observe",
        requires=["task_instruction"],
        provides_by_outcome={"observed": ["eef_xyz", "frame_id"]},
        source='''def run_stage(robot, context):
    instruction = context["task_instruction"]
    frame = robot.observe()
    return {"outcome": "observed", "updates": {
        "eef_xyz": frame["eef_xyz"], "frame_id": frame["frame_id"]}}
''',
    )["node_id"]
    move = nodes.create(
        name="persistent_move", stage_kind="motion", description="move",
        requires=["eef_xyz"], provides_by_outcome={"moved": []},
        source='''def run_stage(robot, context):
    start = context["eef_xyz"]
    target = [start[0], start[1], start[2] + 0.01]
    result = robot.act({"target_eef_xyz": target, "gripper": -1})
    return {"outcome": "moved", "updates": {}}
''',
    )["node_id"]
    verify = nodes.create(
        name="persistent_verification", stage_kind="visual_verification",
        description="verify same episode state", requires=["frame_id"],
        provides_by_outcome={"verified": [], "rejected": []},
        checkpoint_outcomes=["verified"],
    source='''def run_stage(robot, context):
    frame = robot.observe()
    result = robot.call_tool("verify_attachment", {"frame_id": frame["frame_id"]})
    if result.get("verified") is True:
        return {"outcome": "verified", "updates": {}}
    return {"outcome": "rejected", "updates": {}}
''',
    )["node_id"]
    graphs = ControllerGraphWorkspace(tmp_path / "graphs", nodes=nodes)
    graph = graphs.create(
        name="persistent_adapter_graph", description="one episode",
        entry_node="observe",
        bindings={"observe": observe, "move": move, "verify": verify},
        edges=[
            {"from": "observe", "outcome": "observed", "to": "move"},
            {"from": "move", "outcome": "moved", "to": "verify"},
            {"from": "verify", "outcome": "verified", "to": "$success"},
            {"from": "verify", "outcome": "rejected", "to": "$failure"},
        ],
        initial_context_fields=["task_instruction"],
    )["graph_id"]

    class FakePersistentAdapter:
        instances = []

        def __init__(self, **kwargs):
            self.position_z = 0.5
            self.instruction_text = "move and verify"
            self.closed = False
            self.calls = []
            type(self).instances.append(self)

        def dispatch(self, method, arguments):
            self.calls.append(method)
            if method == "observe":
                return {
                    "frame_id": f"z-{self.position_z:.2f}",
                    "eef_xyz": [0.0, 0.0, self.position_z],
                }
            if method == "act":
                self.position_z = float(arguments["action"]["target_eef_xyz"][2])
                return {"reached_target": True}
            if method == "call_tool":
                return {"verified": self.position_z > 0.5}
            raise AssertionError(method)

        def sensor_evidence(self, runtime_report, *, controller_interface="program"):
            return {
                "execution_completed": runtime_report["execution_completed"],
                "sensor_only_conclusion": "sensor_verification_passed",
                "controller_interface": controller_interface,
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr(sdk_module, "LiberoRobotSDKAdapter", FakePersistentAdapter)
    report = execute_libero_graph(
        graphs, graph, suite="libero_spatial", task=4, state=23, seed=7,
        output=tmp_path / "run",
    )

    assert len(FakePersistentAdapter.instances) == 1
    adapter = FakePersistentAdapter.instances[0]
    assert adapter.calls == ["observe", "act", "observe", "call_tool"]
    assert adapter.position_z == pytest.approx(0.51)
    assert adapter.closed is True
    assert report["sensor_evidence"]["controller_interface"] == "graph"
    assert report["sensor_evidence"]["controller_graph"]["verified_prefix_aliases"] == [
        "observe", "move", "verify",
    ]
