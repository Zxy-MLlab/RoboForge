import json
import math
import pytest

from embodied_codex.kernel.embodied_state import (
    EmbodiedState,
    EmbodiedTransition,
    Entity,
    Frame,
    Pose,
    action_frame_error,
    build_transition,
    normalize_entity,
    normalize_embodied_state,
    normalize_robot_state,
    pose_delta,
    relative_pose,
    transform_pose,
    transform_point,
)
from embodied_codex.kernel.evidence import build_execution_digest
from embodied_codex.legacy.agent_loop import AgentLoop


def test_transform_and_explicit_frame_identity():
    transform = ((0, -1, 0, 1), (1, 0, 0, 2), (0, 0, 1, 3), (0, 0, 0, 1))
    assert transform_point((1, 0, 2), transform) == [1.0, 3.0, 5.0]
    frame = Frame("camera", "world", transform)
    assert frame.as_dict()["name"] == "camera"
    assert frame.as_dict()["parent"] == "world"
    assert Pose("camera", (1, 2, 3)).as_dict()["frame"] == "camera"


def test_relative_pose_respects_parent_orientation():
    quarter_turn = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    parent = Pose("world", (1, 1, 0), quarter_turn)
    child = Pose("world", (1, 2, 0), quarter_turn)
    relative = relative_pose(parent, child, result_frame="parent_local")
    assert relative.frame == "parent_local"
    assert relative.position == pytest.approx((1.0, 0.0, 0.0))
    assert relative.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_quaternion_rotation_preserves_non_unit_vector_norm():
    quarter_turn = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    parent = Pose("world", (0, 0, 0), quarter_turn)
    child = Pose("world", (0, 0.2, 0), quarter_turn)
    relative = relative_pose(parent, child, result_frame="parent_local")
    assert relative.position == pytest.approx((0.2, 0.0, 0.0))
    assert math.sqrt(sum(value * value for value in relative.position)) == pytest.approx(0.2)


def test_pose_canonical_serialization_round_trip():
    original = Pose("tool", (0.1, -0.2, 0.3), (0.0, 0.0, 0.2, 0.98))
    encoded = original.as_dict()
    assert set(encoded) == {"frame", "position_m", "orientation_xyzw"}
    assert Pose.from_mapping(encoded).as_dict() == encoded
    with pytest.raises(ValueError):
        Pose.from_mapping({"frame": "tool", "position": [0, 0, 0]})


def test_frame_graph_composes_chained_rotations_and_translations():
    frames = {
        "world": Frame("world"),
        "base": Frame("base", "world", ((0, -1, 0, 1), (1, 0, 0, 2),
                                           (0, 0, 1, 0), (0, 0, 0, 1))),
        "tool": Frame("tool", "base", ((1, 0, 0, 0), (0, 1, 0, 0),
                                           (0, 0, 1, 3), (0, 0, 0, 1))),
    }
    from embodied_codex.kernel.embodied_state import frame_transform
    transform = frame_transform(frames, "tool", "world")
    assert transform_point((0, 0, 0), transform) == pytest.approx([1, 2, 3])
    pose = Pose("tool", (0, 0, 0), (0, 0, 0, 1))
    transformed = transform_pose(pose, transform, target_frame="world")
    assert transformed.frame == "world"
    assert transformed.position == pytest.approx((1, 2, 3))


def test_core_rejects_native_robot_and_entity_aliases():
    native = normalize_robot_state({"proprioception": {"native_pose": [0, 0, 0]}})
    assert native.eef_pose is None
    with pytest.raises(ValueError):
        normalize_entity({"point_ref": "native", "world_xyz": [0, 0, 0]})


def test_pose_delta_is_signed_and_action_frame_is_generic():
    delta = pose_delta(Pose("action", (1, 2, 3)), Pose("action", (0.5, 4, 2)))
    assert delta["frame"] == "action"
    assert delta["signed_error"] == {"dx": -0.5, "dy": 2.0, "dz": -1.0}
    assert round(delta["norm_m"], 6) == round(math.sqrt(5.25), 6)
    decomposition = action_frame_error((0, 0, 0), (0, 0, 0.2), (0, 0, 1))
    assert decomposition["along_approach_axis_error_m"] == 0.2
    assert decomposition["lateral_error_m"] == 0.0


def test_entity_provenance_and_opaque_perception_refs():
    entity = normalize_entity({
        "entity_id": "point-1", "label": "object", "confidence": 0.8,
        "geometry": {"frame": "camera", "center": [1, 2, 3]},
        "perception": {"mask_ref": "artifact://sensor/opaque", "bbox": [1, 2, 3, 4]}},
        provenance={"tool_id": "tool:v1"})
    assert entity.entity_id == "point-1"
    assert entity.geometry["frame"] == "camera"
    assert entity.geometry["center"] == [1, 2, 3]
    assert entity.provenance == {"tool_id": "tool:v1"}
    encoded = json.dumps(entity.as_dict())
    assert "artifact://sensor/opaque" in encoded
    assert "/host/" not in encoded


def test_transition_contains_requested_achieved_and_public_delta():
    transition = build_transition(
        before=EmbodiedState(),
        requested_action={"type": "move", "frame": "world", "target_xyz": [1, 2, 3]},
        achieved_action={"type": "move", "target_xyz": [1, 2, 3],
                         "eef_before": [0, 0, 0], "eef_after": [1.1, 1.8, 3.2],
                         "reached": False},
        after=EmbodiedState(), verification={"verified": False})
    value = transition.as_dict()
    assert value["action"]["requested"]["type"] == "move"
    assert value["achieved"]["reached"] is False
    assert value["delta"]["robot_motion"]["signed_error"] == {
        "dx": 0.10000000000000009, "dy": -0.19999999999999996, "dz": 0.20000000000000018}
    assert "why" not in json.dumps(value).lower()


def test_transition_decodes_canonical_before_after_state_and_action_axis():
    before = {"eef_frame": "world", "robot": {
        "eef_pose": {"frame": "world", "position_m": [0, 0, 0]},
        "gripper": {"width_m": 0.04}, "joint_state": {}, "proprioception": {}}}
    after = {"eef_frame": "world", "robot": {
        "eef_pose": {"frame": "world", "position_m": [0, 0, 0.08]},
        "gripper": {"width_m": 0.02}, "joint_state": {}, "proprioception": {}}}
    transition = build_transition(before=before,
        requested_action={"type": "oriented_move", "frame": "world",
                          "target_xyz": [0, 0, 0.1], "action_frame_axis": [0, 0, 1]},
        achieved_action={"type": "oriented_move", "target_xyz": [0, 0, 0.1],
                         "eef_after": [0, 0, 0.08], "action_frame_axis": [0, 0, 1]},
        after=after)
    delta = transition.as_dict()["delta"]
    assert delta["eef_displacement"] == pytest.approx([0, 0, 0.08])
    assert delta["gripper_width"] == {"before": 0.04, "after": 0.02}
    assert delta["action_frame"]["along_action_axis_error_m"] == pytest.approx(-0.02)


def test_digest_exposes_generic_entities_and_transition_facts():
    digest = build_execution_digest({
        "completed": True, "program_sha256": "sha", "rpc_events": [
            {"method": "use", "arguments": {"tool_id": "perception:v1", "payload": {}},
             "entities": [{"entity_id": "point-1", "label": "object", "confidence": 0.9,
                 "geometry": {"frame": "world", "center": [0, 0, 0]},
                 "perception": {"mask_ref": "artifact://sensor/opaque"},
                 "provenance": {"tool_id": "perception:v1"}}],
             "result": {"result": {"native": []}}},
            {"method": "act", "arguments": {"action": {"type": "move", "frame": "world",
                 "target_xyz": [0, 0, 1]}}, "result": {"type": "move", "target_xyz": [0, 0, 1],
                 "eef_before": [0, 0, 0], "eef_after": [0, 0, 0.9], "reached": True}},
        ]}, controller_sha256="sha")
    assert digest["entities"][0]["entity_id"] == "point-1"
    assert digest["entities"][0]["provenance"]["tool_id"] == "perception:v1"
    assert digest["actions"][0]["transition"]["delta"]["robot_motion"]["frame"] == "world"
    assert "rpc_events" not in json.dumps(digest)


def test_public_observation_normalizes_robot_and_entity_state():
    robot = normalize_robot_state({"frame_id": "f1", "robot": {
        "eef_pose": {"frame": "world", "position_m": [1, 2, 3],
                      "orientation_xyzw": [0, 0, 0, 1]},
        "gripper": {"width_m": 0.3, "state": "open"},
        "joint_state": {"position": [0.0]}, "proprioception": {}}})
    assert robot.eef_pose.frame == "world"
    assert robot.gripper_width == pytest.approx(0.3)
    state = normalize_embodied_state(
        {"frame_id": "f1", "robot": {"eef_pose": {"frame": "world",
         "position_m": [1, 2, 3]}, "gripper": {}, "joint_state": {},
         "proprioception": {}}},
        entities=[{"entity_id": "e", "geometry": {"frame": "world", "center": [0, 0, 0]},
                   "label": "object"}])
    assert state.frames["unknown"].name == "unknown"
    assert state.entities[0].geometry["frame"] == "world"


def test_execution_comparison_reports_transition_facts_only():
    loop = AgentLoop.__new__(AgentLoop)
    first = {"controller_sha256": "a", "agent_evidence": {"digest": {
        "tool_calls": [], "actions": [{"type": "move", "requested": {"target_xyz": [0, 0, 1]},
            "result": {"eef_after": [0, 0, 0.9], "final_position_error_m": 0.1},
            "transition": {"delta": {"robot_motion": {"norm_m": 0.1},
                                         "eef_displacement": [0, 0, 0.9]}}}],
        "verifications": []}}}
    second = {"controller_sha256": "b", "agent_evidence": {"digest": {
        "tool_calls": [], "actions": [{"type": "move", "requested": {"target_xyz": [0, 0, 2]},
            "result": {"eef_after": [0, 0, 1.8], "final_position_error_m": 0.2},
            "transition": {"delta": {"robot_motion": {"norm_m": 0.2},
                                         "eef_displacement": [0, 0, 1.8]}}}],
        "verifications": []}}}
    loop._execution_by_ref = lambda ref: first if ref.endswith("a") else second
    comparison = loop._compare_executions("evidence://a", "evidence://b")
    assert comparison["controller_changed"] is True
    assert comparison["behavior"]["robot_motion_changed"] is True
    assert comparison["behavior"]["eef_displacement_changed"] is True
    assert "should" not in json.dumps(comparison).lower()
