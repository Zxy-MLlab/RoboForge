import numpy as np

from rgbd_pick_place import (
    CircularCandidate,
    CartesianWaypointController,
    allowed_observation,
    backproject_rgbd,
    parse_pick_place_instruction,
    make_thea_rgbd_pick_place_tool,
    select_black_bowl,
    select_plate,
)


def test_parse_spatial_relations():
    between = parse_pick_place_instruction(
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )
    assert between.object_name == "black bowl"
    assert between.relation == "between"
    assert between.reference_names == ("plate", "ramekin")
    assert between.target_name == "plate"

    center = parse_pick_place_instruction(
        "pick up the black bowl from table center and place it on the plate"
    )
    assert center.relation == "table_center"
    assert center.reference_names == ()


def test_backproject_identity_camera():
    depth = np.ones((2, 2, 1))
    intrinsic = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    world = backproject_rgbd(depth, intrinsic, np.eye(4))
    np.testing.assert_allclose(world[0, 0], [0, 0, 1])
    np.testing.assert_allclose(world[1, 1], [1, 1, 1])


def test_cartesian_action_is_bounded_and_keeps_orientation():
    controller = CartesianWaypointController(position_gain=20)
    action = controller.action(np.array([0.0, 0.0, 1.0]), np.array([0.10, -0.01, 1.02]), -1)
    np.testing.assert_allclose(action, [1.0, -0.2, 0.4, 0, 0, 0, -1])


def test_sensor_projection_rejects_object_state():
    required = {
        "agentview_image": np.zeros((2, 2, 3)),
        "agentview_depth": np.zeros((2, 2, 1)),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3)),
        "robot0_eye_in_hand_depth": np.zeros((2, 2, 1)),
        "robot0_joint_pos": np.zeros(7),
        "robot0_joint_vel": np.zeros(7),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.zeros(4),
        "robot0_gripper_qpos": np.zeros(2),
        "robot0_gripper_qvel": np.zeros(2),
        "object-state": np.ones(10),
        "plate_1_pos": np.ones(3),
    }
    projected = allowed_observation(required)
    assert "object-state" not in projected
    assert "plate_1_pos" not in projected


def test_classical_candidate_selection():
    bowl = CircularCandidate((10, 10), 15, (90, 91, 90), 0.65, 0.98, (0, 0, 1))
    distractor = CircularCandidate((20, 20), 16, (150, 130, 100), 0.35, 0.7, (0, 0, 1))
    plate = CircularCandidate((30, 30), 24, (170, 150, 140), 0.3, 0.7, (0, 0, 1))
    candidates = [distractor, plate, bowl]
    assert select_black_bowl(candidates) is bowl
    assert select_plate(candidates, exclude=bowl) is plate


def test_thea_tool_spec_is_evaluator_blind():
    spec = make_thea_rgbd_pick_place_tool()
    assert spec.name == "rgbd_code_pick_place"
    assert spec.input_schema["additionalProperties"] is False
    assert "learned model" in spec.description
