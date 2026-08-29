import numpy as np
import pytest

from embodied_codex.adapters.libero_sdk import SDKContractError, validate_action
from embodied_codex.deployments.libero import LiberoDeployment, LiberoDeploymentError


def _deployment():
    deployment = object.__new__(LiberoDeployment)
    deployment.obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.02, 0.02]),
    }
    deployment.step = 0
    deployment.trace = []
    deployment.references = {}
    deployment.episode = type("Episode", (), {"horizon": 10})()
    deployment._sim_step = lambda _action: None
    return deployment


def test_numeric_move_to_point_is_valid():
    action = {"type": "move_to_point", "frame": "world", "position_m": [0.1, 0.2, 0.3]}
    assert validate_action(action) == "move_to_point"
    result = _deployment()._act(action)
    assert result["target_source"] == "controller_numeric"
    assert result["target_frame"] == "world"


def test_numeric_move_to_pose_is_valid():
    action = {"type": "move_to_pose", "frame": "world", "position_m": [0.1, 0.2, 0.3],
              "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    assert validate_action(action) == "move_to_pose"
    result = _deployment()._act(action)
    assert result["target_source"] == "controller_numeric"


def test_reference_control_still_works():
    deployment = _deployment()
    deployment.references["point-1"] = {"world_xyz": [0.1, 0.2, 0.3]}
    result = deployment._act({"type": "move_to_point", "target_ref": "point-1"})
    assert result["target_source"] == "reference"


def test_numeric_pose_requires_explicit_frame():
    with pytest.raises(SDKContractError, match="one of field sets"):
        validate_action({"type": "move_to_pose", "position_m": [0, 0, 0],
                         "quaternion_xyzw": [0, 0, 0, 1]})


def test_numeric_pose_rejects_nonfinite_values():
    with pytest.raises(SDKContractError, match="finite"):
        validate_action({"type": "move_to_point", "frame": "world",
                         "position_m": [0, float("nan"), 0]})


def test_unknown_frame_is_rejected():
    deployment = _deployment()
    with pytest.raises(LiberoDeploymentError, match="unsupported coordinate frame"):
        deployment._act({"type": "move_to_point", "frame": "camera",
                         "position_m": [0.1, 0.2, 0.3]})


def test_action_receipt_records_target_source():
    result = _deployment()._act({"type": "move_to_point", "frame": "world",
                                 "position_m": [0.1, 0.2, 0.3]})
    assert result["target_source"] == "controller_numeric"
