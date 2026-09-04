import numpy as np
import pytest
from types import SimpleNamespace

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


def test_previous_trial_reference_is_invalid_after_reset():
    deployment = _deployment()
    deployment._retired_references = {"point-old"}
    with pytest.raises(LiberoDeploymentError,
                       match="reference belongs to a previous environment generation"):
        deployment._act({"type": "move_to_point", "target_ref": "point-old"})


def test_blocking_joint_control_reads_live_sim_qpos_like_upstream():
    deployment = object.__new__(LiberoDeployment)
    qpos = np.zeros(7, dtype=float)

    class Model:
        @staticmethod
        def get_joint_qpos_addr(name):
            return int(name.removeprefix("robot0_joint")) - 1

    controller = SimpleNamespace(name="JOINT_POSITION")
    deployment.env = SimpleNamespace(
        sim=SimpleNamespace(model=Model(), data=SimpleNamespace(qpos=qpos)),
        robots=[SimpleNamespace(controller=controller)],
        control_freq=20,
    )
    deployment.obs = {"robot0_joint_pos": np.full(7, 99.0)}
    deployment._gripper_fraction = 1.0
    deployment.trace = []

    def step(action):
        # The JOINT_POSITION controller receives a scaled delta; emulate one
        # convergent control step while proving the eighth gripper field exists.
        assert np.asarray(action).shape == (8,)
        qpos[:] += np.asarray(action)[:7] / 20.0

    deployment._sim_step = step
    deployment.move_to_joints_blocking(np.arange(7) / 10.0, max_steps=2)
    np.testing.assert_allclose(qpos, np.arange(7) / 10.0)
    report = deployment.trace[-1]
    assert report["event"] == "joint_control"
    assert report["status"] == "converged"
    assert report["steps_commanded"] == 1
    assert report["final_l2_error_rad"] == pytest.approx(0.0)
    assert len(report["samples"]) == 2


def test_blocking_joint_control_timeout_is_explicit_and_traced():
    deployment = object.__new__(LiberoDeployment)
    qpos = np.zeros(7, dtype=float)

    class Model:
        @staticmethod
        def get_joint_qpos_addr(name):
            return int(name.removeprefix("robot0_joint")) - 1

    deployment.env = SimpleNamespace(
        sim=SimpleNamespace(model=Model(), data=SimpleNamespace(qpos=qpos)),
        robots=[SimpleNamespace(controller=SimpleNamespace(name="JOINT_POSITION"))],
        control_freq=20,
    )
    deployment.obs = {"robot0_joint_pos": qpos}
    deployment._gripper_fraction = 1.0
    deployment.trace = []
    deployment._sim_step = lambda _action: None

    with pytest.raises(LiberoDeploymentError, match="did not converge"):
        deployment.move_to_joints_blocking(np.ones(7) * 0.1, max_steps=2)
    report = deployment.trace[-1]
    assert report["status"] == "timeout"
    assert report["steps_commanded"] == 2
    assert report["final_l2_error_rad"] > 0.0
