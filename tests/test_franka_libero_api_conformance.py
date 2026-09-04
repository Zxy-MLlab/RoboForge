import base64
import hashlib
import inspect
import io
from pathlib import Path
import sys
import types

import numpy as np

from embodied_codex.adapters.franka_libero_api import (
    FrankaLiberoApi, _obb, decompose_transform, depth_to_point_cloud,
    depth_to_pointcloud, mask_to_world_points,
)
from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
from embodied_codex.adapters.libero import (
    _controller_api_index,
    _runtime_controller_mode_index,
)
from embodied_codex.kernel.runtime import ControllerRuntime, _trace_value
from embodied_codex.kernel.sandbox import default_sandbox


class Env:
    instruction = "pick up the object"
    home_joint_position = np.zeros(7)

    def __init__(self):
        self.calls = []

    def move_to_joints_blocking(self, joints, **kwargs):
        self.calls.append(("move_to_joints_blocking", np.asarray(joints), kwargs))

    def _set_gripper(self, value): self.calls.append(("gripper", value))
    def _step_once(self): self.calls.append(("step",))


def test_public_api_contains_upstream_methods_and_signatures():
    names = set(LIBERO_ROBOT_SDK_CONTRACT["controller_api"]["methods"])
    api = FrankaLiberoApi(Env())
    assert names.issubset(api.functions())
    assert str(inspect.signature(api.goto_pose)) == "(position: 'np.ndarray', quaternion_wxyz: 'np.ndarray', z_approach: 'float' = 0.0) -> 'None'"
    assert str(inspect.signature(api.get_object_pose)) == "(object_name: 'str', use_multiview: 'bool' = True)"


def test_controller_manual_is_derived_from_real_callable_signatures():
    manual = _controller_api_index()
    assert manual["invocation"]["preferred"] == "robot.<method>(*args, **kwargs)"
    assert manual["method_catalog"]["sample_grasp_pose"]["call"].startswith(
        "robot.sample_grasp_pose(object_name: 'str'"
    )
    assert "z_approach" in manual["method_catalog"]["goto_pose"]["signature"]
    assert "robot.sample_grasp_pose(object_name)" in manual["upstream_usage_examples"][0]["code"]
    assert set(manual["method_catalog"]) == set(manual["methods"])


def test_runtime_manual_selects_only_mode_compatible_motion_api():
    joint = _runtime_controller_mode_index("JOINT_POSITION")
    assert "robot.goto_pose" in joint["motion_api"]["use"]
    assert "robot.act(move_to_pose)" in joint["motion_api"]["do_not_use"]
    osc = _runtime_controller_mode_index("OSC_POSE")
    assert "robot.act(move_to_pose)" in osc["motion_api"]["use"]
    assert "robot.goto_pose" in osc["motion_api"]["do_not_use"]


def test_rpc_trace_fingerprints_bulk_arrays_without_losing_semantics():
    array = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    encoded = base64.b64encode(array.tobytes()).decode("ascii")
    traced = _trace_value({
        "method": "get_observation",
        "result": {
            "rgb": {
                "__roboforge_ndarray__": True,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data_base64": encoded,
            },
            "robot_joint_pos": np.arange(7, dtype=np.float64),
        },
    })
    assert traced["method"] == "get_observation"
    assert traced["result"]["rgb"] == {
        "__roboforge_ndarray__": True,
        "dtype": "uint8",
        "shape": [256, 256, 3],
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "byte_length": array.nbytes,
    }
    assert traced["result"]["robot_joint_pos"]["shape"] == [7]
    assert "data_base64" not in str(traced)


def test_upstream_frame_math_and_tcp_offset_semantics():
    K = np.array([[2., 0, 0], [0, 2., 0], [0, 0, 1.]])
    T = np.eye(4); T[:3, 3] = [1, 2, 3]
    points = mask_to_world_points(np.array([[1]], dtype=np.uint8), np.array([[2.]]), K, T)
    np.testing.assert_allclose(points, [[1, 2, 5]])
    pos, q = decompose_transform(np.eye(4))
    np.testing.assert_allclose(pos, [0, 0, 0]); np.testing.assert_allclose(q, [1, 0, 0, 0])
    api = FrankaLiberoApi(Env())
    api._solve_ik_with_prev = lambda position, quaternion_wxyz, prev_cfg=None: np.zeros(7)
    api.goto_pose(np.array([1., 2., 3.]), np.array([1., 0, 0, 0]))
    assert api._TCP_OFFSET.tolist() == [0., 0., -0.1]
    assert api._env.calls[0][0] == "move_to_joints_blocking"


def test_depth_helpers_match_both_upstream_return_shapes():
    depth = np.array([[0.01, 1.0], [2.0, np.nan]])
    K = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]])
    organized = depth_to_point_cloud(depth, K)
    assert organized.shape == (2, 2, 3)
    unfiltered = depth_to_pointcloud(depth, K, filter_invalid=False)
    assert unfiltered.shape == (2, 2, 3)
    filtered = depth_to_pointcloud(depth, K)
    assert filtered.shape == (2, 3)
    np.testing.assert_allclose(filtered[:, 2], [1.0, 2.0])


def test_obb_copies_open3d_readonly_rotation_before_scipy(monkeypatch):
    rotation = np.asfortranarray(np.eye(3))
    rotation.setflags(write=False)

    class Box:
        center = np.array([0.1, 0.2, 0.3])
        extent = np.array([0.4, 0.5, 0.6])
        R = rotation

    class Cloud:
        def __init__(self, _points): pass
        def remove_statistical_outlier(self, **_kwargs): return self, []
        def get_oriented_bounding_box(self): return Box()

    fake_open3d = types.SimpleNamespace(
        geometry=types.SimpleNamespace(PointCloud=Cloud),
        utility=types.SimpleNamespace(Vector3dVector=lambda points: points),
    )
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)

    result = _obb(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
    np.testing.assert_allclose(result["R"], np.eye(3))
    np.testing.assert_allclose(result["quaternion_wxyz"], [1., 0., 0., 0.])
    assert result["R"].flags.writeable and result["R"].flags.c_contiguous


def test_interpolate_segment_matches_upstream_zero_distance_semantics():
    api = FrankaLiberoApi(Env())
    point = np.array([1.0, 2.0, 3.0])
    result = api.interpolate_segment(point, point)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0], point)


def test_gripper_is_blocking_and_dual_arm_fails_explicitly_when_unavailable():
    env = Env(); api = FrankaLiberoApi(env)
    api.open_gripper(); api.close_gripper()
    assert [x[0] for x in env.calls].count("step") == 100
    assert api.supports_dual_arm() is False
    try:
        api.goto_pose_arm1(np.zeros(3), np.array([1., 0, 0, 0]))
    except RuntimeError as exc:
        assert "dual-arm" in str(exc)
    else:
        raise AssertionError("arm1 control must not be fabricated")


def test_sdk_is_not_registered_as_openhands_tool():
    from embodied_codex.kernel.runtime import _ARGUMENT_KEYS
    assert "sdk" not in {"observe", "act", "use", "verify", "record"}
    assert _ARGUMENT_KEYS["sdk"] == {"method", "args", "kwargs"}


def test_controller_runtime_roundtrips_ndarrays_through_sdk(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text(
        "import numpy as np\n"
        "def run(robot):\n"
        "    p = robot.mask_to_world_points(np.array([[1]], dtype=np.uint8), "
        "np.array([[2.0]]), np.eye(3), np.eye(4))\n"
        "    return {'shape': list(p.shape), 'point': p[0].tolist()}\n"
    )

    class Deployment:
        instruction = "test"
        def dispatch(self, method, arguments):
            assert method == "sdk"
            api = FrankaLiberoApi(Env())
            return {"method": arguments["method"],
                    "result": getattr(api, arguments["method"])(*arguments["args"], **arguments["kwargs"])}
        def project_rpc_output(self, method, arguments, result):
            from embodied_codex.deployments.libero import LiberoDeployment
            result = dict(result); result["result"] = LiberoDeployment._encode_sdk_result(result["result"])
            return result

    result = ControllerRuntime(python=Path(__import__("sys").executable),
                               sandbox=default_sandbox()).execute(controller, Deployment())
    assert result["completed"] is True
    assert result["result"] == {"shape": [1, 3], "point": [0.0, 0.0, 2.0]}


def test_upstream_http_service_semantics_are_preserved(monkeypatch):
    api = FrankaLiberoApi(Env())
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    raw_mask = np.array([[[1, 0, 1], [0, 1, 0]]], dtype=np.uint8)
    buf = io.BytesIO(); buf.write(raw_mask.tobytes())
    encoded = base64.b64encode(buf.getvalue()).decode()

    def fake_post(url, payload, timeout=120):
        if url.endswith("segment_point"):
            return {"masks_shape": list(raw_mask.shape), "masks_dtype": "uint8",
                    "masks_base64": encoded, "scores": [0.9]}
        if url.endswith("/ik"):
            # solve_ik follows ASPIRE's reduced API and applies the Franka TCP
            # offset before forwarding the panda_hand target.
            np.testing.assert_allclose(payload["target_pose_wxyz_xyz"],
                                       [1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2])
            return {"joint_positions": [0.0] * 7}
        raise AssertionError(url)

    monkeypatch.setattr("embodied_codex.adapters.franka_libero_api._post", fake_post)
    masks = api.segment_sam3_point_prompt(rgb, (1.0, 0.0))
    assert masks[0]["mask"].dtype == bool and masks[0]["mask"].shape == (2, 3)
    np.testing.assert_allclose(api.solve_ik(np.array([.1, .2, .3]), np.array([1., 0, 0, 0])), np.zeros(7))


def test_goto_pose_discards_pyroki_gripper_coordinate(monkeypatch):
    env = Env()
    api = FrankaLiberoApi(env)
    previous = []

    def fake_post(url, payload, timeout=120):
        assert url.endswith("/ik")
        previous.append(payload["prev_cfg"])
        return {"joint_positions": list(range(8))}

    monkeypatch.setattr("embodied_codex.adapters.franka_libero_api._post", fake_post)
    api.goto_pose(np.array([.1, .2, .3]), np.array([1., 0, 0, 0]), z_approach=.1)
    assert env.calls[0][0] == "move_to_joints_blocking"
    np.testing.assert_array_equal(env.calls[0][1], np.arange(7))
    np.testing.assert_array_equal(api.cfg, np.arange(8))
    assert previous[0] is None
    assert previous[1] == list(range(8))


def test_contact_graspnet_pointcloud_payload_and_post_transform(monkeypatch):
    api = FrankaLiberoApi(Env())
    grasp = np.eye(4)[None]
    scores = np.array([0.7])

    def enc(value):
        buffer = io.BytesIO(); np.save(buffer, value)
        return base64.b64encode(buffer.getvalue()).decode()

    def fake_post(url, payload, timeout=120):
        assert url.endswith("/plan_point_clouds")
        assert payload == {
            "pc_full_base64": payload["pc_full_base64"],
            "pc_segment_base64": payload["pc_segment_base64"],
            "segmap_id": 1, "local_regions": True, "filter_grasps": True,
            "forward_passes": 2, "max_retries": 10,
        }
        return {"grasps_base64": enc(grasp), "scores_base64": enc(scores)}

    monkeypatch.setattr("embodied_codex.adapters.franka_libero_api._post", fake_post)
    transformed, result_scores = api.plan_grasp_from_point_clouds(
        np.zeros((2, 3)), np.zeros((1, 3)))
    np.testing.assert_allclose(transformed[0, :3, 3], [0.0, 0.0, 0.12])
    np.testing.assert_array_equal(result_scores, scores)


def test_molmo_parsing_matches_upstream_supported_formats(monkeypatch):
    monkeypatch.delenv("ROBOFORGE_MOLMO_MODEL", raising=False)
    api = FrankaLiberoApi(Env())
    image = np.zeros((100, 200, 3), np.uint8)
    replies = iter([
        '<points coords="0 1 250 750">object</points>',
        '<points x1="25" y1="75">object</points>',
    ])

    def fake_post(url, payload, **kwargs):
        assert url.endswith("/chat/completions")
        assert payload["stop"] == ["<|endoftext|>"]
        assert "model" not in payload
        return {"choices": [{"message": {"content": next(replies)}}]}

    monkeypatch.setattr("embodied_codex.adapters.franka_libero_api._post_with_retries", fake_post)
    assert api.point_prompt_molmo(image, "object") == {"object": (50, 75)}
    assert api.point_prompt_molmo(image, "object") == {"object": (50, 75)}


def test_molmo_uses_explicit_served_model_alias_when_configured(monkeypatch):
    monkeypatch.setenv("ROBOFORGE_MOLMO_MODEL", "served-molmo")
    captured = {}

    def fake_post(_url, payload, **_kwargs):
        captured.update(payload)
        return {"choices": [{"message": {"content": '<point x="50" y="50" />'}}]}

    monkeypatch.setattr("embodied_codex.adapters.franka_libero_api._post_with_retries", fake_post)
    result = FrankaLiberoApi(Env()).point_prompt_molmo(
        np.zeros((10, 10, 3), np.uint8), "object"
    )
    assert captured["model"] == "served-molmo"
    assert result == {"object": (5, 5)}


def test_sample_grasp_pose_applies_upstream_90_degree_post_rotation(monkeypatch):
    api = FrankaLiberoApi(Env())
    cam = {"rgb": np.zeros((1, 1, 3), np.uint8), "depth": np.ones((1, 1)),
           "intrinsics": np.eye(3), "pose_mat": np.eye(4)}
    monkeypatch.setattr(api, "get_object_3d_points_and_masks_from_language",
                        lambda *args, **kwargs: {"points_3d": np.tile([[0., 0., 1.]], (11, 1))})
    monkeypatch.setattr(api, "get_observation", lambda: {})
    monkeypatch.setattr(api, "_camera", lambda name: cam)
    monkeypatch.setattr(api, "filter_noise", lambda points, colors=None: (points, colors))
    monkeypatch.setattr(api, "plan_grasp_from_point_clouds",
                        lambda full, seg: (np.eye(4)[None], np.array([1.0])))
    pos, quat = api.sample_grasp_pose("object")
    np.testing.assert_allclose(pos, [0.0, 0.0, 0.0])
    expected = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    assert abs(float(np.dot(quat, expected))) > 1 - 1e-8


def test_top_down_grasp_matches_upstream_return_contract():
    api = FrankaLiberoApi(Env())
    grasps = np.tile(np.eye(4), (2, 1, 1)); grasps[0, 2, 2] = -1
    selected, score = api.select_top_down_grasp(grasps, np.array([.2, .8]), np.eye(4))
    np.testing.assert_allclose(selected, grasps[0])
    assert score == .2
    none, score = api.select_top_down_grasp(np.tile(np.eye(4), (1, 1, 1)), np.array([.2]), np.eye(4))
    assert none is None and score == -np.inf


def test_pyroki_traj_plan_and_blocking_trajectory_contract(monkeypatch):
    env = Env(); api = FrankaLiberoApi(env)
    seen = {}

    def fake_post(url, payload, timeout=120):
        assert url.endswith("/plan")
        seen.update(payload)
        return {"waypoints": [[1, 2, 3, 4, 5, 6, 7, 0.5], [2, 3, 4, 5, 6, 7, 8, 0.2]]}

    monkeypatch.setattr("embodied_codex.adapters.franka_libero_api._post", fake_post)
    result = api.traj_plan(np.arange(7), np.arange(7) + 1)
    assert result.shape == (2, 7)
    assert seen == {"start_pose_wxyz_xyz": list(range(7)),
                    "end_pose_wxyz_xyz": list(range(1, 8))}
    api.move_along_trajectory(result)
    moves = [call for call in env.calls if call[0] == "move_to_joints_blocking"]
    assert len(moves) == 2
    assert moves[0][2] == {"tolerance": 0.025, "max_steps": 15}


def test_dual_arm_methods_forward_only_when_provider_supports_them(monkeypatch):
    class DualEnv(Env):
        def move_to_joints_blocking_arm1(self, joints, **kwargs):
            self.calls.append(("arm1", np.asarray(joints), kwargs))
        def move_to_joints_blocking_both(self, joints0, joints1):
            self.calls.append(("both", np.asarray(joints0), np.asarray(joints1)))

    env = DualEnv(); api = FrankaLiberoApi(env)
    assert api.supports_dual_arm() is True
    api.move_to_joints_both(np.zeros(7), np.ones(7))
    api.move_to_joints_arm1(np.ones(7))
    assert [x[0] for x in env.calls] == ["both", "arm1"]
    api._solve_ik_with_prev = lambda position, quaternion_wxyz, prev_cfg=None: np.arange(7.)
    np.testing.assert_array_equal(api.solve_ik_arm0(np.zeros(3), np.array([1., 0, 0, 0])), np.arange(7.))

    class TransformDual(DualEnv):
        arm0_to_arm1_transform = np.eye(4)
    api = FrankaLiberoApi(TransformDual())
    api._solve_ik_with_prev = lambda position, quaternion_wxyz, prev_cfg=None: np.arange(7.)
    np.testing.assert_array_equal(api.solve_ik_arm1(np.zeros(3), np.array([1., 0, 0, 0])), np.arange(7.))
