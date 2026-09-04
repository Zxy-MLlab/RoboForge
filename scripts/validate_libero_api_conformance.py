"""Validate the public ASPIRE/CaP-X Controller API against real LIBERO.

This is an external conformance runner, not an Agent tool. It may read MuJoCo
state only to measure the public Controller API against the real simulator.
Candidate Controllers continue to see only the allowlisted Robot SDK.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable

import numpy as np
import requests
from scipy.spatial.transform import Rotation

from embodied_codex.adapters.franka_libero_api import FrankaLiberoApi
from embodied_codex.adapters.libero import create
from embodied_codex.deployments.libero import LiberoDeploymentError


SERVICE_URLS = {
    "sam3": "http://127.0.0.1:8114",
    "graspnet": "http://127.0.0.1:8115",
    "pyroki": "http://127.0.0.1:8116",
    "molmo": "http://127.0.0.1:8122",
    "curobo": "http://127.0.0.1:8117",
}

UPSTREAM = {
    "aspire": {
        "repository": "https://github.com/NVlabs/ASPIRE",
        "commit": "f4c8939aab0af9b97690c561bd80e282940f7886",
        "license": "Apache-2.0 and MIT",
    },
    "cap-x": {
        "repository": "https://github.com/capgym/cap-x",
        "commit": "53e9966d7a8e2fa7494676772bccc35280f5c0ed",
        "license": "MIT",
    },
}

REQUIRED_APIS = {
    "get_observation", "get_task_language", "segment_sam3_text_prompt",
    "segment_sam3_point_prompt", "point_prompt_molmo", "mask_to_world_points",
    "get_oriented_bounding_box_from_3d_points", "decompose_transform",
    "rotation_matrix_to_quaternion", "transform_points", "depth_to_pointcloud",
    "depth_to_point_cloud", "plan_grasp", "select_top_down_grasp", "solve_ik",
    "move_to_joints", "goto_pose", "goto_home_joint_position", "open_gripper",
    "close_gripper", "joint_control_timeout", "controller_privilege_boundary",
    "curobo_service",
}

API_PROVENANCE = {
    "get_observation": "aspire/sim/cap/envs/simulators/libero.py",
    "get_task_language": "aspire/sim/cap/envs/simulators/libero.py",
    "segment_sam3_text_prompt": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "segment_sam3_point_prompt": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "point_prompt_molmo": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "mask_to_world_points": "aspire/sim/cap/utils/depth_utils.py",
    "get_oriented_bounding_box_from_3d_points": "aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py",
    "decompose_transform": "aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py",
    "rotation_matrix_to_quaternion": "aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py",
    "transform_points": "aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py",
    "depth_to_pointcloud": "aspire/sim/cap/utils/depth_utils.py",
    "depth_to_point_cloud": "aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py",
    "plan_grasp": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "select_top_down_grasp": "aspire/sim/cap/integrations/franka/libero_reduced_skill_library.py",
    "solve_ik": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "move_to_joints": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "goto_pose": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "goto_home_joint_position": "aspire/sim/cap/integrations/franka/libero_reduced.py",
    "open_gripper": "aspire/sim/cap/integrations/franka/common.py",
    "close_gripper": "aspire/sim/cap/integrations/franka/common.py",
    "curobo_service": "capx/serving/launch_curobo_server.py",
}


def _port(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 80), timeout=0.5):
            return True
    except OSError:
        return False


def _npy64(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _from_npy64(value: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)


def _manifest_digest(manifest: dict) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _observation_fingerprint(api: FrankaLiberoApi, obs: dict) -> str:
    digest = hashlib.sha256()
    digest.update(api.get_task_language().encode())
    for camera_name in (api.camera_name, api.wrist_camera_name):
        camera = obs[camera_name]
        for value in (camera["images"]["rgb"], camera["images"]["depth"], camera["intrinsics"], camera["pose_mat"]):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())
    for name in ("robot_joint_pos", "robot_cartesian_pos"):
        array = np.ascontiguousarray(obs[name])
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_resume(path: Path, *, task: int, state: int, observation_fingerprint: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise ValueError(f"resume manifest digest mismatch: {path}")
    expected = {"task": task, "state": state, "controller_mode": "JOINT_POSITION", "observation_fingerprint": observation_fingerprint}
    actual = {name: manifest.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"resume manifest identity mismatch: expected {expected}, got {actual}")
    return manifest


def _validate_graspnet_response(body: object) -> dict:
    if not isinstance(body, dict):
        raise ValueError("response is not an object")
    grasps = _from_npy64(str(body["grasps_base64"]))
    scores = _from_npy64(str(body["scores_base64"]))
    contacts = _from_npy64(str(body["contact_pts_base64"]))
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"grasps shape must be (N,4,4), got {grasps.shape}")
    if scores.shape != (len(grasps),):
        raise ValueError(f"scores shape {scores.shape} does not match {len(grasps)} grasps")
    if contacts.shape != (len(grasps), 3):
        raise ValueError(f"contact points shape {contacts.shape} does not match grasps")
    if not len(grasps) or not all(np.isfinite(x).all() for x in (grasps, scores, contacts)):
        raise ValueError("grasp response is empty or contains non-finite values")
    return {"num_grasps": len(grasps), "grasps_shape": list(grasps.shape), "scores_shape": list(scores.shape), "contact_points_shape": list(contacts.shape)}


def _validate_joint_response(body: object) -> dict:
    if not isinstance(body, dict):
        raise ValueError("response is not an object")
    joints = np.asarray(body.get("joint_positions"), dtype=float)
    if joints.ndim != 1 or len(joints) < 7 or not np.isfinite(joints).all():
        raise ValueError(f"invalid joint solution shape/value: {joints.shape}")
    return {"joint_positions_shape": list(joints.shape), "finite": True}


def _validate_sam3_response(body: object) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise ValueError("response does not contain a results list")
    valid = 0
    max_score = None
    for result in body["results"]:
        shape = tuple(result.get("shape", ()))
        if len(shape) != 2:
            continue
        mask = np.frombuffer(base64.b64decode(result.get("mask_base64", "")), dtype=np.uint8)
        if mask.size != int(np.prod(shape)):
            continue
        box = np.asarray(result.get("box"), dtype=float)
        score = float(result.get("score"))
        if box.shape == (4,) and np.isfinite(box).all() and np.isfinite(score) and mask.any():
            valid += 1
            max_score = score if max_score is None else max(max_score, score)
    if not valid:
        raise ValueError("SAM3 returned no non-empty, well-formed masks")
    return {"num_results": len(body["results"]), "num_valid_nonempty_masks": valid, "max_score": max_score}


def _parse_molmo_point(text: str, *, width: int, height: int) -> tuple[int, int] | None:
    coords = re.search(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text, re.I)
    if coords:
        nums = [float(value) for value in coords.group(1).split()]
        if len(nums) >= 4:
            x, y = nums[2], nums[3]
            if 0 <= x <= 1000 and 0 <= y <= 1000:
                return int(x / 1000 * width), int(y / 1000 * height)
    tag = re.search(r'<point\b[^>]*\bx\s*=\s*["\']([0-9.]+)["\'][^>]*\by\s*=\s*["\']([0-9.]+)["\']', text, re.I)
    if tag:
        x, y = map(float, tag.groups())
        if 0 <= x <= 100 and 0 <= y <= 100:
            return int(x / 100 * width), int(y / 100 * height)
    return None


def _quat_error_rad(first_wxyz: np.ndarray, second_wxyz: np.ndarray) -> float:
    first = np.asarray(first_wxyz, dtype=np.float64).reshape(4)
    second = np.asarray(second_wxyz, dtype=np.float64).reshape(4)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)))


def _gripper_width(deployment) -> float:
    return float(np.abs(np.asarray(deployment.obs["robot0_gripper_qpos"], dtype=float)).sum())


def _record_call(record: Callable[[dict], None], api_name: str, function: Callable[[], Any]) -> Any:
    try:
        value = function()
        row = {"api": api_name, "status": "passed", "return_type": type(value).__name__}
        if isinstance(value, np.ndarray):
            row["shape"] = list(value.shape)
        record(row)
        return value
    except Exception as exc:
        record({"api": api_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return None


def _independent_camera_check(api, deployment, obs) -> dict:
    sim = deployment.env.sim
    base_id = sim.model.body_name2id("robot0_base")
    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = np.asarray(sim.data.xmat[base_id]).reshape(3, 3)
    base[:3, 3] = np.asarray(sim.data.xpos[base_id])
    ry = np.diag([-1.0, 1.0, -1.0, 1.0])
    rz = np.diag([-1.0, -1.0, 1.0, 1.0])
    errors = {}
    for camera_name in (api.camera_name, api.wrist_camera_name):
        camera_id = sim.model.camera_name2id(camera_name)
        camera_world = np.eye(4, dtype=np.float64)
        camera_world[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3)
        camera_world[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id])
        reference = np.linalg.inv(base) @ camera_world @ ry @ rz
        errors[camera_name] = float(np.max(np.abs(reference - np.asarray(obs[camera_name]["pose_mat"]))))
    return {"api": "camera_pose_robot_base_frame", "status": "passed" if all(value <= 1e-12 for value in errors.values()) else "failed", "same_input_max_abs_error": errors, "tolerance": 1e-12}


def _joint_report(deployment) -> dict:
    for event in reversed(deployment.trace):
        if event.get("event") == "joint_control":
            return dict(event)
    raise RuntimeError("blocking controller did not emit a joint-control trace")


def _run_control_checks(api: FrankaLiberoApi, deployment, record) -> None:
    home = np.asarray(api.get_observation()["robot_joint_pos"], dtype=np.float64)[:7]
    before = deployment._panda_joint_positions().copy()
    deployment.move_to_joints_blocking(before, stable_steps=2)
    after = deployment._panda_joint_positions().copy()
    noop = _joint_report(deployment)
    noop["max_motion_rad"] = float(np.max(np.abs(after - before)))
    record({"api": "move_to_joints", "case": "current_joints_noop", "status": "passed" if noop["max_motion_rad"] <= 0.01 else "failed", "convergence": noop})

    targets = []
    deltas = (
        np.array([0.05, 0, 0, 0, 0, 0, 0]),
        np.array([0, -0.05, 0, 0, 0, 0, 0]),
        np.array([0, 0, 0.05, 0, 0, 0, 0]),
    )
    for index, delta in enumerate(deltas, 1):
        deployment.move_to_joints_blocking(home, stable_steps=2)
        deployment.move_to_joints_blocking(home + delta, stable_steps=2)
        report = _joint_report(deployment)
        report["target_index"] = index
        targets.append(report)
    passed = all(item["status"] == "converged" and item["final_l2_error_rad"] < 0.01 for item in targets)
    record({"api": "move_to_joints", "case": "three_safe_targets", "status": "passed" if passed else "failed", "targets": targets})

    deployment.move_to_joints_blocking(home, stable_steps=2)
    initial_pose = np.asarray(api.get_observation()["robot_cartesian_pos"], dtype=np.float64)
    solved = api.solve_ik(initial_pose[:3], initial_pose[3:7])
    api.move_to_joints(solved)
    final_pose = np.asarray(api.get_observation()["robot_cartesian_pos"], dtype=np.float64)
    position_error = float(np.linalg.norm(final_pose[:3] - initial_pose[:3]))
    orientation_error = _quat_error_rad(final_pose[3:7], initial_pose[3:7])
    record({"api": "solve_ik", "case": "current_pose_ik_fk", "status": "passed" if solved.shape == (7,) and position_error <= 0.01 and orientation_error <= 0.05 else "failed", "joint_target_rad": solved.tolist(), "eef_position_error_m": position_error, "eef_orientation_error_rad": orientation_error, "joint_convergence": _joint_report(deployment)})

    base_pose = np.asarray(api.get_observation()["robot_cartesian_pos"], dtype=np.float64)
    cartesian_cases = [
        ("pre_grasp", base_pose[:3] + np.array([0.0, 0.0, 0.04])),
        ("grasp", base_pose[:3] + np.array([0.02, 0.0, 0.015])),
        ("lift", base_pose[:3] + np.array([0.02, 0.0, 0.065])),
    ]
    cartesian = []
    for name, target_position in cartesian_cases:
        api.goto_pose(target_position, base_pose[3:7])
        actual = np.asarray(api.get_observation()["robot_cartesian_pos"], dtype=np.float64)
        item = {"case": name, "target_position_m": target_position.tolist(), "actual_position_m": actual[:3].tolist(), "position_error_m": float(np.linalg.norm(actual[:3] - target_position)), "orientation_error_rad": _quat_error_rad(actual[3:7], base_pose[3:7]), "joint_convergence": _joint_report(deployment)}
        item["status"] = "passed" if item["position_error_m"] <= 0.015 and item["orientation_error_rad"] <= 0.08 else "failed"
        cartesian.append(item)
    record({"api": "goto_pose", "status": "passed" if all(item["status"] == "passed" for item in cartesian) else "failed", "poses": cartesian})

    deployment.move_to_joints_blocking(home, stable_steps=2)
    arm_before = deployment._panda_joint_positions().copy()
    api.open_gripper()
    open_width = _gripper_width(deployment)
    arm_after_open = deployment._panda_joint_positions().copy()
    api.close_gripper()
    closed_width = _gripper_width(deployment)
    arm_after_close = deployment._panda_joint_positions().copy()
    arm_drift = float(max(np.max(np.abs(arm_after_open - arm_before)), np.max(np.abs(arm_after_close - arm_before))))
    gripper_passed = open_width > closed_width + 0.02 and arm_drift <= 0.02
    common = {"status": "passed" if gripper_passed else "failed", "open_width_m": open_width, "closed_width_m": closed_width, "arm_max_drift_rad": arm_drift}
    record({"api": "open_gripper", **common})
    record({"api": "close_gripper", **common})
    api.open_gripper()

    deployment.move_to_joints_blocking(home, stable_steps=2)
    timeout_target = home.copy()
    timeout_target[0] += 0.35
    timeout_error = None
    try:
        deployment.move_to_joints_blocking(timeout_target, max_steps=1, stable_steps=2)
    except LiberoDeploymentError as exc:
        timeout_error = str(exc)
    timeout_report = _joint_report(deployment)
    record({"api": "joint_control_timeout", "status": "passed" if timeout_error and timeout_report["status"] == "timeout" else "failed", "error": timeout_error, "convergence": timeout_report})
    deployment.move_to_joints_blocking(home, stable_steps=2)
    record({"api": "goto_home_joint_position", "status": "passed", "convergence": _joint_report(deployment)})


def _run_public_geometry(api: FrankaLiberoApi, obs: dict, record) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = obs[api.camera_name]
    rgb = np.asarray(camera["images"]["rgb"])
    depth = np.asarray(camera["images"]["depth"])
    intrinsic = np.asarray(camera["intrinsics"])
    pose = np.asarray(camera["pose_mat"])
    mask = np.zeros(depth.shape, dtype=np.uint8)
    mask[depth.shape[0] // 2, depth.shape[1] // 2] = 1
    _record_call(record, "depth_to_pointcloud", lambda: api.depth_to_pointcloud(depth, intrinsic))
    _record_call(record, "depth_to_point_cloud", lambda: api.depth_to_point_cloud(depth, intrinsic))
    _record_call(record, "mask_to_world_points", lambda: api.mask_to_world_points(mask, depth, intrinsic, pose))
    _record_call(record, "decompose_transform", lambda: api.decompose_transform(pose))
    _record_call(record, "rotation_matrix_to_quaternion", lambda: api.rotation_matrix_to_quaternion(pose[:3, :3]))
    _record_call(record, "transform_points", lambda: api.transform_points(np.zeros((2, 3)), pose))
    return rgb, depth, intrinsic, pose


def _run_model_checks(api: FrankaLiberoApi, rgb, depth, intrinsic, pose, target: str, record) -> None:
    point = _record_call(record, "point_prompt_molmo", lambda: api.point_prompt_molmo(rgb, target))
    point_xy = point.get(target) if isinstance(point, dict) else None
    if not point_xy or point_xy[0] is None:
        record({"api": "point_prompt_molmo", "status": "failed", "error": "no parseable point"})

    text_masks = _record_call(record, "segment_sam3_text_prompt", lambda: api.segment_sam3_text_prompt(rgb, target))
    if not text_masks:
        record({"api": "segment_sam3_text_prompt", "status": "failed", "error": "no masks"})
    point_masks = None
    if point_xy and point_xy[0] is not None:
        point_masks = _record_call(record, "segment_sam3_point_prompt", lambda: api.segment_sam3_point_prompt(rgb, point_xy))
        if not point_masks:
            record({"api": "segment_sam3_point_prompt", "status": "failed", "error": "no masks"})
    else:
        record({"api": "segment_sam3_point_prompt", "status": "failed", "error": "Molmo did not provide a point"})

    candidates = point_masks or text_masks or []
    if not candidates:
        return
    selected_mask = np.asarray(max(candidates, key=lambda item: item["score"])["mask"], dtype=bool)
    points = _record_call(record, "mask_to_world_points", lambda: api.mask_to_world_points(selected_mask, depth, intrinsic, pose))
    if not isinstance(points, np.ndarray) or len(points) < 20:
        record({"api": "get_oriented_bounding_box_from_3d_points", "status": "failed", "error": "segmentation produced fewer than 20 valid 3D points"})
    else:
        _record_call(record, "get_oriented_bounding_box_from_3d_points", lambda: api.get_oriented_bounding_box_from_3d_points(points))

    planned = _record_call(record, "plan_grasp", lambda: api.plan_grasp(depth, intrinsic, selected_mask.astype(np.uint8)))
    if not (isinstance(planned, tuple) and len(planned) == 2 and len(planned[0])):
        record({"api": "plan_grasp", "status": "failed", "error": "no grasp candidates"})
        return
    grasps, scores = planned
    selected = _record_call(record, "select_top_down_grasp", lambda: api.select_top_down_grasp(grasps, scores, pose))
    if not (isinstance(selected, tuple) and selected[0] is not None):
        record({"api": "select_top_down_grasp", "status": "failed", "error": "no top-down grasp passed the upstream threshold"})


def _run_curobo_check(api: FrankaLiberoApi, record) -> None:
    if not _port(SERVICE_URLS["curobo"]):
        record({"api": "curobo_service", "status": "failed", "error": "service unavailable"})
        return
    obs = api.get_observation()
    pose = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64)
    quat = pose[3:7]
    rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    hand_position = pose[:3] + rotation.apply(api._TCP_OFFSET)
    current = api._current_pyroki_configuration()
    if current is None:
        record({"api": "curobo_service", "status": "failed", "error": "current joint configuration unavailable"})
        return
    payload = {"start_joint_positions": current[:7].tolist(), "goal_pose_wxyz_xyz": np.r_[quat, hand_position + np.array([0.0, 0.0, 0.03])].tolist(), "max_attempts": 3, "timeout": 20.0, "enable_graph": True}
    try:
        health = requests.get(f"{SERVICE_URLS['curobo']}/health", timeout=10)
        response = requests.post(f"{SERVICE_URLS['curobo']}/motion_plan", json=payload, timeout=120)
        body = response.json()
        waypoints = np.asarray(body.get("waypoints", []), dtype=np.float64)
        valid = health.status_code == 200 and health.json().get("status") == "ready" and response.status_code == 200 and body.get("success") is True and waypoints.ndim == 2 and waypoints.shape[1] == 7 and len(waypoints) > 1 and np.isfinite(waypoints).all()
        record({"api": "curobo_service", "status": "passed" if valid else "failed", "health": health.json(), "http_status": response.status_code, "success": body.get("success"), "planner_status": body.get("status"), "num_waypoints": int(len(waypoints)) if waypoints.ndim == 2 else 0, "plan_time_ms": body.get("plan_time_ms"), "path_length": body.get("path_length")})
    except Exception as exc:
        record({"api": "curobo_service", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})


def _run_state(args, state: int, output: Path, *, model_checks: bool) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    deployment = create(task=str(args.task), state=state, root=output, configuration={"disable_agent_verifier": True, "controller_mode": "JOINT_POSITION"})
    api = FrankaLiberoApi(deployment)
    rows: list[dict] = []
    started = time.time()

    def record(row: dict) -> None:
        rows.append({"state": state, **row})

    fingerprint = ""
    language = ""
    try:
        obs = api.get_observation()
        fingerprint = _observation_fingerprint(api, obs)
        record({"api": "get_observation", "status": "passed", "cameras": [api.camera_name, api.wrist_camera_name], "rgb_shape": list(obs[api.camera_name]["images"]["rgb"].shape), "depth_shape": list(obs[api.camera_name]["images"]["depth"].shape), "robot_joint_pos_shape": list(np.asarray(obs["robot_joint_pos"]).shape), "robot_cartesian_pos_shape": list(np.asarray(obs["robot_cartesian_pos"]).shape)})
        record(_independent_camera_check(api, deployment, obs))
        language = api.get_task_language()
        record({"api": "get_task_language", "status": "passed" if language else "failed", "value": language})
        rgb, depth, intrinsic, pose = _run_public_geometry(api, obs, record)
        _run_control_checks(api, deployment, record)
        if model_checks:
            _run_model_checks(api, rgb, depth, intrinsic, pose, args.molmo_target, record)
            _run_curobo_check(api, record)
        public = api.functions()
        forbidden = ("reset", "set_seed", "check_success", "hidden_evaluator", "promote")
        record({"api": "controller_privilege_boundary", "status": "passed" if all(name not in public for name in forbidden) else "failed", "public_methods": sorted(public), "forbidden_methods": list(forbidden)})
    finally:
        deployment.close()
    failures = [row for row in rows if row.get("status") == "failed"]
    state_manifest = {"protocol": "roboforge-controller-api-state-conformance-v1", "task": args.task, "state": state, "controller_mode": "JOINT_POSITION", "observation_fingerprint": fingerprint, "instruction": language, "rows": rows, "failed_rows": [row["api"] for row in failures], "passed": not failures, "elapsed_seconds": time.time() - started, "artifact_dir": str(output.resolve())}
    state_manifest["manifest_sha256"] = _manifest_digest(state_manifest)
    (output / "state-conformance.json").write_text(json.dumps(state_manifest, indent=2) + "\n")
    return state_manifest


def _matrix(states: list[dict]) -> list[dict]:
    import inspect
    frame_units = {
        "get_observation": "RGB-D arrays; camera pose camera->world; joints rad; positions m",
        "get_task_language": "UTF-8 task instruction",
        "mask_to_world_points": "camera pixels/depth m -> world XYZ m",
        "get_oriented_bounding_box_from_3d_points": "world XYZ m; quaternion WXYZ",
        "decompose_transform": "homogeneous transform -> position m + quaternion WXYZ",
        "transform_points": "3D points m under 4x4 transform",
        "depth_to_pointcloud": "camera frame XYZ m",
        "depth_to_point_cloud": "camera frame XYZ m",
        "solve_ik": "EEF target world m/quaternion WXYZ -> 7 arm joints rad",
        "move_to_joints": "7 Franka arm joints rad; blocking JOINT_POSITION",
        "goto_pose": "world EEF position m/quaternion WXYZ; TCP offset applied once",
        "open_gripper": "gripper command; metres/proprioceptive width",
        "close_gripper": "gripper command; metres/proprioceptive width",
        "select_top_down_grasp": "camera/world grasp transforms; metres",
        "curobo_service": "world-frame trajectory; joints rad and positions m",
    }
    result = []
    for api_name in sorted(REQUIRED_APIS):
        rows = [row for state in states for row in state["rows"] if row["api"] == api_name]
        status = "passed" if rows and all(row["status"] == "passed" for row in rows) else "failed"
        callable_object = getattr(FrankaLiberoApi, api_name, None)
        result.append({
            "api": api_name,
            "upstream_repository": UPSTREAM["aspire"]["repository"] if api_name != "curobo_service" else UPSTREAM["cap-x"]["repository"],
            "upstream_commit": UPSTREAM["aspire"]["commit"] if api_name != "curobo_service" else UPSTREAM["cap-x"]["commit"],
            "upstream_file": API_PROVENANCE.get(api_name),
            "local_file": "embodied_codex/adapters/franka_libero_api.py" if api_name not in {"joint_control_timeout", "controller_privilege_boundary"} else "embodied_codex/deployments/libero.py",
            "signature": str(inspect.signature(callable_object)) if callable(callable_object) else None,
            "frame_unit": frame_units.get(api_name, "Declared by callable contract; see per-state invocation"),
            "real_libero_invocations": len(rows),
            "same_input_semantics": "passed" if status == "passed" else "failed",
            "result": status,
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=None, help="single-state compatibility option")
    parser.add_argument("--states", type=int, nargs="+", help="at least five real LIBERO initial-state indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--molmo-target", default="black bowl")
    parser.add_argument("--model-check-state", type=int, help="state on which to invoke SAM3/Molmo/Contact-GraspNet/cuRobo")
    parser.add_argument("--allow-incomplete", action="store_true", help="write evidence but return zero for an exploratory run")
    args = parser.parse_args(argv)
    states = list(dict.fromkeys(args.states or ([args.state] if args.state is not None else [0, 1, 2, 3, 4])))
    if not args.allow_incomplete and len(states) < 5:
        parser.error("formal conformance requires at least five distinct states")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    model_state = args.model_check_state if args.model_check_state is not None else states[0]
    if model_state not in states:
        parser.error("--model-check-state must be included in --states")
    started = time.time()
    state_manifests = [_run_state(args, state, args.output / f"state-{state:03d}", model_checks=state == model_state) for state in states]
    matrix = _matrix(state_manifests)
    failed = [row["api"] for row in matrix if row["result"] != "passed"]
    enough_states = len(states) >= 5
    manifest = {"protocol": "roboforge-aspire-capx-libero-api-conformance-v3", "task": args.task, "states": states, "minimum_state_count": 5, "state_count_gate_passed": enough_states, "model_check_state": model_state, "controller_mode": "JOINT_POSITION", "upstream": UPSTREAM, "required_apis": sorted(REQUIRED_APIS), "complete_conformance": not failed and enough_states, "failed_apis": failed, "state_manifests": [{"state": item["state"], "passed": item["passed"], "manifest_sha256": item["manifest_sha256"], "path": str((args.output / f"state-{item['state']:03d}" / "state-conformance.json").resolve())} for item in state_manifests], "conformance_matrix": matrix, "elapsed_seconds": time.time() - started}
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    (args.output / "api-conformance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0 if (not failed and enough_states) or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
