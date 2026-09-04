"""Run the non-privileged ASPIRE/CaP-X Controller API against real LIBERO.

Service-backed calls are reported as unavailable when their upstream service is
not running; no mock response is substituted.  The resulting JSON is a light
weight invocation manifest suitable for the conformance matrix.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
import socket
import time
import base64
import io
import requests

import numpy as np

from embodied_codex.adapters.franka_libero_api import FrankaLiberoApi
from embodied_codex.adapters.libero import create


SERVICE_URLS = {
    "sam3": "http://127.0.0.1:8114",
    "graspnet": "http://127.0.0.1:8115",
    "pyroki": "http://127.0.0.1:8116",
    "molmo": "http://127.0.0.1:8122",
    "curobo": "http://127.0.0.1:8117",
}

SERVICE_PROBE_APIS = {
    "sam3": "sam3_text_segmentation_live",
    "graspnet": "contact_graspnet_live",
    "pyroki": "pyroki_live",
    "molmo": "molmo_vision_live",
    "curobo": "curobo_live",
}

BASE_REQUIRED_APIS = {
    "get_observation",
    "camera_pose_robot_base_frame",
    "robot_state_shapes",
    "depth_to_pointcloud",
    "mask_to_world_points",
    "pixel_to_world_point",
    "decompose_transform",
    "rotation_matrix_to_quaternion",
    "transform_points",
    "interpolate_segment",
    "normalize_vector",
    "goto_home_joint_position",
    "open_gripper",
    "close_gripper",
    "task_language",
    "supports_dual_arm",
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
    buf = io.BytesIO(); np.save(buf, np.asarray(value)); return base64.b64encode(buf.getvalue()).decode("ascii")


def _from_npy64(value: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(value)), allow_pickle=False)


def _manifest_digest(manifest: dict) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True, indent=2).encode()
    return hashlib.sha256(encoded).hexdigest()


def _observation_fingerprint(api: FrankaLiberoApi, obs: dict) -> str:
    """Hash controller-visible reset state used by every serial probe phase."""
    digest = hashlib.sha256()
    digest.update(api.task_language().encode())
    for camera_name in (api.camera_name, api.wrist_camera_name):
        camera = obs[camera_name]
        for value in (
            camera["images"]["rgb"], camera["images"]["depth"],
            camera["intrinsics"], camera["pose_mat"],
        ):
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


def _validated_resume(path: Path, *, task: int, state: int,
                      observation_fingerprint: str) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("manifest_sha256") != _manifest_digest(manifest):
        raise ValueError(f"resume manifest digest mismatch: {path}")
    expected = {
        "task": task,
        "state": state,
        "controller_mode": "JOINT_POSITION",
        "observation_fingerprint": observation_fingerprint,
    }
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
    return {"num_grasps": len(grasps), "grasps_shape": list(grasps.shape),
            "scores_shape": list(scores.shape), "contact_points_shape": list(contacts.shape)}


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
    return {"num_results": len(body["results"]), "num_valid_nonempty_masks": valid,
            "max_score": max_score}


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--services",
        default=",".join(SERVICE_URLS),
        help="comma-separated live services to probe in this phase",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="verified earlier phase for the same task/state/observation",
    )
    parser.add_argument(
        "--molmo-target",
        default="black bowl",
        help="text target used by the live Molmo point probe",
    )
    parser.add_argument(
        "--molmo-expected-box",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="optional audited pixel box; the returned point must fall inside",
    )
    parser.add_argument(
        "--allow-unavailable-services",
        action="store_true",
        help="run only the LIBERO/geometry smoke; complete conformance is false",
    )
    args = parser.parse_args()
    selected_services = {name.strip() for name in args.services.split(",") if name.strip()}
    unknown_services = selected_services - set(SERVICE_URLS)
    if unknown_services:
        parser.error(f"unknown services: {sorted(unknown_services)}")
    args.output.mkdir(parents=True, exist_ok=False)
    deployment = create(task=str(args.task), state=args.state, root=args.output,
                        configuration={"disable_agent_verifier": True,
                                       "controller_mode": "JOINT_POSITION"})
    api = FrankaLiberoApi(deployment)
    rows_by_api: dict[str, dict] = {}
    started = time.time()
    try:
        obs = api.get_observation()
        observation_fingerprint = _observation_fingerprint(api, obs)
        prior_manifest = None
        if args.resume_from:
            prior_manifest = _validated_resume(
                args.resume_from,
                task=args.task,
                state=args.state,
                observation_fingerprint=observation_fingerprint,
            )
            for prior_row in prior_manifest.get("rows", []):
                row = copy.deepcopy(prior_row)
                row["preserved_from"] = str(args.resume_from.resolve())
                rows_by_api[row["api"]] = row

        def record(row: dict) -> None:
            rows_by_api[row["api"]] = row

        record({"api": "get_observation", "status": "passed",
                     "cameras": sorted(k for k in obs if k in (api.camera_name, api.wrist_camera_name)),
                     "rgb_shape": list(obs[api.camera_name]["images"]["rgb"].shape),
                     "depth_shape": list(obs[api.camera_name]["images"]["depth"].shape)})
        # Independently reproduce ASPIRE FrankaLiberoEnv's camera transform
        # from MuJoCo raw state.  This comparison deliberately does not call
        # the implementation under test.
        sim = deployment.env.sim
        base_id = sim.model.body_name2id("robot0_base")
        base = np.eye(4, dtype=np.float64)
        base[:3, :3] = np.asarray(sim.data.xmat[base_id]).reshape(3, 3)
        base[:3, 3] = np.asarray(sim.data.xpos[base_id])
        ry = np.diag([-1.0, 1.0, -1.0, 1.0])
        rz = np.diag([-1.0, -1.0, 1.0, 1.0])
        frame_errors = {}
        for camera_name in (api.camera_name, api.wrist_camera_name):
            camera_id = sim.model.camera_name2id(camera_name)
            camera_world = np.eye(4, dtype=np.float64)
            camera_world[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3)
            camera_world[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id])
            reference = np.linalg.inv(base) @ camera_world @ ry @ rz
            frame_errors[camera_name] = float(np.max(np.abs(
                reference - np.asarray(obs[camera_name]["pose_mat"], dtype=np.float64))))
        frame_passed = all(value <= 1e-12 for value in frame_errors.values())
        record({"api": "camera_pose_robot_base_frame", "status": "passed" if frame_passed else "failed",
                     "same_input_max_abs_error": frame_errors, "tolerance": 1e-12})
        record({"api": "robot_state_shapes", "status": "passed"
                     if np.asarray(obs["robot_joint_pos"]).shape == (8,)
                     and np.asarray(obs["robot_cartesian_pos"]).shape == (8,) else "failed",
                     "robot_joint_pos_shape": list(np.asarray(obs["robot_joint_pos"]).shape),
                     "robot_cartesian_pos_shape": list(np.asarray(obs["robot_cartesian_pos"]).shape)})
        K = np.asarray(obs[api.camera_name]["intrinsics"]); pose = np.asarray(obs[api.camera_name]["pose_mat"])
        depth = np.asarray(obs[api.camera_name]["images"]["depth"])
        mask = np.zeros(depth.shape, dtype=np.uint8); mask[depth.shape[0] // 2, depth.shape[1] // 2] = 1
        for name, fn, call in [
            ("depth_to_pointcloud", api.depth_to_pointcloud, (depth, K)),
            ("mask_to_world_points", api.mask_to_world_points, (mask, depth, K, pose)),
            ("pixel_to_world_point", api.pixel_to_world_point, (depth.shape[1] // 2, depth.shape[0] // 2, float(depth[depth.shape[0] // 2, depth.shape[1] // 2]), K, pose)),
            ("decompose_transform", api.decompose_transform, (pose,)),
            ("rotation_matrix_to_quaternion", api.rotation_matrix_to_quaternion, (pose[:3, :3],)),
            ("transform_points", api.transform_points, (np.zeros((2, 3)), pose)),
            ("interpolate_segment", api.interpolate_segment, (np.zeros(3), np.ones(3))),
            ("normalize_vector", api.normalize_vector, (np.ones(3),)),
        ]:
            try:
                value = fn(*call)
                record({"api": name, "status": "passed", "return_type": type(value).__name__})
            except Exception as exc:
                record({"api": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        api.goto_home_joint_position(); record({"api": "goto_home_joint_position", "status": "passed"})
        api.open_gripper(); record({"api": "open_gripper", "status": "passed"})
        api.close_gripper(); record({"api": "close_gripper", "status": "passed"})
        record({"api": "task_language", "status": "passed", "value": api.task_language()})
        record({"api": "supports_dual_arm", "status": "passed", "value": api.supports_dual_arm()})
        # Exercise the deployed planning services with this real LIBERO
        # observation.  Responses are recorded by shape/status only here;
        # the complete native payloads are retained in the experiment evidence
        # directory by the dedicated live-invocation runner.
        service_calls = [
            ("contact_graspnet_live", f"{SERVICE_URLS['graspnet']}/plan", {
                "depth_base64": _npy64(depth), "cam_K_base64": _npy64(K),
                "segmap_base64": _npy64(mask.astype(np.uint8)), "segmap_id": 1,
                "local_regions": True, "filter_grasps": False,
                "skip_border_objects": False, "z_range": [0.2, 2.0],
                "forward_passes": 1, "max_retries": 1}),
            ("pyroki_live", f"{SERVICE_URLS['pyroki']}/ik", {
                "target_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.3]}),
            ("curobo_live", f"{SERVICE_URLS['curobo']}/ik", {
                "target_pose_wxyz_xyz": [1.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.3]}),
        ]
        service_names = {"contact_graspnet_live": "graspnet", "pyroki_live": "pyroki", "curobo_live": "curobo"}
        validators = {"contact_graspnet_live": _validate_graspnet_response,
                      "pyroki_live": _validate_joint_response, "curobo_live": _validate_joint_response}
        for name, url, payload in service_calls:
            if service_names[name] not in selected_services:
                continue
            try:
                response = requests.post(url, json=payload, timeout=180)
                body = response.json()
                validation = validators[name](body) if response.status_code == 200 else {}
                record({"api": name, "status": "passed" if response.status_code == 200 else "failed",
                        "http_status": response.status_code,
                        "response_keys": sorted(body) if isinstance(body, dict) else [], **validation})
            except Exception as exc:
                record({"api": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        # Exercise SAM3 and Molmo with the actual LIBERO RGB frame.  Port
        # availability alone is not conformance: each model must return a
        # real inference payload for this image.
        import base64 as _b64
        image_bytes = None
        if "sam3" in selected_services:
          try:
            image_bytes = api.get_observation()[api.camera_name]["images"]["rgb"]
            from PIL import Image
            _buf = io.BytesIO(); Image.fromarray(np.asarray(image_bytes)).save(_buf, format="PNG")
            image_payload = _b64.b64encode(_buf.getvalue()).decode("ascii")
            sam = requests.post(f"{SERVICE_URLS['sam3']}/segment", json={"image_base64": image_payload, "text_prompt": "black bowl"}, timeout=180)
            body = sam.json()
            validation = _validate_sam3_response(body) if sam.status_code == 200 else {}
            record({"api": "sam3_text_segmentation_live", "status": "passed" if sam.status_code == 200 else "failed", "http_status": sam.status_code, **validation})
          except Exception as exc:
            record({"api": "sam3_text_segmentation_live", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        else:
            image_bytes = obs[api.camera_name]["images"]["rgb"]
            from PIL import Image
            _buf = io.BytesIO(); Image.fromarray(np.asarray(image_bytes)).save(_buf, format="PNG")
            image_payload = _b64.b64encode(_buf.getvalue()).decode("ascii")
        if "molmo" in selected_services:
          try:
            prompt = f"Point at {args.molmo_target}"
            mol = requests.post(f"{SERVICE_URLS['molmo']}/v1/chat/completions", json={"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_payload}}]}], "max_tokens": 128, "temperature": 0.0, "stop": ["<|endoftext|>"]}, timeout=180)
            body = mol.json()
            output_text = str(body.get("choices", [{}])[0].get("message", {}).get("content", "")) if isinstance(body, dict) and body.get("choices") else ""
            point = _parse_molmo_point(output_text, width=np.asarray(image_bytes).shape[1], height=np.asarray(image_bytes).shape[0])
            expected_box = args.molmo_expected_box
            point_in_expected_box = None
            if point is not None and expected_box is not None:
                x1, y1, x2, y2 = expected_box
                point_in_expected_box = x1 <= point[0] <= x2 and y1 <= point[1] <= y2
            semantic_valid = point is not None and point_in_expected_box is not False
            record({"api": "molmo_vision_live", "status": "passed" if mol.status_code == 200 and semantic_valid else "failed", "http_status": mol.status_code, "prompt": prompt, "output_text_present": bool(output_text), "point_coordinate_parseable": point is not None, "pixel_point": point, "audited_expected_box_xyxy": expected_box, "point_inside_expected_box": point_in_expected_box, "output_preview": output_text[:256], "error": body.get("detail") if isinstance(body, dict) else None})
          except Exception as exc:
            record({"api": "molmo_vision_live", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        for name in selected_services:
            url = SERVICE_URLS[name]
            record({"api": f"service:{name}", "status": "available" if _port(url) else "unavailable", "url": url})
    finally:
        deployment.close()
    rows = list(rows_by_api.values())
    required = BASE_REQUIRED_APIS | set(SERVICE_PROBE_APIS.values()) | {f"service:{name}" for name in SERVICE_URLS}
    missing = sorted(required - set(rows_by_api))
    unavailable = [row["api"] for row in rows if row["status"] == "unavailable"]
    failed = [row["api"] for row in rows if row["status"] == "failed"]
    complete = not failed and not unavailable and not missing
    phases = list(prior_manifest.get("service_phases", [])) if prior_manifest else []
    phases.append({"services": sorted(selected_services), "started_unix": started,
                   "elapsed_seconds": time.time() - started})
    manifest = {"protocol": "roboforge-aspire-capx-libero-api-conformance-v2",
                "task": args.task, "state": args.state, "controller_mode": "JOINT_POSITION",
                "observation_fingerprint": observation_fingerprint,
                "upstream": {"aspire": {"commit": "f4c8939aab0af9b97690c561bd80e282940f7886"},
                             "cap-x": {"commit": "53e9966d7a8e2fa7494676772bccc35280f5c0ed"}},
                "complete_conformance": complete,
                "unavailable_services": unavailable,
                "failed_rows": failed,
                "missing_rows": missing,
                "service_phases": phases,
                "rows": rows, "elapsed_seconds": time.time() - started,
                "deployment_artifact_dir": str((args.output / "adapter").resolve())}
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    (args.output / "api-conformance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if failed:
        return 1
    if (unavailable or missing) and not args.allow_unavailable_services:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
