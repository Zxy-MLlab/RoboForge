"""Run the non-privileged ASPIRE/CaP-X Controller API against real LIBERO.

Service-backed calls are reported as unavailable when their upstream service is
not running; no mock response is substituted.  The resulting JSON is a light
weight invocation manifest suitable for the conformance matrix.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-unavailable-services",
        action="store_true",
        help="run only the LIBERO/geometry smoke; complete conformance is false",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    deployment = create(task=str(args.task), state=args.state, root=args.output,
                        configuration={"disable_agent_verifier": True,
                                       "controller_mode": "JOINT_POSITION"})
    api = FrankaLiberoApi(deployment)
    rows = []
    started = time.time()
    try:
        obs = api.get_observation()
        rows.append({"api": "get_observation", "status": "passed",
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
        rows.append({"api": "camera_pose_robot_base_frame", "status": "passed" if frame_passed else "failed",
                     "same_input_max_abs_error": frame_errors, "tolerance": 1e-12})
        rows.append({"api": "robot_state_shapes", "status": "passed"
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
                rows.append({"api": name, "status": "passed", "return_type": type(value).__name__})
            except Exception as exc:
                rows.append({"api": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        api.goto_home_joint_position(); rows.append({"api": "goto_home_joint_position", "status": "passed"})
        api.open_gripper(); rows.append({"api": "open_gripper", "status": "passed"})
        api.close_gripper(); rows.append({"api": "close_gripper", "status": "passed"})
        rows.append({"api": "task_language", "status": "passed", "value": api.task_language()})
        rows.append({"api": "supports_dual_arm", "status": "passed", "value": api.supports_dual_arm()})
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
        for name, url, payload in service_calls:
            try:
                response = requests.post(url, json=payload, timeout=180)
                body = response.json()
                rows.append({"api": name, "status": "passed" if response.status_code == 200 else "failed",
                             "http_status": response.status_code,
                             "response_keys": sorted(body) if isinstance(body, dict) else []})
            except Exception as exc:
                rows.append({"api": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        # Exercise SAM3 and Molmo with the actual LIBERO RGB frame.  Port
        # availability alone is not conformance: each model must return a
        # real inference payload for this image.
        import base64 as _b64
        image_bytes = None
        try:
            image_bytes = api.get_observation()[api.camera_name]["images"]["rgb"]
            from PIL import Image
            _buf = io.BytesIO(); Image.fromarray(np.asarray(image_bytes)).save(_buf, format="PNG")
            image_payload = _b64.b64encode(_buf.getvalue()).decode("ascii")
            sam = requests.post(f"{SERVICE_URLS['sam3']}/segment", json={"image_base64": image_payload, "text_prompt": "black bowl"}, timeout=180)
            body = sam.json()
            rows.append({"api": "sam3_text_segmentation_live", "status": "passed" if sam.status_code == 200 and isinstance(body, dict) and isinstance(body.get("results"), list) else "failed", "http_status": sam.status_code, "num_results": len(body.get("results", [])) if isinstance(body, dict) else 0})
        except Exception as exc:
            rows.append({"api": "sam3_text_segmentation_live", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        try:
            mol = requests.post(f"{SERVICE_URLS['molmo']}/v1/chat/completions", json={"messages": [{"role": "user", "content": [{"type": "text", "text": "Point at black bowl"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_payload}}]}], "max_tokens": 128, "temperature": 0.0, "stop": ["<|endoftext|>"]}, timeout=180)
            body = mol.json()
            output_text = str(body.get("choices", [{}])[0].get("message", {}).get("content", "")) if isinstance(body, dict) and body.get("choices") else ""
            point_parseable = bool(
                re.search(r"<points\s+coords\s*=", output_text, re.I)
                or re.search(r"<point\b[^>]*\bx\s*=.*\by\s*=", output_text, re.I)
                or re.search(r"<points\b[^>]*\bx\d+\s*=.*\by\d+\s*=", output_text, re.I)
            )
            rows.append({"api": "molmo_vision_live", "status": "passed" if mol.status_code == 200 and point_parseable else "failed", "http_status": mol.status_code, "output_text_present": bool(output_text), "point_coordinate_parseable": point_parseable, "output_preview": output_text[:256], "error": body.get("detail") if isinstance(body, dict) else None})
        except Exception as exc:
            rows.append({"api": "molmo_vision_live", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        for name, url in SERVICE_URLS.items():
            rows.append({"api": f"service:{name}", "status": "available" if _port(url) else "unavailable", "url": url})
    finally:
        deployment.close()
    unavailable = [row["api"] for row in rows if row["status"] == "unavailable"]
    failed = [row["api"] for row in rows if row["status"] == "failed"]
    complete = not failed and not unavailable
    manifest = {"protocol": "roboforge-aspire-capx-libero-api-conformance-v2",
                "task": args.task, "state": args.state, "controller_mode": "JOINT_POSITION",
                "upstream": {"aspire": {"commit": "f4c8939aab0af9b97690c561bd80e282940f7886"},
                             "cap-x": {"commit": "53e9966d7a8e2fa7494676772bccc35280f5c0ed"}},
                "complete_conformance": complete,
                "unavailable_services": unavailable,
                "failed_rows": failed,
                "rows": rows, "elapsed_seconds": time.time() - started,
                "deployment_artifact_dir": str((args.output / "adapter").resolve())}
    encoded = json.dumps(manifest, sort_keys=True, indent=2).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    (args.output / "api-conformance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if failed:
        return 1
    if unavailable and not args.allow_unavailable_services:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
