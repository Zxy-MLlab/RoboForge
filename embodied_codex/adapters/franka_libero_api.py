"""Controller-facing Franka/LIBERO API adapted from ASPIRE and CaP-X.

This module deliberately is a Python API, not an OpenHands tool set.  The
implementation keeps the upstream method names and call signatures while the
vision, grasp and IK backends remain the upstream HTTP services.  A deployment
may expose an instance to a controller through the ``sdk`` RPC operation; the
RPC layer only serializes the returned numpy values.

Sources (read-only vendored in the RoboForge archive):
ASPIRE ``f4c8939aab0af9b97690c561bd80e282940f7886`` (Apache-2.0/MIT) and
CaP-X ``53e9966d7a8e2fa7494676772bccc35280f5c0ed`` (MIT).
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import requests
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN


_TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)
SAM3_URL = os.environ.get("ROBOFORGE_SAM3_URL", "http://127.0.0.1:8114")
MOLMO_URL = os.environ.get("ROBOFORGE_MOLMO_URL", "http://127.0.0.1:8122/v1")
GRASPNET_URL = os.environ.get("ROBOFORGE_GRASPNET_URL", os.environ.get("GRASPNET_SERVICE_URL", "http://127.0.0.1:8115"))
PYROKI_URL = os.environ.get("ROBOFORGE_PYROKI_URL", "http://127.0.0.1:8116")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _post(url: str, payload: Mapping[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    response = requests.post(url, json=dict(payload), timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise RuntimeError(f"service returned non-object JSON: {url}")
    return result


def _post_with_retries(url: str, payload: Mapping[str, Any], *, timeout: float = 120.0,
                       max_retries: int = 5) -> dict[str, Any]:
    """The retry contract used by ASPIRE ``serve_utils.post_with_retries``."""
    import time
    deadline = time.time() + float(timeout)
    interval = 1.0; attempts = 0; last: Exception | None = None
    while time.time() < deadline and attempts < int(max_retries):
        try:
            return _post(url, payload, timeout=timeout)
        except Exception as exc:
            last = exc; attempts += 1
            if time.time() >= deadline or attempts >= int(max_retries): break
            time.sleep(min(interval, max(0.0, deadline - time.time())))
            interval = min(interval * 2.0, 8.0)
    raise RuntimeError(f"Request to {url} failed after {attempts} retries / {timeout:.2f}s. Last error: {last}")


def _encode_image(image: np.ndarray | Image.Image) -> str:
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO(); pil.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_array(value: str, shape: Sequence[int], dtype: str = "uint8") -> np.ndarray:
    raw = base64.b64decode(value)
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(tuple(shape))


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    subsample_factor: int = 1,
    depth_clip_range: tuple[float, float] = (0.015, 20.0),
    filter_invalid: bool = True,
) -> np.ndarray:
    """ASPIRE ``depth_to_pointcloud`` (camera-frame points).

    The optional arguments intentionally mirror ``aspire.sim.cap.utils.depth_utils``.
    ``filter_invalid=False`` preserves one point per pixel, which is required when
    applying an image mask after deprojection.
    """
    depth = np.asarray(depth)
    if depth.ndim == 3: depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("depth must be a HxW array")
    if np.asarray(intrinsics).shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {np.asarray(intrinsics).shape}")
    if int(subsample_factor) <= 0:
        raise ValueError("subsample_factor must be positive")
    factor = int(subsample_factor)
    h, w = depth.shape
    depth = depth[::factor, ::factor]
    K = np.asarray(intrinsics, dtype=np.float64).copy()
    K[0, :3] /= factor; K[1, :3] /= factor
    hs, ws = depth.shape
    yy, xx = np.mgrid[:hs, :ws]
    z = depth.astype(np.float64)
    pts = np.stack(((xx - K[0, 2]) * z / K[0, 0],
                    (yy - K[1, 2]) * z / K[1, 1], z), axis=-1)
    if not filter_invalid:
        return pts
    near, far = depth_clip_range
    valid = (~np.isnan(pts).any(axis=-1) & ~np.isinf(pts).any(axis=-1)
             & (pts[..., 2] >= float(near)) & (pts[..., 2] <= float(far)))
    return pts[valid]


def depth_to_point_cloud(depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """ASPIRE skill-library organized point cloud (H, W, 3), no filtering."""
    depth = np.asarray(depth_img)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"depth_img must be 2D, got {depth.shape}")
    K = np.asarray(intrinsics)
    if K.shape != (3, 3):
        raise ValueError(f"intrinsics must be (3, 3), got {K.shape}")
    h, w = depth.shape
    y, x = np.mgrid[0:h, 0:w]
    z = depth
    return np.dstack(((x - K[0, 2]) * z / K[0, 0],
                      (y - K[1, 2]) * z / K[1, 1], z))


def mask_to_world_points(mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray,
                         extrinsics: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask); depth = np.asarray(depth)
    if mask.ndim == 3: mask = mask[..., 0]
    if depth.ndim == 3: depth = depth[..., 0]
    if mask.shape != depth.shape:
        raise ValueError("mask and depth shapes differ")
    ys, xs = np.where(mask > 0); z = depth[ys, xs]
    valid = z > 0
    ys, xs, z = ys[valid], xs[valid], z[valid]
    cam = np.c_[(xs - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (ys - intrinsics[1, 2]) * z / intrinsics[1, 1], z,
                np.ones(len(z))]
    return (np.asarray(extrinsics) @ cam.T).T[:, :3]


def pixel_to_world_point(u: int, v: int, z: float, intrinsics: np.ndarray,
                         extrinsics: np.ndarray) -> np.ndarray:
    p = np.array([(u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                  (v - intrinsics[1, 2]) * z / intrinsics[1, 1], z, 1.0])
    return (np.asarray(extrinsics) @ p)[:3]


def decompose_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError("T must be 4x4")
    return T[:3, 3].copy(), _rotation_matrix_to_wxyz(T[:3, :3])


def _rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd branch used by ASPIRE's skill library (WXYZ order)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("R must be 3x3")
    tr = float(np.trace(R))
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S; x = (R[2, 1] - R[1, 2]) / S; y = (R[0, 2] - R[2, 0]) / S; z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S; x = 0.25 * S; y = (R[0, 1] + R[1, 0]) / S; z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S; x = (R[0, 1] + R[1, 0]) / S; y = 0.25 * S; z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S; x = (R[0, 2] + R[2, 0]) / S; y = (R[1, 2] + R[2, 1]) / S; z = 0.25 * S
    return np.array([w, x, y, z], dtype=np.float64)


def _obb(points: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("points must be an Nx3 array with at least three points")
    import open3d as o3d
    # Preserve the upstream degeneracy handling and Open3D exception semantics;
    # silently swapping in a PCA box would change orientations and failure modes.
    noisy = points + np.random.normal(0, 0.0001, points.shape)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(noisy))
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    box = cloud.get_oriented_bounding_box()
    center = np.array(box.center, dtype=np.float64, copy=True)
    extent = np.array(box.extent, dtype=np.float64, copy=True)
    # Open3D exposes ``OrientedBoundingBox.R`` as a read-only Fortran-order
    # pybind view.  SciPy's Rotation Cython buffer requires a writable array
    # in the LIBERO Python 3.11 environment and otherwise raises
    # ``ValueError: buffer source array is read-only``.  Materializing the
    # same matrix as an owned C-order array preserves the upstream geometry
    # while making the library boundary explicit.
    R = np.array(box.R, dtype=np.float64, copy=True, order="C")
    q = Rotation.from_matrix(R).as_quat(); wxyz = np.array([q[3], q[0], q[1], q[2]])
    return {"center": center, "extent": extent, "R": R, "quaternion_wxyz": wxyz}


class FrankaLiberoApi:
    """ASPIRE/CaP-X compatible non-privileged Franka API.

    ``env`` is a LIBERO adapter exposing ``get_observation`` (or RoboForge
    deployment exposing ``_observe``), ``move_to_joints_blocking`` and
    ``_step_once``.  No simulator internals are read by this class.
    """
    _TCP_OFFSET = _TCP_OFFSET
    _DUAL_TCP_OFFSET = np.array([0.0, 0.0, -0.107], dtype=np.float64)
    upstream = {
        "aspire": {"repository": "https://github.com/NVlabs/ASPIRE", "commit": "f4c8939aab0af9b97690c561bd80e282940f7886"},
        "cap-x": {"repository": "https://github.com/capgym/cap-x", "commit": "53e9966d7a8e2fa7494676772bccc35280f5c0ed"},
    }

    def __init__(self, env: Any, use_sam3: bool = True) -> None:
        self._env = env; self.use_sam3 = bool(use_sam3)
        self.camera_name = "agentview"; self.wrist_camera_name = "robot0_eye_in_hand"
        self.cfg = None; self._curobo_world_config = None

    def functions(self) -> dict[str, Callable[..., Any]]:
        names = (
            "get_observation", "segment_sam3_point_prompt", "segment_sam3_text_prompt",
            "point_prompt_molmo", "get_oriented_bounding_box_from_3d_points", "goto_pose",
            "open_gripper", "close_gripper", "get_object_pose", "sample_grasp_pose",
            "get_object_3d_points_and_masks_from_language", "goto_home_joint_position",
            "subsample_point_cloud", "filter_noise", "plan_grasp", "plan_grasp_from_point_clouds",
            "parse_grasp_poses_for_curobo", "create_curobo_world_from_depth",
            "create_curobo_world_from_pointcloud", "create_curobo_world_from_observation",
            "update_curobo_world", "plan_grasp_trajectory", "execute_joint_trajectory",
            "update_curobo_world_with_object", "plan_with_grasped_object",
            "mask_to_world_points", "pixel_to_world_point", "decompose_transform",
            "depth_to_pointcloud", "depth_to_point_cloud", "select_top_down_grasp", "solve_ik", "move_to_joints",
            "traj_plan", "move_along_trajectory", "move_to_joints_both",
            "rotation_matrix_to_quaternion", "transform_points", "interpolate_segment", "normalize_vector",
            "task_language", "supports_dual_arm", "move_to_joints_arm0", "move_to_joints_arm1",
            "get_arm0_gripper_pose", "get_arm1_gripper_pose",
            "open_gripper_arm0", "close_gripper_arm0", "open_gripper_arm1", "close_gripper_arm1",
            "solve_ik_arm0", "solve_ik_arm1",
            "goto_pose_arm0", "goto_pose_arm1", "goto_pose_both",
        )
        return {name: getattr(self, name) for name in names if hasattr(self, name)}

    def _native_observation(self) -> Mapping[str, Any]:
        if hasattr(self._env, "get_franka_libero_observation"):
            return self._env.get_franka_libero_observation()
        if hasattr(self._env, "obs") and self._env.obs is not None:
            return self._env.obs
        if hasattr(self._env, "get_observation"):
            return self._env.get_observation()
        if hasattr(self._env, "_observe"):
            # RoboForge deployment's public observation projection retains the
            # same camera/world data as the ASPIRE adapter.
            return self._env._observe("rgbd", {})
        if hasattr(self._env, "obs") and self._env.obs is not None:
            return self._env.obs
        raise RuntimeError("LIBERO environment does not expose public observation")

    def get_observation(self) -> dict[str, Any]:
        raw = self._native_observation()
        if (isinstance(raw, Mapping) and self.camera_name in raw
                and isinstance(raw[self.camera_name], Mapping)
                and "images" in raw[self.camera_name]):
            result = dict(raw)
            for name in (self.camera_name, self.wrist_camera_name):
                camera = dict(result[name]); images = dict(camera["images"])
                images["depth"] = np.asarray(images["depth"]).squeeze()
                camera["images"] = images; result[name] = camera
            return result
        if isinstance(raw, Mapping) and "agentview_image" in raw:
            from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix, get_real_depth_map
            result = {}
            for name in (self.camera_name, self.wrist_camera_name):
                rgb = np.ascontiguousarray(raw[f"{name}_image"][::-1])
                depth = np.asarray(raw[f"{name}_depth"][::-1]).squeeze()
                sim = getattr(self._env, "sim", None)
                if sim is None and hasattr(self._env, "env"):
                    sim = getattr(self._env.env, "sim", None)
                if sim is not None:
                    depth = get_real_depth_map(sim, depth)
                    K = get_camera_intrinsic_matrix(sim, name, rgb.shape[1], rgb.shape[0])
                    pose = get_camera_extrinsic_matrix(sim, name)
                    # ASPIRE exposes camera poses in the robot base frame, not
                    # the global MuJoCo frame.  Keep the robosuite camera-axis
                    # correction from get_camera_extrinsic_matrix, then apply
                    # the inverse robot0_base transform exactly as upstream.
                    try:
                        base_id = sim.model.body_name2id("robot0_base")
                        base = np.eye(4)
                        base[:3, :3] = np.asarray(sim.data.xmat[base_id]).reshape(3, 3)
                        base[:3, 3] = np.asarray(sim.data.xpos[base_id])
                        pose = np.linalg.inv(base) @ pose
                    except Exception:
                        pass
                else:
                    K = np.eye(3); pose = np.eye(4)
                result[name] = {"images": {"rgb": rgb, "depth": depth}, "intrinsics": K, "pose_mat": pose}
            result["robot_joint_pos"] = np.asarray(raw.get("robot0_joint_pos", []))
            eef_pos = np.asarray(raw.get("robot0_eef_pos", [0, 0, 0]), dtype=np.float64)
            eef_xyzw = np.asarray(raw.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=np.float64)
            eef_wxyz = np.array([eef_xyzw[3], eef_xyzw[0], eef_xyzw[1], eef_xyzw[2]])
            if sim is not None:
                try:
                    base_id = sim.model.body_name2id("robot0_base")
                    base = np.eye(4); base[:3, :3] = np.asarray(sim.data.xmat[base_id]).reshape(3, 3); base[:3, 3] = np.asarray(sim.data.xpos[base_id])
                    p = np.r_[eef_pos, 1.0]; eef_pos = (np.linalg.inv(base) @ p)[:3]
                    eef_wxyz = self.rotation_matrix_to_quaternion(np.linalg.inv(base[:3, :3]) @ Rotation.from_quat(eef_xyzw).as_matrix())
                except Exception:
                    pass
            result["robot_cartesian_pos"] = np.r_[eef_pos, eef_wxyz,
                                                    np.asarray(raw.get("robot0_gripper_qpos", [0])).reshape(-1)[:1]]
            return result
        # A RoboForge deployment stores controller-visible sensors as files;
        # materialize the exact ASPIRE dictionary shape for Python callers.
        if isinstance(raw, Mapping) and isinstance(raw.get("cameras"), Mapping):
            import cv2
            out = {}
            for name, camera in raw["cameras"].items():
                rgb = cv2.cvtColor(cv2.imread(str(camera["rgb_path"]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                depth = np.load(str(camera["depth_path"]), allow_pickle=False) if camera.get("depth_path") else None
                out[str(name)] = {"images": {"rgb": rgb, "depth": depth},
                                  "intrinsics": np.asarray(camera["intrinsic"], float),
                                  "pose_mat": np.asarray(camera["camera_to_world"], float)}
            proprio = raw.get("proprioception") or {}
            eef = proprio.get("eef_pose") or {}; joints = proprio.get("joint_state") or {}
            q = list(eef.get("orientation_xyzw", [0,0,0,1])); wxyz = [q[3], q[0], q[1], q[2]]
            out["robot_cartesian_pos"] = np.r_[eef.get("position_m", [0,0,0]), wxyz,
                                                (proprio.get("gripper") or {}).get("width_m", 0.0)]
            out["robot_joint_pos"] = np.asarray(joints.get("position", []), float)
            out["proprioception"] = proprio
            return out
        return dict(raw)

    def task_language(self) -> str:
        return str(getattr(self._env, "instruction", getattr(self._env, "_instruction", "")))

    def segment_sam3_point_prompt(self, rgb: np.ndarray, point_coords: tuple[float, float]) -> list[dict[str, Any]]:
        try:
            data = _post_with_retries(f"{SAM3_URL.rstrip('/')}/segment_point", {
                "image_base64": _encode_image(rgb), "point_coords": point_coords})
        except RuntimeError as exc:
            raise RuntimeError(f"Failed to communicate with SAM3 service at {SAM3_URL}: {exc}") from exc
        shape = tuple(data.get("masks_shape", (0, 0, 0)))
        masks = _decode_array(data.get("masks_base64", ""), shape, data.get("masks_dtype", "float32")).astype(bool)
        return [{"mask": mask, "score": score} for mask, score in zip(masks, data.get("scores", []))]

    def segment_sam3_text_prompt(self, rgb: np.ndarray, text_prompt: str) -> list[dict[str, Any]]:
        try:
            data = _post_with_retries(f"{SAM3_URL.rstrip('/')}/segment", {
                "image_base64": _encode_image(rgb), "text_prompt": text_prompt})
        except Exception:
            # ASPIRE's ``init_sam3`` client treats a failed text request as an
            # empty detection list; callers may then use the Molmo/point path.
            return []
        result = []
        for item in data.get("results", []):
            result.append({"mask": _decode_array(item["mask_base64"], tuple(item["shape"])).astype(bool),
                           "box": item["box"], "score": item["score"], "label": item.get("label")})
        return result

    def point_prompt_molmo(self, image: np.ndarray, text_prompt: str) -> dict[str, tuple[int | None, int | None]]:
        # Keep the upstream Molmo prompt and parsing contract, with a small
        # parser that accepts Molmo1/Molmo2 XML and normalized coordinates.
        payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": f"Point at {text_prompt}"},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"}}]}],
                   "max_tokens": 1024, "temperature": 0.0, "stop": ["<|endoftext|>"]}
        # A stock vLLM server may expose a local path or another served-model
        # alias rather than the upstream Hugging Face repository name.  When
        # no explicit alias is configured, omit ``model`` just like the live
        # conformance probe so vLLM selects its sole served model.
        model = os.environ.get("ROBOFORGE_MOLMO_MODEL")
        if model:
            payload["model"] = model
        try:
            data = _post_with_retries(f"{MOLMO_URL.rstrip('/')}/chat/completions", payload,
                                       max_retries=3)
        except RuntimeError:
            data = {}
        text = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        import re
        points: list[tuple[float, float]] = []
        coords = re.search(r'<points\s+coords\s*=\s*["\']([^"\']+)["\']', text, re.I)
        scale = 1000.0
        if coords:
            nums = [float(x) for x in coords.group(1).split()]
            i = 1
            while i + 2 < len(nums):
                points.append((nums[i + 1], nums[i + 2])); i += 3
        else:
            scale = 100.0
            for tag in re.findall(r"<point\b[^>]*>", text, re.I):
                mx = re.search(r"\bx\s*=\s*['\"]([0-9]*\.?[0-9]+)", tag, re.I)
                my = re.search(r"\by\s*=\s*['\"]([0-9]*\.?[0-9]+)", tag, re.I)
                if mx and my and 0 <= float(mx.group(1)) <= 100 and 0 <= float(my.group(1)) <= 100:
                    points.append((float(mx.group(1)), float(my.group(1))))
            if not points:
                tag = re.search(r"<points\b[^>]*>", text, re.I)
                if tag:
                    xs = {int(i): float(v) for i, v in re.findall(r"x(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)", tag.group(0))}
                    ys = {int(i): float(v) for i, v in re.findall(r"y(\d+)\s*=\s*['\"]([0-9]*\.?[0-9]+)", tag.group(0))}
                    points = [(xs[i], ys[i]) for i in sorted(set(xs) & set(ys)) if 0 <= xs[i] <= 100 and 0 <= ys[i] <= 100]
            if not points:
                points = [(float(x), float(y)) for x, y in re.findall(r"([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)", text)
                           if 0 <= float(x) <= 100 and 0 <= float(y) <= 100]
        if not points: return {text_prompt: (None, None)}
        w, h = Image.fromarray(np.asarray(image)).size
        x, y = points[0]
        return {text_prompt: (int(x / scale * w), int(y / scale * h))}

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        return _obb(points)

    def mask_to_world_points(self, mask, depth, intrinsics, extrinsics): return mask_to_world_points(mask, depth, intrinsics, extrinsics)
    def pixel_to_world_point(self, u, v, z, intrinsics, extrinsics): return pixel_to_world_point(u, v, z, intrinsics, extrinsics)
    def decompose_transform(self, T): return decompose_transform(T)
    def depth_to_pointcloud(self, depth_img: np.ndarray, intrinsics: np.ndarray,
                            subsample_factor: int = 1,
                            depth_clip_range: tuple[float, float] = (0.015, 20.0),
                            filter_invalid: bool = True) -> np.ndarray:
        return depth_to_pointcloud(depth_img, intrinsics, subsample_factor,
                                   depth_clip_range, filter_invalid)
    def depth_to_point_cloud(self, depth_img: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        return depth_to_point_cloud(depth_img, intrinsics)

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Return an ASPIRE-compatible WXYZ quaternion from a 3x3 rotation."""
        r = np.asarray(R, dtype=float)
        if r.shape != (3, 3): raise ValueError("R must be 3x3")
        return _rotation_matrix_to_wxyz(r)

    def transform_points(self, points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
        original_shape = np.asarray(points).shape
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        h = np.c_[p, np.ones(len(p))]
        return ((np.asarray(transform_matrix) @ h.T).T[:, :3]).reshape(original_shape)

    def interpolate_segment(self, p1: np.ndarray, p2: np.ndarray, step: float = 0.03) -> list[np.ndarray]:
        a, b = np.asarray(p1, float), np.asarray(p2, float)
        distance = float(np.linalg.norm(b - a))
        if distance < 1e-6:
            return [a]
        if float(step) <= 0:
            raise ValueError("step must be positive")
        count = int(np.ceil(distance / float(step)))
        return [a + (b - a) * t for t in np.linspace(0.0, 1.0, count + 1)]

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        value = np.asarray(v, dtype=float)
        norm = float(np.linalg.norm(value))
        if norm < 1e-6: return value
        return value / norm

    def _camera(self, camera_name: str) -> Mapping[str, Any]:
        obs = self._native_observation()
        if isinstance(obs, Mapping) and f"{camera_name}_image" in obs:
            cam = self.get_observation()[camera_name]
            return {"rgb": cam["images"]["rgb"], "depth": cam["images"]["depth"],
                    "intrinsics": cam["intrinsics"], "pose_mat": cam["pose_mat"]}
        cam = obs[camera_name]
        images = cam.get("images", cam)
        return {"rgb": images.get("rgb"), "depth": np.asarray(images.get("depth")).squeeze(),
                "intrinsics": np.asarray(cam.get("intrinsics", cam.get("intrinsic"))),
                "pose_mat": np.asarray(cam.get("pose_mat", cam.get("pose")))}

    def get_object_3d_points_and_masks_from_language(self, text_prompt: str, use_multiview: bool = True) -> dict[str, Any]:
        camera_data = {}
        cameras = [self.camera_name] + ([self.wrist_camera_name] if use_multiview else [])
        for name in cameras:
            cam = self._camera(name); point = self.point_prompt_molmo(cam["rgb"], text_prompt).get(text_prompt)
            masks = self.segment_sam3_point_prompt(cam["rgb"], point) if point and point[0] is not None else []
            if not masks: masks = self.segment_sam3_text_prompt(cam["rgb"], text_prompt)
            if not masks: raise ValueError(f"SAM3 segmentation failed for '{text_prompt}' on {name}")
            best = max(masks, key=lambda x: x["score"])
            depth = np.asarray(cam["depth"])
            if depth.ndim == 3: depth = depth[..., 0]
            # Preserve the image-pixel correspondence used by ASPIRE before
            # applying the segmentation mask (filtered point clouds cannot be
            # indexed by a flattened mask).
            camera_points = depth_to_pointcloud(depth, cam["intrinsics"], filter_invalid=False)
            valid = np.asarray(best["mask"], dtype=bool) & np.isfinite(depth) & (depth > 0)
            world_h = np.c_[camera_points[valid], np.ones(int(valid.sum()))]
            pts = (np.asarray(cam["pose_mat"]) @ world_h.T).T[:, :3]
            camera_data[name] = {"mask": best["mask"], "score": best["score"], "points_3d": pts}
        agent = camera_data[self.camera_name]; wrist = camera_data.get(self.wrist_camera_name)
        agent_pts = agent["points_3d"]
        if wrist is None:
            pts = agent_pts; wrist_pts = None
        else:
            wrist_pts = wrist["points_3d"]
            if len(wrist_pts) > 0 and len(agent_pts) > 0:
                distances = np.linalg.norm(agent_pts[:, None, :] - wrist_pts[None, :, :], axis=2)
                min_distances = np.min(distances, axis=1)
                if float(np.min(min_distances)) < 0.01:
                    pts = np.concatenate([agent_pts, wrist_pts])
                elif wrist["score"] > agent["score"]:
                    pts = wrist_pts
                else:
                    pts = agent_pts
            elif len(wrist_pts) > 0:
                pts = wrist_pts
            else:
                pts = agent_pts
        return {"agentview_mask": agent["mask"], "wrist_mask": None if wrist is None else wrist["mask"],
                "points_3d": pts, "agentview_points_3d": agent["points_3d"],
                "wrist_points_3d": wrist_pts,
                "agentview_score": agent["score"], "wrist_score": None if wrist is None else wrist["score"]}

    def filter_noise(self, points, colors=None):
        points = np.asarray(points); labels = DBSCAN(eps=.005, min_samples=10).fit_predict(points)
        keep = labels != -1; return points[keep], None if colors is None else np.asarray(colors)[keep]

    def subsample_point_cloud(self, pc, max_points=10000):
        pc = np.asarray(pc); return pc if len(pc) <= max_points else pc[np.random.choice(len(pc), max_points, replace=False)]

    def get_object_pose(self, object_name: str, use_multiview: bool = True):
        result = self.get_object_3d_points_and_masks_from_language(object_name, use_multiview)
        points, _ = self.filter_noise(result["points_3d"])
        if len(points) == 0: return None, None
        box = _obb(points); R = np.asarray(box["R"])
        if R[2, 2] > 0: R = R @ np.diag([-1, 1, -1])
        q = _rotation_matrix_to_wxyz(R); return np.asarray(box["center"]), q

    def plan_grasp_from_point_clouds(self, pc_full, pc_segment):
        import io as _io
        def enc(a):
            b = _io.BytesIO(); np.save(b, np.asarray(a)); return base64.b64encode(b.getvalue()).decode("ascii")
        data = _post(f"{GRASPNET_URL.rstrip('/')}/plan_point_clouds", {
            "pc_full_base64": enc(pc_full), "pc_segment_base64": enc(pc_segment),
            "segmap_id": 1, "local_regions": True, "filter_grasps": True,
            "forward_passes": 2, "max_retries": 10})
        grasps = np.load(_io.BytesIO(base64.b64decode(data["grasps_base64"])))
        scores = np.load(_io.BytesIO(base64.b64decode(data["scores_base64"])))
        if len(grasps) == 0:
            raise AssertionError("No grasp candidates found")
        offset = np.eye(4); offset[:3, 3] = [0.0, 0.0, 0.12]
        return np.einsum("nij,jk->nik", grasps, offset), scores

    def plan_grasp(self, depth: np.ndarray, intrinsics: np.ndarray, segmentation: np.ndarray):
        """ASPIRE reduced API: plan grasps from depth, intrinsics and mask."""
        depth = np.asarray(depth)
        if depth.ndim == 3: depth = depth[..., 0]
        segmentation = np.asarray(segmentation)
        if segmentation.ndim == 3: segmentation = segmentation[..., 0]
        if depth.shape != segmentation.shape: raise ValueError("depth and segmentation shapes differ")
        def enc(value):
            buf = io.BytesIO(); np.save(buf, value); return base64.b64encode(buf.getvalue()).decode("ascii")
        data = _post(f"{GRASPNET_URL.rstrip('/')}/plan", {
            "depth_base64": enc(depth), "cam_K_base64": enc(np.asarray(intrinsics)),
            "segmap_base64": enc(segmentation), "segmap_id": 1,
            "local_regions": True, "filter_grasps": True,
            "skip_border_objects": False, "z_range": [0.2, 2.0],
            "forward_passes": 2, "max_retries": 10,
        })
        grasps = np.load(io.BytesIO(base64.b64decode(data["grasps_base64"])))
        scores = np.load(io.BytesIO(base64.b64decode(data["scores_base64"])))
        # ASPIRE adds a 12cm tool-frame translation after Contact-GraspNet.
        offset = np.eye(4); offset[:3, 3] = [0, 0, 0.12]
        return np.einsum("nij,jk->nik", grasps, offset), scores

    def sample_grasp_pose(self, object_name: str, use_multiview: bool = True):
        result = self.get_object_3d_points_and_masks_from_language(object_name, use_multiview); seg, _ = self.filter_noise(result["points_3d"])
        obs = self.get_observation(); full = []
        for name in (self.camera_name, self.wrist_camera_name):
            cam = self._camera(name); p = depth_to_pointcloud(cam["depth"], cam["intrinsics"]); h = np.c_[p, np.ones(len(p))]; full.append((cam["pose_mat"] @ h.T).T[:, :3])
        poses, scores = self.plan_grasp_from_point_clouds(np.concatenate(full), seg)
        i = int(np.argmax(scores)); T = np.asarray(poses[i])
        rz = Rotation.from_euler("z", np.pi / 2.0).as_matrix()
        post = np.eye(4); post[:3, :3] = rz
        T = T @ post
        q = _rotation_matrix_to_wxyz(T[:3, :3])
        return T[:3, 3], q

    def solve_ik(self, position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve panda_hand IK using the upstream reduced LIBERO API.

        The public target is the gripper TCP; the PyRoKi service solves the
        ``panda_hand`` frame, so the ASPIRE TCP offset is applied before the
        request.  Returned values are the seven arm joints (the service may
        internally return an eighth gripper entry, which is discarded by the
        upstream adapter).
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        # The reduced upstream API clamps targets to its documented reachable
        # LIBERO workspace before applying the TCP offset.  Keep this guard in
        # the controller-facing implementation so out-of-range requests have
        # identical solver inputs and do not turn into service timeouts.
        pos = np.clip(pos, [-0.1, -0.5, 0.005], [0.75, 0.5, 0.9])
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        offset_pos = pos + rot.apply(self._TCP_OFFSET)
        # ASPIRE's non-real LIBERO adapter uses the shared convergence helper:
        # five warm-started calls, stopping early once the configuration is
        # stable to 1e-3.  Keep this behavior rather than reducing IK to a
        # single request (which changes both convergence and service load).
        result = self.cfg
        previous = self.cfg
        for _ in range(5):
            result = self._solve_ik_with_prev(offset_pos, quat, previous)
            if previous is not None and np.allclose(result, previous, atol=1e-3):
                break
            previous = result
        self.cfg = result
        return np.asarray(result, dtype=np.float64).reshape(-1)[:7]

    def _solve_ik_with_prev(self, position: np.ndarray, quaternion_wxyz: np.ndarray, prev_cfg: np.ndarray | None = None) -> np.ndarray:
        pos = np.asarray(position, dtype=float).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
        target = np.r_[quat, pos]
        data = _post(f"{PYROKI_URL.rstrip('/')}/ik", {"target_pose_wxyz_xyz": _jsonable(target), "prev_cfg": _jsonable(prev_cfg)})
        return np.asarray(data["joint_positions"], dtype=np.float64)

    def move_to_joints(self, joints: np.ndarray) -> None:
        env = self._env
        if not hasattr(env, "move_to_joints_blocking"): raise RuntimeError("blocking joint control unavailable")
        env.move_to_joints_blocking(np.asarray(joints, dtype=float).reshape(7))

    def goto_pose(self, position: np.ndarray, quaternion_wxyz: np.ndarray, z_approach: float = 0.0) -> None:
        pos = np.asarray(position, float).reshape(3); q = np.asarray(quaternion_wxyz, float).reshape(4)
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]]); target = pos + rot.apply(self._TCP_OFFSET)
        if z_approach: self._goto_pose_once(target + rot.apply([0, 0, -z_approach]), q)
        self._goto_pose_once(target, q)

    def _goto_pose_once(self, pos, q):
        joints = self._solve_ik_with_prev(pos, q, self.cfg); self.cfg = joints
        self.move_to_joints(joints)

    def open_gripper(self) -> None:
        if hasattr(self._env, "_set_gripper"):
            self._env._set_gripper(1.0)
            for _ in range(40): self._env._step_once()
        else: self._dispatch_act({"type": "gripper", "command": "open", "repeat": 40})

    def close_gripper(self) -> None:
        if hasattr(self._env, "_set_gripper"):
            self._env._set_gripper(0.0)
            for _ in range(60): self._env._step_once()
        else: self._dispatch_act({"type": "gripper", "command": "close", "repeat": 60})

    def goto_home_joint_position(self) -> None:
        home = getattr(self._env, "home_joint_position", None)
        if home is None: raise RuntimeError("Home joint position is unavailable in the current environment.")
        self.move_to_joints(home)

    def traj_plan(self, start_pose_wxyz_xyz: np.ndarray,
                  end_pose_wxyz_xyz: np.ndarray) -> np.ndarray:
        """Plan a joint-space trajectory using the upstream PyRoKi endpoint.

        This preserves ASPIRE/CaP-X ``traj_plan``'s pose-vector signature.  A
        PyRoKi server may return a gripper column; the upstream control API
        strips that column before returning arm waypoints.
        """
        start = np.asarray(start_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        end = np.asarray(end_pose_wxyz_xyz, dtype=np.float64).reshape(7)
        data = _post(f"{PYROKI_URL.rstrip('/')}/plan", {
            "start_pose_wxyz_xyz": start.tolist(),
            "end_pose_wxyz_xyz": end.tolist(),
        })
        waypoints = np.asarray(data.get("waypoints"), dtype=np.float64)
        if waypoints.ndim != 2 or waypoints.shape[1] < 7:
            raise RuntimeError(f"PyRoKi returned invalid trajectory shape {waypoints.shape}")
        return waypoints[:, :7]

    def move_along_trajectory(self, trajectory: np.ndarray) -> None:
        """Execute each trajectory waypoint with blocking joint control."""
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 7:
            raise ValueError(f"trajectory must be (N, 7), got shape {traj.shape}")
        for waypoint in traj:
            self._env.move_to_joints_blocking(waypoint[:7], tolerance=0.025, max_steps=15)

    def move_to_joints_both(self, joints0: np.ndarray, joints1: np.ndarray) -> None:
        """Move both arms simultaneously when the provider supports it."""
        fn = getattr(self._env, "move_to_joints_blocking_both", None)
        if fn is None:
            raise RuntimeError("dual-arm simultaneous control unavailable")
        fn(np.asarray(joints0, dtype=np.float64).reshape(7),
           np.asarray(joints1, dtype=np.float64).reshape(7))

    def _dispatch_act(self, action):
        if hasattr(self._env, "dispatch"): return self._env.dispatch("act", {"action": action})
        raise RuntimeError("environment does not provide action dispatch")

    def select_top_down_grasp(self, grasps: np.ndarray, scores: np.ndarray, cam_to_world: np.ndarray, vertical_threshold: float = 0.8) -> tuple:
        """Select the highest scoring grasp whose approach is vertical enough.

        The reduced upstream helper accepts ``(N,4,4)`` transforms, scores and
        a camera-to-world transform and returns ``(position, quaternion_wxyz)``.
        ``(N,3,3)`` rotations are accepted as a compatibility convenience.
        """
        poses = np.asarray(grasps, dtype=float); scores = np.asarray(scores, dtype=float)
        if poses.ndim == 3 and poses.shape[-2:] == (3, 3):
            homogeneous = np.tile(np.eye(4), (len(poses), 1, 1)); homogeneous[:, :3, :3] = poses; poses = homogeneous
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) != len(scores):
            raise ValueError("grasps must be (N,4,4) and scores must be (N,)")
        poses = np.einsum("ij,njk->nik", np.asarray(cam_to_world, float), poses)
        if not len(poses): return None, None
        vertical = -poses[:, 2, 2]
        valid = np.flatnonzero(vertical > float(vertical_threshold))
        if not len(valid): return None, -np.float64("inf")
        i = int(valid[np.argmax(scores[valid])])
        return poses[i], float(scores[i])

    def parse_grasp_poses_for_curobo(self, grasp_poses_world, grasp_scores, top_k=15):
        poses = np.asarray(grasp_poses_world); scores = np.asarray(grasp_scores); order = np.argsort(-scores)[:min(top_k, len(scores))]
        qs = []
        for T in poses[order]:
            q = Rotation.from_matrix(T[:3, :3]).as_quat(); qs.append([q[3], q[0], q[1], q[2]])
        return poses[order, :3, 3], np.asarray(qs), scores[order]

    def _upstream_curobo(self):
        """Load the actual ASPIRE cuRobo implementation lazily.

        CuRobo returns native ``WorldConfig``/trajectory objects and is not a
        JSON HTTP API in ASPIRE.  We therefore delegate to the archived source
        when its optional dependencies are installed; otherwise the public
        method fails explicitly instead of sending invented endpoints.
        """
        try:
            from aspire.sim.cap.integrations.motion import curobo_api
            return curobo_api
        except Exception as exc:
            raise RuntimeError("CuRobo provider unavailable; install ASPIRE CuRobo dependencies") from exc

    def create_curobo_world_from_depth(self, depth_image, object_mask, intrinsics, camera_pose=None, **kwargs):
        world = self._upstream_curobo().create_curobo_world_from_depth(
            depth_image, object_mask, intrinsics, camera_pose=camera_pose, **kwargs)
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_pointcloud(self, point_cloud, object_mask, **kwargs):
        world = self._upstream_curobo().create_curobo_world_from_pointcloud(point_cloud, object_mask, **kwargs)
        self._curobo_world_config = world
        return world

    def create_curobo_world_from_observation(self, object_mask, *, camera_name=None,
                                             object_name="object", scene_name="scene", **kwargs):
        c = self._camera(camera_name or self.camera_name)
        world = self.create_curobo_world_from_depth(c["depth"], object_mask, c["intrinsics"],
                                                    camera_pose=c["pose_mat"], object_name=object_name,
                                                    scene_name=scene_name, **kwargs)
        return world

    def update_curobo_world(self, *, camera_name=None, robot_distance_threshold=0.15,
                            robot_file="franka.yml", **kwargs):
        c = self._camera(camera_name or self.camera_name)
        obs = self.get_observation()
        pose = np.asarray(c["pose_mat"])
        robot_joints = np.asarray(obs["robot_joint_pos"], dtype=np.float64)
        world = self._upstream_curobo().create_curobo_world_from_depth_full(
            c["depth"], c["intrinsics"], camera_pose=pose,
            robot_joint_position=robot_joints, robot_file=robot_file,
            robot_distance_threshold=robot_distance_threshold, **kwargs)
        self._curobo_world_config = world
        return world

    def plan_grasp_trajectory(self, object_name, *, object_mask, grasp_poses,
                              top_k_grasps=15, use_world_collision=True,
                              robot_distance_threshold=0.15,
                              robot_collision_sphere_buffer=-0.01,
                              collision_activation_distance=0.001,
                              world_config=None, **kwargs):
        if world_config is None:
            world_config = self.update_curobo_world_with_object(
                object_name, object_mask=object_mask,
                robot_distance_threshold=robot_distance_threshold, **kwargs)
            ignore = [getattr(self, "_curobo_world_object_name", object_name.replace(" ", "_"))]
        else:
            ignore = []
        obs = self.get_observation()
        poses = [(np.asarray(p), np.asarray(q)) for p, q in grasp_poses]
        return self._upstream_curobo().plan_to_grasp_poses(
            world_config, np.asarray(obs["robot_joint_pos"]), poses,
            use_world_collision=use_world_collision,
            robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance,
            ignore_obstacle_names=ignore if use_world_collision else None, **kwargs)

    def execute_joint_trajectory(self, joint_trajectory, *, subsample=1, tolerance=0.01, max_steps=120):
        traj = np.asarray(joint_trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] < 7:
            raise ValueError(f"joint_trajectory must be (T, 7), got shape {traj.shape}")
        indices = list(range(0, len(traj), int(subsample)))
        if indices and indices[-1] != len(traj) - 1:
            indices.append(len(traj) - 1)
        for i in indices:
            self._env.move_to_joints_blocking(traj[i, :7], tolerance=tolerance, max_steps=max_steps)

    def update_curobo_world_with_object(self, object_name, *, object_mask=None, camera_name=None,
                                        robot_distance_threshold=0.15, robot_file="franka.yml",
                                        object_name_in_world=None, scene_name="scene", **kwargs):
        if object_mask is None:
            raise ValueError(f"object_mask is required to build CuRobo world for '{object_name}'")
        c = self._camera(camera_name or self.camera_name)
        obs = self.get_observation()
        ee = np.asarray(obs["robot_cartesian_pos"], dtype=np.float64)
        world = self._upstream_curobo().create_curobo_world_from_depth_with_object(
            c["depth"], object_mask, c["intrinsics"], camera_pose=c["pose_mat"],
            object_name=object_name_in_world or object_name.replace(" ", "_"),
            scene_name=scene_name, robot_distance_threshold=robot_distance_threshold,
            robot_file=robot_file,
            robot_joint_position=np.asarray(obs["robot_joint_pos"], dtype=np.float64),
            object_pose_override=(ee[:3], ee[3:7]), **kwargs)
        self._curobo_world_config = world
        self._curobo_world_object_name = object_name_in_world or object_name.replace(" ", "_")
        return world

    def plan_with_grasped_object(self, target_pose, object_name, *, object_pose=None,
                                 object_mask=None, world_config=None,
                                 robot_collision_sphere_buffer=-0.01,
                                 collision_activation_distance=0.01, **kwargs):
        world = world_config or self._curobo_world_config
        if world is None:
            raise RuntimeError("no CuRobo world configured; call update_curobo_world_with_object first")
        obs = self.get_observation()
        object_name = getattr(self, "_curobo_world_object_name", object_name.replace(" ", "_"))
        return self._upstream_curobo().plan_with_grasped_object(
            world, np.asarray(obs["robot_joint_pos"], dtype=np.float64), target_pose,
            object_name, robot_collision_sphere_buffer=robot_collision_sphere_buffer,
            collision_activation_distance=collision_activation_distance, **kwargs)

    def supports_dual_arm(self) -> bool: return all(hasattr(self._env, n) for n in ("move_to_joints_blocking_arm1", "move_to_joints_blocking_both"))
    def move_to_joints_arm0(self, joints): return self.move_to_joints(joints)
    def move_to_joints_arm1(self, joints):
        fn = getattr(self._env, "move_to_joints_blocking_arm1", None)
        if fn is None: raise RuntimeError("dual-arm arm1 control unavailable")
        return fn(joints)
    def open_gripper_arm0(self): return self.open_gripper()
    def close_gripper_arm0(self): return self.close_gripper()
    def open_gripper_arm1(self):
        fn = getattr(self._env, "_set_gripper_arm1", None)
        if fn is None: raise RuntimeError("dual-arm arm1 gripper unavailable")
        fn(1.0)
        for _ in range(40): self._env._step_once()
    def close_gripper_arm1(self):
        fn = getattr(self._env, "_set_gripper_arm1", None)
        if fn is None: raise RuntimeError("dual-arm arm1 gripper unavailable")
        fn(0.0)
        for _ in range(60): self._env._step_once()

    def get_arm0_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.get_observation()
        return np.asarray(obs["robot_cartesian_pos"][:3]), np.asarray(obs["robot_cartesian_pos"][3:7])

    def get_arm1_gripper_pose(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self._native_observation()
        value = obs.get("robot1_cartesian_pos") if isinstance(obs, Mapping) else None
        if value is None:
            raise RuntimeError("dual-arm arm1 gripper pose unavailable")
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if len(value) < 7:
            raise RuntimeError("robot1_cartesian_pos must contain position and WXYZ quaternion")
        return value[:3], value[3:7]

    def solve_ik_arm0(self, position: np.ndarray,
                      quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve arm-0 IK with the upstream TCP-offset convention."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        target = pos + rot.apply(self._TCP_OFFSET)
        result = self.cfg
        previous = self.cfg
        for _ in range(5):
            result = self._solve_ik_with_prev(target, quat, previous)
            if previous is not None and np.allclose(result, previous, atol=1e-3):
                break
            previous = result
        self.cfg = result
        return np.asarray(result, dtype=np.float64).reshape(-1)[:7]

    def solve_ik_arm1(self, position: np.ndarray,
                      quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Solve arm-1 IK after the provider's arm-frame transform."""
        if not self.supports_dual_arm():
            raise RuntimeError("Environment does not support Arm 1 control")
        pos0 = np.asarray(position, dtype=np.float64).reshape(3)
        quat0 = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        transform = getattr(self._env, "transform_pose_arm0_to_arm1", None)
        if callable(transform):
            pos1, quat1 = transform(pos0, quat0)
        else:
            T = getattr(self._env, "arm0_to_arm1_transform", None)
            if T is None:
                raise RuntimeError("dual-arm frame transform unavailable")
            T = np.asarray(T, dtype=np.float64).reshape(4, 4)
            target = np.eye(4, dtype=np.float64)
            target[:3, :3] = Rotation.from_quat([quat0[1], quat0[2], quat0[3], quat0[0]]).as_matrix()
            target[:3, 3] = pos0
            target = T @ target
            pos1 = target[:3, 3]
            qxyzw = Rotation.from_matrix(target[:3, :3]).as_quat()
            quat1 = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]])
        pos1 = np.asarray(pos1, dtype=np.float64).reshape(3)
        quat1 = np.asarray(quat1, dtype=np.float64).reshape(4)
        rot1 = Rotation.from_quat([quat1[1], quat1[2], quat1[3], quat1[0]])
        target = pos1 + rot1.apply(self._DUAL_TCP_OFFSET)
        result = self.cfg
        previous = self.cfg
        for _ in range(5):
            result = self._solve_ik_with_prev(target, quat1, previous)
            if previous is not None and np.allclose(result, previous, atol=1e-3):
                break
            previous = result
        self.cfg = result
        return np.asarray(result, dtype=np.float64).reshape(-1)[:7]
    def goto_pose_arm0(self, position, quaternion_wxyz, z_approach=0.0): return self.goto_pose(position, quaternion_wxyz, z_approach)
    def goto_pose_arm1(self, position, quaternion_wxyz, z_approach=0.0): raise RuntimeError("dual-arm arm1 pose control unavailable")
    def goto_pose_both(self, position0, quaternion_wxyz0, position1, quaternion_wxyz1, z_approach=0.0):
        self.goto_pose_arm0(position0, quaternion_wxyz0, z_approach); return self.goto_pose_arm1(position1, quaternion_wxyz1, z_approach)


class FrankaLiberoApiProxy:
    """JSON-RPC proxy with the same controller-facing method names."""
    def __init__(self, rpc: Callable[[str, tuple, dict], Any]): self._rpc = rpc
    def __getattr__(self, name: str):
        if name.startswith("_"): raise AttributeError(name)
        return lambda *args, **kwargs: self._rpc(name, args, kwargs)


__all__ = ["FrankaLiberoApi", "FrankaLiberoApiProxy", "depth_to_pointcloud", "depth_to_point_cloud", "mask_to_world_points", "pixel_to_world_point", "decompose_transform"]
