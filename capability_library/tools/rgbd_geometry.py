"""Sensor-only RGB-D geometry primitives.

These functions use pixels, depth, and declared calibration only. They do not
read simulator object poses, task identifiers, rewards, or demonstrations.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def robust_depth(depth_m: np.ndarray, mask: np.ndarray, *, low: float = 0.05, high: float = 5.0) -> float:
    """Return a trimmed median depth over a sensor-provided object mask."""
    values = np.asarray(depth_m, dtype=float)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values) & (values >= low) & (values <= high)]
    if values.size < 3:
        raise ValueError("insufficient valid depth pixels")
    lo, hi = np.quantile(values, [0.1, 0.9])
    trimmed = values[(values >= lo) & (values <= hi)]
    return float(np.median(trimmed if trimmed.size else values))


def pixel_to_camera(u: float, v: float, depth: float, intrinsic: np.ndarray) -> np.ndarray:
    """Back-project one RGB-D pixel into the camera frame."""
    matrix = np.asarray(intrinsic, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    z = float(depth)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("depth must be positive and finite")
    fx, fy = matrix[0, 0], matrix[1, 1]
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")
    return np.array([(float(u) - matrix[0, 2]) * z / fx, (float(v) - matrix[1, 2]) * z / fy, z], dtype=float)


def camera_to_robot(point_camera: Sequence[float], camera_to_robot_matrix: np.ndarray) -> np.ndarray:
    """Transform one camera-frame point with a declared homogeneous extrinsic."""
    point = np.asarray(point_camera, dtype=float)
    transform = np.asarray(camera_to_robot_matrix, dtype=float)
    if point.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("point must be 3D and transform must be 4x4")
    if not np.isfinite(point).all() or not np.isfinite(transform).all():
        raise ValueError("geometry inputs must be finite")
    homogeneous = transform @ np.r_[point, 1.0]
    if abs(homogeneous[3]) < 1e-9:
        raise ValueError("invalid homogeneous transform")
    return homogeneous[:3] / homogeneous[3]


__all__ = ["camera_to_robot", "pixel_to_camera", "robust_depth"]
