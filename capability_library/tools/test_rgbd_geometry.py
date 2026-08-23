import numpy as np
import pytest

from rgbd_geometry import camera_to_robot, pixel_to_camera, robust_depth


def test_robust_depth_rejects_outliers():
    depth = np.array([[1.0, 1.01, 1.02, 9.0, 0.0]])
    mask = np.ones_like(depth, dtype=bool)
    assert robust_depth(depth, mask) == pytest.approx(1.01, abs=0.01)


def test_pixel_backprojection_and_transform():
    intrinsic = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    point = pixel_to_camera(60, 50, 2.0, intrinsic)
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    assert np.allclose(camera_to_robot(point, transform), [1.2, 2.2, 5.0])


def test_invalid_depth_is_rejected():
    with pytest.raises(ValueError):
        pixel_to_camera(0, 0, 0, np.eye(3))
