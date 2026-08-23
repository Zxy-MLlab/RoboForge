"""Sensor-only checks used by autonomous grasp/place recovery.

These helpers deliberately operate on estimated RGB-D region geometry and
proprioception. They never consume evaluator values or simulator object state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import cv2
import numpy as np


@dataclass(frozen=True)
class AttachmentObservation:
    object_xyz: np.ndarray
    eef_xyz: np.ndarray
    previous_object_xyz: np.ndarray
    gripper_width: float


@dataclass(frozen=True)
class PlacementObservation:
    object_xyz: np.ndarray
    target_xyz: np.ndarray
    object_radius: float = 0.045
    target_radius: float = 0.075
    height_tolerance: float = 0.045


def attachment_verified(obs: AttachmentObservation, *, max_eef_distance: float = 0.12,
                        min_object_motion: float = 0.025,
                        max_gripper_width: float = 0.075) -> bool:
    """Return true only when the observed object follows a lifted EEF."""
    obj = np.asarray(obs.object_xyz, dtype=float)
    eef = np.asarray(obs.eef_xyz, dtype=float)
    prev = np.asarray(obs.previous_object_xyz, dtype=float)
    if not (np.isfinite(obj).all() and np.isfinite(eef).all() and np.isfinite(prev).all()):
        return False
    return (np.linalg.norm(obj - eef) <= max_eef_distance and
            np.linalg.norm(obj - prev) >= min_object_motion and
            float(obs.gripper_width) <= max_gripper_width)


def topdown_rotation_with_candidate_yaw(grasp_rotation: np.ndarray) -> np.ndarray:
    """Keep a candidate's planar yaw while enforcing vertical EEF approach."""
    rotation = np.asarray(grasp_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("grasp_rotation must be a finite 3x3 matrix")
    # GraspNet local X is approach. Under the calibrated Panda convention its
    # local Y maps to EEF X; retain only that axis's observable table-plane yaw.
    x_axis = rotation[:, 1].copy()
    x_axis[2] = 0.0
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-6:
        x_axis = rotation[:, 2].copy()
        x_axis[2] = 0.0
        norm = float(np.linalg.norm(x_axis))
    if norm < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    x_axis /= norm
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, x_axis)
    result = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(result) < 0.999:
        raise ValueError("failed to construct right-handed top-down rotation")
    return result


def select_orientation_compatible_grasp_pool(audit: Mapping, orientation: str):
    """Select strict grasps or the declared calibrated-orientation fallback."""
    strict = list(audit.get("grasps") or ())
    if strict:
        return strict, "full_6dof"
    if str(orientation) == "robot-topdown":
        fallback = list(audit.get("orientation_override_grasps") or ())
        if fallback:
            return fallback, "translation_with_calibrated_robot_orientation"
    return [], "none"


def merge_orientation_compatible_grasp_pools(audit: Mapping, orientation: str):
    """Return strict candidates followed by bounded calibrated fallbacks.

    A strict 6-DoF pool can be non-empty yet fail on contact geometry.  The
    orientation-override candidates are still legal public GraspNet proposals
    when the controller uses the calibrated robot-topdown wrist orientation,
    so they must remain available after the strict pool is exhausted.  Near
    duplicate translations are removed to keep retries diverse and bounded.
    """
    strict = list(audit.get("grasps") or ())
    if str(orientation) != "robot-topdown":
        return strict, "full_6dof" if strict else "none"
    fallback = list(audit.get("orientation_override_grasps") or ())
    merged = list(strict)
    for candidate in fallback:
        translation = np.asarray(candidate.get("translation_world", ()), dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            continue
        if any(
            np.linalg.norm(translation - np.asarray(existing["translation_world"], dtype=float))
            < 0.012
            for existing in merged
        ):
            continue
        merged.append(candidate)
    if strict and len(merged) > len(strict):
        return merged, "full_6dof_then_calibrated_fallback"
    if strict:
        return strict, "full_6dof"
    if fallback:
        return fallback, "translation_with_calibrated_robot_orientation"
    return [], "none"


def placement_verified(obs: PlacementObservation) -> bool:
    """Check a conservative top-down support relation from estimated geometry."""
    obj = np.asarray(obs.object_xyz, dtype=float)
    target = np.asarray(obs.target_xyz, dtype=float)
    if not (np.isfinite(obj).all() and np.isfinite(target).all()):
        return False
    xy_limit = max(0.0, float(obs.target_radius) - float(obs.object_radius))
    return (np.linalg.norm(obj[:2] - target[:2]) <= xy_limit and
            abs(float(obj[2] - target[2])) <= float(obs.height_tolerance))


def image_support_overlap_verified(object_bbox: Sequence[float], target_bbox: Sequence[float],
                                   *, min_object_overlap: float = 0.25) -> bool:
    """Verify support from legal image regions when the target is occluded.

    The overlap is normalized by the object's box, not the partially visible
    target box.  This complements rather than replaces the RGB-D height check.
    """
    obj=np.asarray(object_bbox,dtype=float); target=np.asarray(target_bbox,dtype=float)
    if obj.shape!=(4,) or target.shape!=(4,) or not (np.isfinite(obj).all() and np.isfinite(target).all()):
        return False
    ox0,oy0,ox1,oy1=obj; tx0,ty0,tx1,ty1=target
    object_area=max(0.,ox1-ox0)*max(0.,oy1-oy0)
    if object_area<=0: return False
    intersection=max(0.,min(ox1,tx1)-max(ox0,tx0))*max(0.,min(oy1,ty1)-max(oy0,ty0))
    return intersection/object_area>=float(min_object_overlap)


def mask_support_metrics(object_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    """Measure footprint containment in a previously observed support mask.

    The target's convex hull reconstructs the usable support surface from a
    legal pre-placement SAM observation.  This is deliberately stricter than
    detector-box intersection: a bowl merely touching a plate edge can have a
    large box overlap while lacking enough clearance for stable support.
    """
    obj = np.asarray(object_mask, dtype=np.uint8)
    target = np.asarray(target_mask, dtype=np.uint8)
    if obj.ndim != 2 or target.ndim != 2 or obj.shape != target.shape:
        raise ValueError("object_mask and target_mask must be same-shaped 2-D arrays")
    obj = (obj > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    object_pixels = int(obj.sum())
    target_pixels = int(target.sum())
    if object_pixels == 0 or target_pixels < 3:
        return {
            "object_pixels": float(object_pixels),
            "target_pixels": float(target_pixels),
            "containment": 0.0,
            "center_clearance_px": 0.0,
            "equivalent_object_radius_px": 0.0,
            "clearance_ratio": 0.0,
            "centroid_x": float("nan"),
            "centroid_y": float("nan"),
        }

    target_y, target_x = np.nonzero(target)
    points = np.c_[target_x, target_y].astype(np.int32)
    hull = cv2.convexHull(points[:, None, :])
    support = np.zeros_like(target, dtype=np.uint8)
    cv2.fillConvexPoly(support, hull[:, 0, :], 1)

    object_y, object_x = np.nonzero(obj)
    centroid_x = float(np.mean(object_x))
    centroid_y = float(np.mean(object_y))
    px = int(np.clip(round(centroid_x), 0, obj.shape[1] - 1))
    py = int(np.clip(round(centroid_y), 0, obj.shape[0] - 1))
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 5)
    clearance = float(distance[py, px])
    radius = float(np.sqrt(object_pixels / np.pi))
    containment = float(np.count_nonzero(obj & support) / object_pixels)
    return {
        "object_pixels": float(object_pixels),
        "target_pixels": float(target_pixels),
        "containment": containment,
        "center_clearance_px": clearance,
        "equivalent_object_radius_px": radius,
        "clearance_ratio": clearance / max(radius, 1e-9),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    }


def mask_support_verified(object_mask: np.ndarray, target_mask: np.ndarray, *,
                          min_containment: float = 0.55,
                          min_clearance_ratio: float = 0.25) -> bool:
    """Return true only when an object footprint is stably inside support."""
    metrics = mask_support_metrics(object_mask, target_mask)
    return (metrics["containment"] >= float(min_containment) and
            metrics["clearance_ratio"] >= float(min_clearance_ratio))


def temporal_mask_support_verified(first_object_mask: np.ndarray,
                                   second_object_mask: np.ndarray,
                                   target_mask: np.ndarray, *,
                                   max_centroid_motion_px: float = 8.0,
                                   min_containment: float = 0.55,
                                   min_clearance_ratio: float = 0.25) -> bool:
    """Require support in two observations and negligible post-settle motion."""
    first = mask_support_metrics(first_object_mask, target_mask)
    second = mask_support_metrics(second_object_mask, target_mask)
    centers = np.array([[first["centroid_x"], first["centroid_y"]],
                        [second["centroid_x"], second["centroid_y"]]], dtype=float)
    if not np.isfinite(centers).all():
        return False
    motion = float(np.linalg.norm(centers[1] - centers[0]))
    return (first["containment"] >= float(min_containment) and
            second["containment"] >= float(min_containment) and
            first["clearance_ratio"] >= float(min_clearance_ratio) and
            second["clearance_ratio"] >= float(min_clearance_ratio) and
            motion <= float(max_centroid_motion_px))


def sensor_centered_place_goal(provisional_eef_goal: Sequence[float],
                               target_object_xyz: Sequence[float],
                               carried_object_xyz: Sequence[float],
                               observed_eef_xyz: Sequence[float]) -> np.ndarray:
    """Recenter a place goal using the observed carried-object/EFF transform.

    Grasp proposals describe pre-contact geometry, while an object can settle
    asymmetrically between the fingers.  After a verified lift, RGB-D and
    proprioception directly reveal that realized transform.  Preserve the
    provisional release height but correct its tabletop coordinates so the
    carried object, rather than the wrist, is centered on the support.
    """
    goal = np.asarray(provisional_eef_goal, dtype=float).copy()
    target = np.asarray(target_object_xyz, dtype=float)
    carried = np.asarray(carried_object_xyz, dtype=float)
    eef = np.asarray(observed_eef_xyz, dtype=float)
    if any(value.shape != (3,) for value in (goal, target, carried, eef)):
        raise ValueError("all placement geometry inputs must be 3-D")
    if not all(np.isfinite(value).all() for value in (goal, target, carried, eef)):
        raise ValueError("all placement geometry inputs must be finite")
    goal[:2] = target[:2] - (carried - eef)[:2]
    return goal


def ranked_retry_indices(candidate_count: int, *, max_attempts: int = 3) -> tuple[int, ...]:
    """Return bounded, deterministic candidate order for autonomous retry."""
    if candidate_count < 0 or max_attempts <= 0:
        raise ValueError("candidate_count must be non-negative and max_attempts positive")
    return tuple(range(min(candidate_count, max_attempts)))


def diverse_grasp_retry_indices(grasps: Sequence[Mapping[str, object]], *, max_attempts: int = 5) -> tuple[int, ...]:
    """Select high-ranked but geometrically distinct grasp candidates."""
    if max_attempts<=0: raise ValueError("max_attempts must be positive")
    if not grasps: return ()
    translations=[]; closing_axes=[]
    for grasp in grasps:
        translations.append(np.asarray(grasp["translation_world"],float))
        rotation=np.asarray(grasp["rotation_world"],float)
        closing_axes.append(rotation[:,1]/max(np.linalg.norm(rotation[:,1]),1e-9))
    selected=[0]
    while len(selected)<min(len(grasps),max_attempts):
        best=None; best_score=-np.inf
        for index in range(len(grasps)):
            if index in selected: continue
            novelty=min(
                (1.-abs(float(np.dot(closing_axes[index],closing_axes[other]))))
                +.5*min(1.,float(np.linalg.norm(translations[index]-translations[other]))/.05)
                for other in selected
            )
            # Retain a small rank prior while making duplicate pose clusters
            # substantially less valuable than a distinct lower-ranked grasp.
            score=novelty-.01*index
            if score>best_score: best_score=score; best=index
        if best is None: break
        selected.append(best)
    return tuple(selected)


__all__ = ["AttachmentObservation", "PlacementObservation", "attachment_verified",
           "placement_verified", "image_support_overlap_verified", "ranked_retry_indices",
           "mask_support_metrics", "mask_support_verified",
           "temporal_mask_support_verified", "sensor_centered_place_goal",
           "diverse_grasp_retry_indices"]
