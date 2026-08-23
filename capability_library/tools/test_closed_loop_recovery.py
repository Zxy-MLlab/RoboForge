import numpy as np
import pytest

from closed_loop_recovery import (AttachmentObservation, PlacementObservation,
                                  attachment_verified, placement_verified,
                                  image_support_overlap_verified, ranked_retry_indices)
from closed_loop_recovery import diverse_grasp_retry_indices
from closed_loop_recovery import (mask_support_metrics, mask_support_verified,
                                  temporal_mask_support_verified,
                                  sensor_centered_place_goal,
                                  topdown_rotation_with_candidate_yaw,
                                  select_orientation_compatible_grasp_pool,
                                  merge_orientation_compatible_grasp_pools)


def test_attachment_requires_motion_and_eef_proximity():
    base = dict(eef_xyz=np.array([0., 0., 1.]), previous_object_xyz=np.array([0., 0., .95]), gripper_width=.04)
    assert attachment_verified(AttachmentObservation(np.array([0., 0., .98]), **base))
    assert not attachment_verified(AttachmentObservation(np.array([.3, 0., .98]), **base))
    assert not attachment_verified(AttachmentObservation(np.array([0., 0., .96]), **base))


def test_placement_requires_supporting_xy_overlap():
    assert placement_verified(PlacementObservation(np.array([0., 0., .92]), np.array([.01, 0., .94])))
    assert not placement_verified(PlacementObservation(np.array([.2, 0., .92]), np.array([0., 0., .94])))


def test_retry_order_is_bounded():
    assert ranked_retry_indices(5, max_attempts=3) == (0, 1, 2)
    with pytest.raises(ValueError):
        ranked_retry_indices(1, max_attempts=0)


def test_image_support_overlap_handles_occluded_target_center():
    # Object overlaps the visible upper portion of a partially occluded plate.
    assert image_support_overlap_verified([348,313,426,383],[344,356,442,418])
    assert not image_support_overlap_verified([50,50,100,100],[300,300,400,400])


def test_diverse_retry_escapes_duplicate_top_ranked_cluster():
    def grasp(x,closing):
        rotation=np.eye(3); rotation[:,1]=closing
        return {"translation_world":[x,0,0],"rotation_world":rotation.tolist()}
    grasps=[grasp(0,[0,1,0]),grasp(.001,[0,1,0]),grasp(.002,[0,1,0]),grasp(.05,[1,0,0])]
    selected=diverse_grasp_retry_indices(grasps,max_attempts=2)
    assert selected==(0,3)


def test_mask_support_rejects_edge_contact_despite_box_like_overlap():
    yy, xx = np.mgrid[:128, :128]
    target = (np.square(xx - 64) + np.square(yy - 64) <= 38 ** 2)
    centered = (np.square(xx - 64) + np.square(yy - 64) <= 20 ** 2)
    edge = (np.square(xx - 98) + np.square(yy - 64) <= 20 ** 2)
    assert mask_support_verified(centered, target)
    assert not mask_support_verified(edge, target)
    metrics = mask_support_metrics(edge, target)
    assert metrics["containment"] > 0.5
    assert metrics["clearance_ratio"] < 0.25


def test_temporal_mask_support_requires_both_frames_and_stability():
    yy, xx = np.mgrid[:128, :128]
    target = (np.square(xx - 64) + np.square(yy - 64) <= 38 ** 2)
    first = (np.square(xx - 64) + np.square(yy - 64) <= 20 ** 2)
    stable = (np.square(xx - 67) + np.square(yy - 64) <= 20 ** 2)
    sliding = (np.square(xx - 78) + np.square(yy - 64) <= 20 ** 2)
    assert temporal_mask_support_verified(first, stable, target)
    assert not temporal_mask_support_verified(first, sliding, target)


def test_conservative_support_profile_rejects_rim_contact_false_positive():
    yy, xx = np.mgrid[:128, :128]
    target = (np.square(xx - 64) + np.square(yy - 64) <= 38 ** 2)
    rim_contact = (np.square(xx - 94) + np.square(yy - 64) <= 20 ** 2)
    # This placement passes the former permissive thresholds but lacks enough
    # boundary clearance to establish full support.
    assert temporal_mask_support_verified(
        rim_contact, rim_contact, target,
        min_containment=.55, min_clearance_ratio=.25,
    )
    assert not temporal_mask_support_verified(
        rim_contact, rim_contact, target,
        min_containment=.75, min_clearance_ratio=.75,
    )


def test_mask_support_validates_shapes_and_empty_masks():
    with pytest.raises(ValueError):
        mask_support_metrics(np.zeros((3, 3)), np.zeros((4, 4)))
    assert not mask_support_verified(np.zeros((8, 8)), np.ones((8, 8)))


def test_sensor_centered_goal_uses_realized_carried_offset_and_preserves_height():
    goal = sensor_centered_place_goal(
        [0.10, 0.20, 0.95], [0.06, 0.18, 0.91],
        [0.02, 0.10, 1.15], [0.00, 0.05, 1.14],
    )
    assert np.allclose(goal, [0.04, 0.13, 0.95])
    with pytest.raises(ValueError):
        sensor_centered_place_goal([0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])


def test_topdown_rotation_keeps_candidate_planar_yaw():
    angle = np.deg2rad(35.0)
    candidate = np.array([
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
        [1.0, 0.0, 0.0],
    ])
    result = topdown_rotation_with_candidate_yaw(candidate)
    assert np.allclose(result[:, 2], [0.0, 0.0, -1.0])
    assert np.allclose(result[:2, 0], [np.cos(angle), np.sin(angle)])
    assert np.allclose(result.T @ result, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(result), 1.0)


def test_grasp_pool_fallback_is_bounded_by_declared_orientation():
    audit = {"grasps": [], "orientation_override_grasps": [{"id": 1}]}
    pool, kind = select_orientation_compatible_grasp_pool(audit, "robot-topdown")
    assert pool == [{"id": 1}]
    assert kind == "translation_with_calibrated_robot_orientation"
    assert select_orientation_compatible_grasp_pool(audit, "model") == ([], "none")


def test_grasp_pool_tries_calibrated_fallback_after_strict_pool():
    strict = {"translation_world": [0.0, 0.0, 1.0], "rotation_world": np.eye(3).tolist()}
    duplicate = {"translation_world": [0.005, 0.0, 1.0], "rotation_world": np.eye(3).tolist()}
    fallback = {"translation_world": [0.04, 0.0, 1.0], "rotation_world": np.eye(3).tolist()}
    pool, kind = merge_orientation_compatible_grasp_pools(
        {"grasps": [strict], "orientation_override_grasps": [duplicate, fallback]},
        "robot-topdown",
    )
    assert [item["translation_world"] for item in pool] == [[0.0, 0.0, 1.0], [0.04, 0.0, 1.0]]
    assert kind == "full_6dof_then_calibrated_fallback"
