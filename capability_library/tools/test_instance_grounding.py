import numpy as np

from instance_grounding import (
    bbox_overlap_fraction,
    drawer_pull_direction,
    project_relation_region_to_movable,
    relation_allows_fused_source_reference,
    relation_verifier_region_is_consistent,
)


def _region(region_id, bbox, query, score=0.5):
    return {
        "id": region_id,
        "bbox_xyxy": bbox,
        "detections": [{"query": query, "score": score, "bbox_xyxy": bbox}],
    }


def test_nested_bowl_is_projected_out_of_large_drawer_region():
    drawer = _region(7, [0, 213, 245, 439], "top drawer")
    bowl = _region(1, [99, 249, 193, 323], "black bowl", 0.67)
    projected, audit = project_relation_region_to_movable(
        [drawer, bowl], drawer, "black bowl", initial_source_id=1
    )
    assert projected["id"] == 1
    assert audit["rule"] == "nested_movable_detection"
    assert audit["bbox_overlap"] == 1.0


def test_initial_vlm_object_wins_when_multiple_objects_overlap_support():
    support = _region(0, [0, 0, 200, 200], "cabinet")
    expected = _region(2, [20, 20, 70, 70], "black bowl", 0.4)
    distractor = _region(3, [100, 100, 150, 150], "black bowl", 0.9)
    projected, _ = project_relation_region_to_movable(
        [support, expected, distractor], support, "black bowl", initial_source_id=2
    )
    assert projected["id"] == 2


def test_fused_region_is_for_containment_not_adjacency():
    assert relation_allows_fused_source_reference("pick bowl in the drawer")
    assert relation_allows_fused_source_reference("pick bowl on the cabinet")
    assert not relation_allows_fused_source_reference("pick bowl next to the plate")
    assert not relation_allows_fused_source_reference("pick bowl between the plate and ramekin")
    assert bbox_overlap_fraction([10, 10, 20, 20], [0, 0, 30, 30]) == 1.0


def test_relation_verifier_cannot_jump_to_unrelated_support_like_region():
    initial = _region(1, [0, 176, 68, 246], "black bowl", 0.73)
    unrelated = _region(5, [150, 185, 207, 232], "ramekin", 0.49)
    containing = _region(10, [0, 169, 245, 439], "top drawer", 0.31)
    direct = _region(8, [150, 185, 208, 232], "black bowl", 0.34)
    assert not relation_verifier_region_is_consistent(
        unrelated, initial, "black bowl"
    )
    assert relation_verifier_region_is_consistent(containing, initial, "black bowl")
    assert relation_verifier_region_is_consistent(direct, initial, "black bowl")


def test_drawer_pull_direction_points_from_cabinet_to_handle():
    direction = drawer_pull_direction([0.08, -0.05, 1.1], [-0.04, -0.17, 1.12])
    assert np.allclose(direction, [2 ** -0.5, 2 ** -0.5])
    aligned = drawer_pull_direction(
        [0.084, -0.055, 1.10], [-0.04, -0.17, 1.12], [0.084, -0.139, 1.11]
    )
    assert np.allclose(aligned, [0.0, 1.0], atol=1e-7)
    assert drawer_pull_direction([0, 0, 1], [0.001, 0, 1]) is None
