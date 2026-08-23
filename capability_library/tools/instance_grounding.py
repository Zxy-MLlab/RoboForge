"""Sensor-only helpers for resolving relation regions to movable instances."""
from __future__ import annotations

import numpy as np


def bbox_overlap_fraction(inner, outer):
    """Return how much of ``inner`` lies inside ``outer`` in image space."""
    ix0, iy0, ix1, iy1 = [float(value) for value in inner]
    ox0, oy0, ox1, oy1 = [float(value) for value in outer]
    area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if area <= 0:
        return 0.0
    intersection = (
        max(0.0, min(ix1, ox1) - max(ix0, ox0))
        * max(0.0, min(iy1, oy1) - max(iy0, oy0))
    )
    return intersection / area


def project_relation_region_to_movable(
    regions,
    relation_region,
    preferred_query,
    initial_source_id=None,
    min_overlap=0.35,
):
    """Resolve a containing relation region to a specific movable detection.

    The input is restricted to open-vocabulary detections and visible bounding
    boxes.  No simulator instance IDs or object poses are accepted.
    """
    direct = [
        item
        for item in relation_region["detections"]
        if item["query"] == preferred_query
    ]
    if direct:
        return relation_region, None

    candidates = []
    for region in regions:
        matches = [
            item for item in region["detections"] if item["query"] == preferred_query
        ]
        if not matches:
            continue
        overlap = bbox_overlap_fraction(
            region["bbox_xyxy"], relation_region["bbox_xyxy"]
        )
        if overlap < min_overlap:
            continue
        confidence = max(float(item["score"]) for item in matches)
        preferred = 1 if region["id"] == initial_source_id else 0
        candidates.append((preferred, overlap, confidence, region))

    if not candidates:
        return relation_region, None
    _, overlap, _, projected = max(candidates, key=lambda item: item[:3])
    return projected, {
        "from": relation_region["id"],
        "to": projected["id"],
        "rule": "nested_movable_detection",
        "bbox_overlap": overlap,
    }


def relation_allows_fused_source_reference(source_clause: str) -> bool:
    """Only containment/support relations can denote a fused visual stack."""
    clause = " ".join(str(source_clause).lower().split())
    return " on the " in clause or " in the " in clause


def relation_verifier_region_is_consistent(
    relation_region, initial_source_region, preferred_query, min_overlap=0.35
):
    """Reject a second-pass region that neither is nor contains the source."""
    if any(
        item["query"] == preferred_query for item in relation_region["detections"]
    ):
        return True
    return (
        bbox_overlap_fraction(
            initial_source_region["bbox_xyxy"], relation_region["bbox_xyxy"]
        )
        >= min_overlap
    )


def drawer_pull_direction(
    handle_xyz, cabinet_xyz, contained_object_xyz=None, min_separation=0.02
):
    """Infer the outward drawer axis from visible handle/interior geometry."""
    handle = np.asarray(handle_xyz, dtype=float)
    cabinet = np.asarray(cabinet_xyz, dtype=float)
    if handle.shape != (3,) or cabinet.shape != (3,):
        raise ValueError("handle_xyz and cabinet_xyz must be 3-vectors")
    origin = cabinet
    if contained_object_xyz is not None:
        contained = np.asarray(contained_object_xyz, dtype=float)
        if contained.shape != (3,):
            raise ValueError("contained_object_xyz must be a 3-vector")
        if np.isfinite(contained).all():
            origin = contained
    delta = handle[:2] - origin[:2]
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= float(min_separation):
        return None
    return delta / norm


__all__ = [
    "bbox_overlap_fraction",
    "project_relation_region_to_movable",
    "relation_allows_fused_source_reference",
    "relation_verifier_region_is_consistent",
    "drawer_pull_direction",
]
