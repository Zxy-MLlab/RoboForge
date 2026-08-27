"""Benchmark-neutral reliability evidence for detection and instance association.

The auditor does not correct labels or choose a task target.  It exposes when a
perception result is insufficient to support such a choice, leaving acquisition
and recovery decisions to the coding agent.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    if not all(_finite_number(item) for item in value):
        return None
    x0, y0, x1, y1 = (float(item) for item in value)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _point(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    if not all(_finite_number(item) for item in value):
        return None
    return tuple(float(item) for item in value)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    intersection = width * height
    union = ((lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0)
             - intersection)
    return intersection / union if union > 0 else 0.0


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _detections(result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped = result.get("detections")
    if not isinstance(grouped, Mapping):
        return {}
    return {
        str(query): [dict(item) for item in items if isinstance(item, Mapping)]
        for query, items in grouped.items() if isinstance(items, list)
    }


def audit_detection_result(
    result: Mapping[str, Any], *, required_queries: Iterable[str] = (),
    distinct_query_pairs: Iterable[Sequence[str]] = (),
    ambiguous_score_margin: float = 0.08,
    alias_iou_threshold: float = 0.80,
    alias_world_distance_m: float = 0.025,
) -> dict[str, Any]:
    """Describe evidence quality without asserting a corrected semantic label.

    ``distinct_query_pairs`` must come from task semantics: for example, an
    instruction that relates one object to a support container establishes that the two
    entities cannot be represented by the same image region.  The auditor does
    not infer semantic exclusivity from label strings.
    """
    grouped = _detections(result)
    issues: list[dict[str, Any]] = []
    required = list(dict.fromkeys(str(item).strip() for item in required_queries
                                  if str(item).strip()))
    for query in required:
        if not grouped.get(query):
            issues.append({"kind": "required_query_missing", "query": query})

    for query, candidates in grouped.items():
        usable = []
        for index, candidate in enumerate(candidates):
            box = _box(candidate.get("box_xyxy"))
            point = _point(candidate.get("world_xyz"))
            if candidate.get("projection_error") or point is None:
                issues.append({"kind": "metric_projection_unavailable", "query": query,
                               "candidate_index": index,
                               "detail": str(candidate.get("projection_error") or
                                             "world_xyz is absent or invalid")[:300]})
            if box is None:
                issues.append({"kind": "pixel_geometry_invalid", "query": query,
                               "candidate_index": index})
            score = candidate.get("score")
            if _finite_number(score):
                usable.append((float(score), index))
        usable.sort(reverse=True)
        if len(usable) >= 2:
            margin = usable[0][0] - usable[1][0]
            if margin <= float(ambiguous_score_margin):
                issues.append({"kind": "same_query_competing_candidates", "query": query,
                               "candidate_indices": [usable[0][1], usable[1][1]],
                               "score_margin": margin})

    seen_pairs = set()
    for raw_pair in distinct_query_pairs:
        if (not isinstance(raw_pair, Sequence) or isinstance(raw_pair, (str, bytes))
                or len(raw_pair) != 2):
            raise ValueError("each distinct_query_pairs item must contain two query names")
        left_query, right_query = (str(item).strip() for item in raw_pair)
        if not left_query or not right_query or left_query == right_query:
            raise ValueError("distinct query names must be nonempty and different")
        pair = tuple(sorted((left_query, right_query)))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        for left_index, left in enumerate(grouped.get(left_query, [])):
            left_box, left_point = _box(left.get("box_xyxy")), _point(left.get("world_xyz"))
            for right_index, right in enumerate(grouped.get(right_query, [])):
                right_box = _box(right.get("box_xyxy")); right_point = _point(right.get("world_xyz"))
                overlap = _iou(left_box, right_box) if left_box and right_box else 0.0
                world_distance = (_distance(left_point, right_point)
                                  if left_point and right_point else None)
                if (overlap >= float(alias_iou_threshold)
                        or (world_distance is not None
                            and world_distance <= float(alias_world_distance_m))):
                    issues.append({"kind": "distinct_entities_share_visual_instance",
                                   "queries": [left_query, right_query],
                                   "candidate_indices": [left_index, right_index],
                                   "box_iou": overlap,
                                   "world_distance_m": world_distance})

    fatal_kinds = {"required_query_missing"}
    status = ("unusable" if any(item["kind"] in fatal_kinds for item in issues)
              else "uncertain" if issues else "supported")
    return {
        "protocol": "embodied-codex-perception-reliability-v1",
        "frame_id": result.get("frame_id"),
        "status": status,
        "requires_independent_confirmation": bool(issues),
        "issues": issues,
        "query_candidate_counts": {query: len(items) for query, items in grouped.items()},
        "decision_boundary": (
            "This report measures evidence sufficiency. It does not correct labels, "
            "select an instance, or establish task success."
        ),
    }


def audit_temporal_association(
    previous_result: Mapping[str, Any], current_result: Mapping[str, Any], *,
    query: str, previous_candidate_index: int,
    maximum_world_displacement_m: float,
    ambiguity_margin_m: float = 0.015,
) -> dict[str, Any]:
    """Assess whether geometry alone identifies one prior instance in a new frame.

    The displacement bound is supplied by the controller's motion hypothesis;
    no benchmark-specific threshold or assumption about whether the object was
    grasped is embedded here.
    """
    previous = _detections(previous_result).get(str(query), [])
    current = _detections(current_result).get(str(query), [])
    if previous_candidate_index < 0 or previous_candidate_index >= len(previous):
        raise ValueError("previous_candidate_index is outside the prior candidates")
    anchor = _point(previous[previous_candidate_index].get("world_xyz"))
    if anchor is None:
        return {"protocol": "embodied-codex-temporal-association-audit-v1",
                "status": "unusable", "associated_candidate_index": None,
                "reason": "previous candidate has no metric position"}
    ranked = sorted(
        (_distance(anchor, point), index)
        for index, item in enumerate(current)
        for point in [_point(item.get("world_xyz"))] if point is not None
    )
    if not ranked or ranked[0][0] > float(maximum_world_displacement_m):
        return {"protocol": "embodied-codex-temporal-association-audit-v1",
                "status": "unresolved", "associated_candidate_index": None,
                "reason": "no current candidate lies within the declared motion bound",
                "nearest_distance_m": ranked[0][0] if ranked else None}
    margin = ranked[1][0] - ranked[0][0] if len(ranked) > 1 else None
    if margin is not None and margin <= float(ambiguity_margin_m):
        return {"protocol": "embodied-codex-temporal-association-audit-v1",
                "status": "ambiguous", "associated_candidate_index": None,
                "candidate_indices": [ranked[0][1], ranked[1][1]],
                "distances_m": [ranked[0][0], ranked[1][0]],
                "association_margin_m": margin,
                "reason": "geometry does not uniquely preserve instance identity"}
    return {"protocol": "embodied-codex-temporal-association-audit-v1",
            "status": "supported", "associated_candidate_index": ranked[0][1],
            "distance_m": ranked[0][0], "association_margin_m": margin,
            "decision_boundary": (
                "Geometric continuity supports this association but does not prove semantic identity."
            )}


__all__ = ["audit_detection_result", "audit_temporal_association"]
