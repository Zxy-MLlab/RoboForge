import pytest
from pathlib import Path

from embodied_codex.capabilities.perception_reliability import (
    audit_detection_result, audit_temporal_association,
)
from embodied_codex.examples.evaluate_libero_skill import _load_class


def candidate(*, score, box, xyz=None, projection_error=None):
    item = {"score": score, "box_xyxy": box}
    if xyz is not None:
        item["world_xyz"] = xyz
    if projection_error is not None:
        item["projection_error"] = projection_error
    return item


def test_detection_audit_reports_supported_independent_entities():
    report = audit_detection_result({"frame_id": "f1", "detections": {
        "black bowl": [candidate(score=.91, box=[10, 10, 30, 30], xyz=[.1, .2, 1.0])],
        "cookie box": [candidate(score=.87, box=[80, 20, 110, 60], xyz=[.4, .2, 1.0])],
    }}, required_queries=["black bowl", "cookie box"],
       distinct_query_pairs=[["black bowl", "cookie box"]])
    assert report["status"] == "supported"
    assert report["requires_independent_confirmation"] is False


def test_detection_audit_exposes_missing_projection_competition_and_aliasing():
    report = audit_detection_result({"frame_id": "f2", "detections": {
        "black bowl": [
            candidate(score=.427, box=[10, 10, 40, 40], xyz=[.1, .2, 1.0]),
            candidate(score=.401, box=[70, 10, 100, 40], xyz=[.4, .2, 1.0]),
        ],
        "cookie box": [candidate(score=.868, box=[11, 11, 39, 39],
                                 projection_error="insufficient valid depth")],
    }}, required_queries=["black bowl", "cookie box", "plate"],
       distinct_query_pairs=[["black bowl", "cookie box"]])
    kinds = {item["kind"] for item in report["issues"]}
    assert report["status"] == "unusable"
    assert {"required_query_missing", "metric_projection_unavailable",
            "same_query_competing_candidates",
            "distinct_entities_share_visual_instance"} <= kinds


def test_detection_audit_does_not_assume_two_query_strings_are_distinct_entities():
    shared = candidate(score=.9, box=[10, 10, 40, 40], xyz=[.1, .2, 1.0])
    report = audit_detection_result({"detections": {
        "bowl": [shared], "black bowl": [shared],
    }}, required_queries=["bowl", "black bowl"])
    assert report["status"] == "supported"


def test_temporal_audit_supports_unique_metric_association():
    before = {"detections": {"bowl": [
        candidate(score=.8, box=[0, 0, 10, 10], xyz=[.10, .20, 1.0])]}}
    after = {"detections": {"bowl": [
        candidate(score=.7, box=[1, 0, 11, 10], xyz=[.11, .20, 1.0]),
        candidate(score=.9, box=[50, 0, 60, 10], xyz=[.40, .20, 1.0])]}}
    report = audit_temporal_association(before, after, query="bowl",
        previous_candidate_index=0, maximum_world_displacement_m=.05)
    assert report["status"] == "supported"
    assert report["associated_candidate_index"] == 0


def test_temporal_audit_refuses_ambiguous_same_class_identity():
    before = {"detections": {"bowl": [
        candidate(score=.8, box=[0, 0, 10, 10], xyz=[.10, .20, 1.0])]}}
    after = {"detections": {"bowl": [
        candidate(score=.7, box=[1, 0, 11, 10], xyz=[.11, .20, 1.0]),
        candidate(score=.9, box=[2, 0, 12, 10], xyz=[.115, .20, 1.0])]}}
    report = audit_temporal_association(before, after, query="bowl",
        previous_candidate_index=0, maximum_world_displacement_m=.05,
        ambiguity_margin_m=.01)
    assert report["status"] == "ambiguous"
    assert report["associated_candidate_index"] is None


def test_temporal_audit_requires_valid_prior_index():
    with pytest.raises(ValueError, match="previous_candidate_index"):
        audit_temporal_association({"detections": {"bowl": []}},
            {"detections": {"bowl": []}}, query="bowl",
            previous_candidate_index=0, maximum_world_displacement_m=.05)


def test_open_vocab_capability_loads_from_a_frozen_sibling_dependency(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "embodied_codex" / "capabilities"
    tool = tmp_path / "tool.py"
    dependency = tmp_path / "perception_reliability.py"
    tool.write_bytes((source_root / "open_vocab_rgbd.py").read_bytes())
    dependency.write_bytes((source_root / "perception_reliability.py").read_bytes())
    loaded = _load_class(tool, "OpenVocabularyRGBD",
                         relative_modules={"perception_reliability": dependency})
    assert loaded.__name__ == "OpenVocabularyRGBD"
