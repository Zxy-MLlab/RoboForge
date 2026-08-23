from __future__ import annotations

import hashlib

import pytest

from runtime_capabilities import (
    FrozenCapabilityRuntime,
    grasp_execution_waypoints,
    transport_waypoints,
)


PYTHON = "/data/zxy/envs/vla-report/bin/python"


def _binding(tmp_path, hook, source):
    module = tmp_path / f"{hook}.py"
    module.write_text(source)
    return {
        hook: {
            "tool_id": f"test_{hook}:v001",
            "module": module.name,
            "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        }
    }, module


def test_transport_profile_is_invoked_and_bounded(tmp_path):
    bindings, _ = _binding(
        tmp_path,
        "transport_profile",
        "def run(payload):\n"
        "    return {'lift_margin_m': 0.12, 'horizontal_segments': 4, "
        "'position_gain': 0.25, 'max_translation_action': 0.3}\n",
    )
    runtime = FrozenCapabilityRuntime(tmp_path, bindings, python=PYTHON)
    result = runtime.invoke("transport_profile", {"current_eef_xyz": [0, 0, 1]})
    assert result["applied"] is True
    assert result["output"]["horizontal_segments"] == 4
    assert result["output"]["position_gain"] == 0.25


def test_invalid_or_tampered_capability_falls_back(tmp_path):
    bindings, module = _binding(
        tmp_path,
        "transport_profile",
        "def run(payload):\n"
        "    return {'lift_margin_m': 9, 'horizontal_segments': 1, "
        "'position_gain': .3, 'max_translation_action': .3}\n",
    )
    runtime = FrozenCapabilityRuntime(tmp_path, bindings, python=PYTHON)
    assert runtime.invoke("transport_profile", {})["applied"] is False
    module.write_text("def run(payload):\n    return {}\n")
    result = runtime.invoke("transport_profile", {})
    assert result["applied"] is False
    assert "hash changed" in result["error"]


def test_grasp_ranking_cannot_escape_observed_candidates(tmp_path):
    bindings, _ = _binding(
        tmp_path,
        "grasp_retry_ranking",
        "def run(payload):\n    return {'candidate_indices': [2, 0]}\n",
    )
    runtime = FrozenCapabilityRuntime(tmp_path, bindings, python=PYTHON)
    valid = runtime.invoke(
        "grasp_retry_ranking", {"candidate_count": 3, "max_attempts": 2}
    )
    assert valid["applied"] is True
    assert valid["output"]["candidate_indices"] == [2, 0]
    invalid = runtime.invoke(
        "grasp_retry_ranking", {"candidate_count": 2, "max_attempts": 2}
    )
    assert invalid["applied"] is False


def test_transport_profile_changes_executed_route_and_control_gain():
    route = transport_waypoints(
        [0.0, 0.0, 1.0],
        [0.3, 0.0, 1.1],
        {
            "lift_margin_m": 0.12,
            "horizontal_segments": 3,
            "position_gain": 0.25,
            "max_translation_action": 0.3,
        },
    )
    assert route["lift"] == pytest.approx([0.0, 0.0, 1.22])
    assert len(route["horizontal"]) == 3
    assert route["horizontal"][0][0] == pytest.approx(0.1)
    assert route["horizontal"][-1] == pytest.approx([0.3, 0.0, 1.22])
    assert route["position_gain"] == 0.25


def test_support_relation_profile_is_conservative_and_bounded(tmp_path):
    bindings, _ = _binding(
        tmp_path,
        "support_relation_profile",
        "def run(payload):\n"
        "    return {'min_containment': .8, 'min_clearance_ratio': .9, "
        "'max_centroid_motion_px': 5., 'max_world_motion_m': .008, "
        "'max_xy_center_error_m': .018}\n",
    )
    runtime = FrozenCapabilityRuntime(tmp_path, bindings, python=PYTHON)
    result = runtime.invoke("support_relation_profile", {})
    assert result["applied"] is True
    assert result["output"]["min_clearance_ratio"] == 0.9

    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    unsafe_bindings, _ = _binding(
        unsafe_root,
        "support_relation_profile",
        "def run(payload):\n"
        "    return {'min_containment': .55, 'min_clearance_ratio': .25, "
        "'max_centroid_motion_px': 8., 'max_world_motion_m': .015, "
        "'max_xy_center_error_m': .025}\n",
    )
    unsafe = FrozenCapabilityRuntime(unsafe_root, unsafe_bindings, python=PYTHON)
    assert unsafe.invoke("support_relation_profile", {})["applied"] is False


def test_grasp_execution_profile_changes_real_waypoints_and_is_bounded(tmp_path):
    source = (
        "def run(payload):\n"
        "    return {'approach_clearance_m': .09, 'grasp_z_offset_m': -.01, "
        "'source_recenter_gain': .5, 'position_gain': .4, "
        "'max_translation_action': .4, 'close_steps': 55, "
        "'post_close_settle_steps': 12, 'lift_height_m': .14, "
        "'reobserve_before_attempt': True}\n"
    )
    bindings, _ = _binding(tmp_path, "grasp_execution_profile", source)
    runtime = FrozenCapabilityRuntime(tmp_path, bindings, python=PYTHON)
    event = runtime.invoke("grasp_execution_profile", {})
    assert event["applied"] is True
    route = grasp_execution_waypoints(
        [0.10, 0.20, 0.95],
        [0.08, 0.18, 0.94],
        [0.12, 0.16, 0.94],
        [0.0, 0.0, -1.0],
        event["output"],
        model_orientation=False,
    )
    assert route["grasp"] == pytest.approx([0.12, 0.19, 0.94])
    assert route["pregrasp"] == pytest.approx([0.12, 0.19, 1.03])
    assert route["lift"] == pytest.approx([0.12, 0.19, 1.08])
    assert route["profile"]["post_close_settle_steps"] == 12
