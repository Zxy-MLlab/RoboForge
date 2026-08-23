import json
import hashlib
from types import SimpleNamespace

from libero_generated_controller import _sensor_summary, execute_controller


SPEC = {"stages": ["observe_rgbd", "verify_attachment", "verify_placement", "finish"]}


def test_sensor_summary_rejects_placement_without_transport(tmp_path):
    (tmp_path / "trace.json").write_text(json.dumps([{"phase": "place", "error": 0.0}]))
    result = {
        "protocol": "groundingdino-rgbd-closed-loop-v1",
        "attachment_verified": True,
        "placement_verification": {
            "verified": True,
            "post_transfer_attachment": {"verified": False},
        },
    }
    evidence = _sensor_summary(tmp_path, result, SPEC)
    assert evidence["transport_verified"] is False
    assert evidence["sensor_only_conclusion"] == "transport_not_verified"


def test_sensor_summary_accepts_verified_correction_transport(tmp_path):
    (tmp_path / "trace.json").write_text(json.dumps([{"phase": "place", "error": 0.0}]))
    result = {
        "protocol": "groundingdino-rgbd-closed-loop-v1",
        "attachment_verified": True,
        "placement_verification": {
            "verified": False,
            "post_transfer_attachment": {"verified": False},
            "correction_attempts": [{
                "verified_placement": True,
                "post_transfer_attachment": {"verified": True},
            }],
        },
    }
    evidence = _sensor_summary(tmp_path, result, SPEC)
    assert evidence["transport_verified"] is True
    assert evidence["sensor_only_conclusion"] == "sensor_verification_passed"


def test_sensor_summary_rejects_legacy_rim_contact_label(tmp_path):
    (tmp_path / "trace.json").write_text(json.dumps([{"phase": "place", "error": 0.0}]))
    metrics = {
        "containment": 0.653,
        "clearance_ratio": 0.371,
        "centroid_x": 395.0,
        "centroid_y": 346.0,
    }
    result = {
        "protocol": "groundingdino-rgbd-closed-loop-v1",
        "attachment_verified": True,
        "placement_verification": {
            "verified": True,
            "first": {"mask_metrics": metrics, "height_ok": True},
            "second": {"mask_metrics": metrics, "height_ok": True},
            "world_motion_m": 0.0,
            "first_xy_center_error_m": 0.014,
            "second_xy_center_error_m": 0.014,
            "post_transfer_attachment": {"verified": True},
        },
    }
    evidence = _sensor_summary(tmp_path, result, SPEC)
    assert evidence["transport_verified"] is True
    assert evidence["support_relation_recheck"] is False
    assert evidence["placement_verified"] is False
    assert evidence["sensor_only_conclusion"] == "placement_not_verified"


def test_execute_controller_passes_frozen_capability_to_backend(tmp_path, monkeypatch):
    capability = tmp_path / "transport.py"
    capability.write_text(
        "def run(payload):\n"
        "    return {'lift_margin_m': .12, 'horizontal_segments': 3, "
        "'position_gain': .25, 'max_translation_action': .3}\n"
    )
    binding = {
        "transport_profile": {
            "tool_id": "gentle_transport:v001",
            "module": capability.name,
            "sha256": hashlib.sha256(capability.read_bytes()).hexdigest(),
        }
    }
    captured = {}

    def run_one_closed_loop(task, **kwargs):
        del task
        produced = kwargs["output_root"] / "fake"
        produced.mkdir(parents=True)
        event = kwargs["capability_hook"](
            "transport_profile", {"current_eef_xyz": [0, 0, 1]}
        )
        captured.update(event)
        (produced / "trace.json").write_text(json.dumps([
            {"phase": "approach", "error": 0.01},
            {"event": "capability_hook", **event},
        ]))
        return {
            "protocol": "groundingdino-rgbd-closed-loop-v1",
            "success": False,
            "attachment_verified": False,
            "capability_hook_invocations": [event],
        }

    backend = SimpleNamespace(run_one_closed_loop=run_one_closed_loop)
    monkeypatch.setattr("libero_generated_controller._load_backend", lambda: backend)
    output = tmp_path / "run"
    execute_controller(
        {
            "stages": ["observe_rgbd", "verify_attachment", "finish"],
            "runtime_capability_hooks": binding,
        },
        suite="libero_spatial",
        task=0,
        state=0,
        seed=7,
        output=output,
        controller_root=tmp_path,
    )
    observation = json.loads((output / "agent_observation.json").read_text())
    assert captured["applied"] is True
    assert captured["output"]["horizontal_segments"] == 3
    assert observation["capability_hook_invocations"][0]["tool_id"] == "gentle_transport:v001"
    assert json.loads((output / "_evaluator_only" / "result.json").read_text())["success"] is False
