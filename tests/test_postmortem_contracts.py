import json
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from embodied_codex.adapters.libero import _perception_contract
from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
from embodied_codex.capabilities.open_vocab_rgbd import OpenVocabularyRGBD
from embodied_codex.deployments.libero import (
    LiberoDeployment,
    _public_execution_diagnostics,
)
from embodied_codex.legacy.campaign import CampaignAdapter


def _canonical_frame(tmp_path):
    return {
        "frame_id": "frame-synthetic",
        "step": 3,
        "cameras": {
            "agentview": {
                "rgb_path": str(tmp_path / "rgb.png"),
                "rgb_sha256": "a" * 64,
                "depth_path": str(tmp_path / "depth.npy"),
                "depth_sha256": "b" * 64,
                "shape": [2, 2, 3],
                "depth_range_m": [0.1, 2.0],
                "intrinsic": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1]],
            }
        },
        "proprioception": {
            "eef_pose": {"frame": "world", "position_m": [0.1, 0.2, 0.3],
                          "orientation_xyzw": [0, 0, 0, 1]},
            "gripper": {"width_m": 0.04},
            "joint_state": {"position": [0.0], "velocity": [0.0],
                            "gripper_velocity": [0.0]},
            "proprioception": {"joint_position": [0.0], "joint_velocity": [0.0]},
        },
    }


def test_canonical_frame_is_accepted_by_public_perception_contract(tmp_path):
    frame = _canonical_frame(tmp_path)
    contract = _perception_contract()["input_schema"]
    Draft202012Validator(contract).validate({"frame": frame, "queries": ["<object>"]})


def test_attachment_verifier_consumes_canonical_proprioception_at_adapter_boundary(tmp_path):
    verifier = OpenVocabularyRGBD.__new__(OpenVocabularyRGBD)
    verifier.detect = lambda payload: {"detections": {
        "<object>": [{"world_xyz": [0.1, 0.2, 0.3]}]}}
    frame = _canonical_frame(tmp_path)
    result = verifier.verify_attachment({
        "frame": frame, "object_query": "<object>",
        "source_world_xyz": [0.0, 0.0, 0.0],
    })
    assert result["verified"] is True


def test_outcome_verifier_receives_complete_generic_before_after_payload(tmp_path):
    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False
    deployment.episode = SimpleNamespace(case_handle=None)
    deployment.outcome_verifier = lambda payload: {
        "verified": isinstance(payload.get("instruction"), str)
        and bool(payload["instruction"])
        and isinstance(payload.get("before"), dict)
        and isinstance(payload.get("after"), dict)
    }
    deployment._outcome_report = None
    deployment._execution_sensor_report = None
    deployment.trace = []
    deployment.last_verify = False
    deployment._execution_artifacts = {}
    deployment._outcome_before = {"rgb_path": "before", "rgb_sha256": "a"}
    deployment._outcome_after = {"rgb_path": "after", "rgb_sha256": "b"}
    deployment._instruction = "<generic instruction>"
    deployment._capture_outcome_rgb = lambda name: deployment._outcome_after
    deployment._finalize_execution_artifacts = lambda: None
    deployment.canonical_embodied_state = lambda: {"robot": {}}
    deployment.step = 1
    report = deployment.sensor_report({"completed": True})
    assert report["independent_task_outcome"]["verified"] is True


def test_public_execution_diagnostics_exposes_tool_contract_failure():
    execution = {
        "completed": True,
        "error": None,
        "result": {
            "tool_id": "libero.rgbd_perception:v001",
            "step": 12,
            "result": {
                "ok": False,
                "tool_error": {
                    "type": "ToolContractError",
                    "message": "20 is greater than the maximum of 12",
                },
            },
        },
        "rpc_events": [
            {
                "method": "use",
                "arguments": {
                    "tool_id": "libero.rgbd_perception:v001",
                    "payload": {"max_detections_per_query": 20},
                },
                "result": {
                    "tool_id": "libero.rgbd_perception:v001",
                    "step": 12,
                    "result": {
                        "ok": False,
                        "tool_error": {
                            "type": "ToolContractError",
                            "message": "20 is greater than the maximum of 12",
                        },
                    },
                },
            }
        ],
    }

    public = _public_execution_diagnostics(execution)

    assert public["controller_termination"] == "completed"
    assert public["tool_errors"] == [
        {
            "index": 0,
            "tool_id": "libero.rgbd_perception:v001",
            "step": 12,
            "type": "ToolContractError",
            "message": "20 is greater than the maximum of 12",
        }
    ]
    assert public["controller_result"] == execution["result"]


def test_public_execution_diagnostics_preserves_actions_but_not_privileged_state():
    execution = {
        "completed": False,
        "error": "RuntimeError: failed at /root/private/controller.py token=abc",
        "stderr": "secret: value from /tmp/private.log",
        "rpc_events": [
            {
                "method": "act",
                "arguments": {
                    "action": {
                        "type": "move_to_point",
                        "position_m": [0.1, 0.2, 0.3],
                    }
                },
                "result": {
                    "type": "move_to_point",
                    "step": 23,
                    "reached": False,
                },
                "state_before": {
                    "robot": {"gripper": {"width_m": 0.07}},
                    "reward": 99,
                    "case_handle": "sealed-case",
                },
                "state_after": {
                    "robot": {"gripper": {"width_m": 0.06}},
                    "hidden_evaluator": True,
                },
            }
        ],
    }

    public = _public_execution_diagnostics(execution)
    encoded = json.dumps(public, sort_keys=True)

    assert public["controller_termination"] == "controller_error"
    assert public["action_trace"][0]["requested"]["position_m"] == [0.1, 0.2, 0.3]
    assert public["action_trace"][0]["result"]["reached"] is False
    assert public["action_trace"][0]["state_before"]["robot"]["gripper"] == {
        "width_m": 0.07
    }
    assert "/root/private" not in encoded and "/tmp/private" not in encoded
    for private in ("reward", "case_handle", "hidden_evaluator", "abc", "value"):
        assert private not in encoded


def test_campaign_delegates_execution_boundary_for_outcome_capture():
    class Adapter:
        instruction = "<generic instruction>"
        def __init__(self):
            self.started = 0
        def begin_controller_execution(self):
            self.started += 1

    active = Adapter()
    campaign = CampaignAdapter([("synthetic", active)])
    campaign.begin_controller_execution()
    assert active.started == 1
