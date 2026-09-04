from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_codex.deployments.libero import (
    LiberoDeployment,
    LiberoDeploymentError,
    _public_execution_diagnostics,
)
from embodied_codex.kernel.runtime import ControllerRuntime


def _dispatch_deployment(tmp_path: Path) -> LiberoDeployment:
    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.capabilities = {}
    deployment.capability_contracts = {}
    deployment._native_capability_ids = frozenset()
    deployment.trace = [{"event": "formal_trace", "value": "immutable-prefix"}]
    deployment.step = 0
    deployment.verified_attachments = set()
    deployment.last_verify = False
    deployment.artifact_dir = tmp_path
    deployment.references = {}
    deployment._controller_artifact_paths = {}
    return deployment


@pytest.mark.parametrize(
    "method",
    ["_env", "reset", "reset_case", "check_success", "hidden_evaluator", "sim.data", "sim.model", "__class__"],
)
def test_sdk_allowlist_rejects_private_reflection_and_provider_methods(tmp_path, method):
    deployment = _dispatch_deployment(tmp_path)
    with pytest.raises(LiberoDeploymentError):
        deployment.dispatch("sdk", {"method": method, "args": [], "kwargs": {}})


def test_use_only_invokes_registered_candidate_capabilities(tmp_path):
    deployment = _dispatch_deployment(tmp_path)
    with pytest.raises(LiberoDeploymentError, match="unregistered Tool"):
        deployment.dispatch("use", {"tool_id": "candidate://not-in-bundle", "payload": {}})


def test_record_is_append_only_and_cannot_replace_formal_trace(tmp_path):
    deployment = _dispatch_deployment(tmp_path)
    result = deployment.dispatch("record", {
        "event": {"trace": [], "task_success": True, "hidden_evaluator": "forged"}
    })
    assert result == {"recorded": True}
    assert deployment.trace[0] == {"event": "formal_trace", "value": "immutable-prefix"}
    assert deployment.trace[1]["event"] == "controller_record"


def test_controller_sandbox_cannot_read_files_outside_frozen_bundle(tmp_path):
    workspace = tmp_path / "bundle"
    workspace.mkdir()
    secret = tmp_path / "verifier-key"
    secret.write_text("private")
    controller = workspace / "controller.py"
    controller.write_text(
        "def run(robot):\n"
        f"    path = {str(secret)!r}\n"
        "    try:\n"
        "        open(path).read()\n"
        "        return {'outside_read': True}\n"
        "    except OSError:\n"
        "        return {'outside_read': False}\n"
    )

    class Deployment:
        instruction = "security test"

    result = ControllerRuntime(timeout_seconds=10).execute(
        controller, Deployment(), source_root=workspace
    )
    assert result["completed"] is True
    assert result["result"] == {"outside_read": False}


def test_public_trace_drops_privileged_simulator_and_evaluator_fields():
    result = _public_execution_diagnostics({
        "completed": True,
        "rpc_events": [{
            "method": "sdk",
            "arguments": {"method": "get_observation", "args": [], "kwargs": {}},
            "result": {
                "method": "get_observation",
                "result": {
                    "public": 1,
                    "sim.data": "private",
                    "sim.model": "private",
                    "hidden_evaluator": "private",
                    "reward": 1,
                    "done": True,
                },
            },
        }],
    })
    encoded = json.dumps(result)
    assert '"public": 1' in encoded
    for private in ("sim.data", "sim.model", "hidden_evaluator", "reward", "done"):
        assert private not in encoded


def test_observable_condition_is_sensor_only_not_official_task_success(tmp_path):
    deployment = _dispatch_deployment(tmp_path)
    deployment.references = {"ref": {"world_xyz": [0.0, 0.0, 0.0]}}
    deployment.verifiers = {
        "visual_attachment": lambda _payload: {"verified": True, "sensor_only": True}
    }
    result = deployment.dispatch(
        "check_observable_condition",
        {"verifier": "visual_attachment", "payload": {
            "frame": {}, "object_query": "object", "source_ref": "ref"
        }},
    )
    assert result == {"verified": True, "sensor_only": True}
    assert "task_success" not in result
