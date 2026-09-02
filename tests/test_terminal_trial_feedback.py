import json
from pathlib import Path

import pytest

from roboforge.evidence import derive_status, extract_first_error
from roboforge.fakes import FakeAdapter
from roboforge.models import AdapterResult, RawArtifact
from roboforge.preflight import preflight_controller
from roboforge.service import ExperimentService, ProtocolError
from roboforge.trial_artifacts import materialize_trial


PERCEPTION = {
    "input_schema": {
        "type": "object",
        "properties": {
            "frame": {"type": "object"},
            "queries": {"type": "array"},
            "max_detections_per_query": {
                "type": "integer", "minimum": 1, "maximum": 12,
            },
        },
        "required": ["frame", "queries"],
        "additionalProperties": False,
    }
}


def test_contract_preflight_rejects_literal_field_but_accepts_dynamic_frame(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text(
        "def run(robot):\n"
        "    frame = robot.observe()\n"
        "    return robot.use('perception:v1', {'frame': frame, 'queries': ['bowl'], "
        "'max_detections_per_query': 20})\n"
    )
    report = preflight_controller(controller, capability_contracts={"perception:v1": PERCEPTION})
    assert report["ok"] is False
    assert report["errors"][0]["error_type"] == "ToolContractError"
    assert report["errors"][0]["message"] == "20 is greater than the maximum of 12"


def test_preflight_failure_does_not_consume_physical_trial(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return robot.use('bad:v1', {'limit': 20})\n")
    adapter = FakeAdapter()
    adapter.preflight = lambda **_: {"ok": False, "errors": [{"message": "invalid"}]}
    service = ExperimentService(tmp_path / "run", adapter)
    with pytest.raises(ProtocolError, match="contract preflight failed"):
        service.run_controller(request_id="bad", controller_path=controller, intent="invalid")
    assert service.status()["physical_trials"] == 0
    assert adapter.reset_count == adapter.controller_runs == 0


def test_first_error_is_generic_and_prefers_earliest_tool_error():
    public = {"tool_errors": [
        {"tool_id": "perception:v1", "step": 12, "type": "ToolContractError", "message": "20 > 12"},
        {"tool_id": "grasp:v1", "step": 13, "type": "RuntimeError", "message": "later"},
    ]}
    assert extract_first_error(public) == {
        "path": "$.tool_errors[0]", "error_type": "ToolContractError",
        "message": "20 > 12", "api": "perception:v1", "step": 12,
    }


class ErrorAdapter(FakeAdapter):
    def execute_controller(self, **kwargs):
        return AdapterResult(
            public={
                "controller_termination": "completed",
                "tool_errors": [{"tool_id": "perception:v1", "step": 12,
                                 "type": "ToolContractError", "message": "20 > 12"}],
                "sanitized_trace": [{"event": "use", "step": 12,
                                     "tool_error": {"type": "ToolContractError", "message": "20 > 12"}}],
                "action_trace": [],
            },
            artifacts=(RawArtifact("rollout.mp4", "video/mp4", b"video"),),
            private_receipt={"kind": "physical", "controller_sha256": kwargs["controller_sha256"],
                             "environment_generation": kwargs["environment_generation"], "verified": False},
        )


def test_trial_materialization_exposes_sanitized_files_and_nonzero_status(tmp_path):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    controller = workspace / "controller.py"; controller.write_text("def run(robot): return {}\n")
    service = ExperimentService(tmp_path / "run", ErrorAdapter())
    evidence = service.run_controller(request_id="trial", controller_path=controller, intent="regression")
    result = materialize_trial(service, evidence, workspace, controller_path=controller)
    trial = workspace / ".roboforge" / "trials" / "physical-000001"
    expected = {"manifest.json", "frozen_source", "trace.json", "first_error.json",
                "action_receipts.jsonl", "keyframes", "rollout.mp4", "stdout.log",
                "stderr.log", "result.json"}
    assert expected.issubset({item.name for item in trial.iterdir()})
    assert result["runner_exit_code"] == 1
    assert result["controller_status"] == "error"
    assert json.loads((trial / "first_error.json").read_text())["message"] == "20 > 12"
    encoded = (trial / "trace.json").read_text()
    for private in ("sim.data", "sim.model", "hidden_evaluator", "promotion key"):
        assert private not in encoded


def test_runner_controller_environment_and_task_status_are_distinct():
    status = derive_status({"controller_termination": "completed", "task_success": False})
    assert status == {"runner_exit_code": 0, "controller_status": "completed",
                      "environment_status": "ok", "task_success": False,
                      "termination_reason": "completed", "first_error": None,
                      "trial_status": "task_failed"}
