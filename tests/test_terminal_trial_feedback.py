import json
import hashlib
from pathlib import Path

import pytest

from roboforge.evidence import derive_status, extract_first_error
from roboforge.cli import (
    _conversation_termination_reason,
    _public_status_fallback,
    _write_campaign_result,
)
from roboforge.fakes import FakeAdapter
from roboforge.models import AdapterResult, RawArtifact
from roboforge.preflight import preflight_controller
from roboforge.service import ExperimentService, ProtocolError
from roboforge.stop_gate import campaign_stop_decision
from roboforge.trial_artifacts import materialize_preflight_failure, materialize_trial


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


def test_campaign_stop_requires_changed_candidate_with_matching_valid_trial(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return robot.observe()\n")
    baseline = hashlib.sha256(controller.read_bytes()).hexdigest()
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"records": []}))
    assert campaign_stop_decision(ledger, controller, baseline)["decision"] == "deny"

    controller.write_text("def run(robot): return robot.get_task_language()\n")
    candidate = hashlib.sha256(controller.read_bytes()).hexdigest()
    ledger.write_text(json.dumps({"records": [{
        "controller_sha256": candidate, "valid_trial": True,
    }]}))
    assert campaign_stop_decision(ledger, controller, baseline)["decision"] == "allow"


def test_campaign_stop_rejects_stale_or_invalid_candidate_trial(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("candidate\n")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"records": [{
        "controller_sha256": "0" * 64, "valid_trial": True,
    }, {
        "controller_sha256": hashlib.sha256(controller.read_bytes()).hexdigest(),
        "valid_trial": False,
    }]}))
    decision = campaign_stop_decision(ledger, controller, "f" * 64)
    assert decision["decision"] == "deny"
    assert "exact source digest" in decision["additionalContext"]


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


class ResetFailureAdapter(FakeAdapter):
    def reset_to_s0(self):
        self.reset_count += 1
        raise RuntimeError("simulator reset failed")


def test_environment_failure_is_evidenced_but_does_not_consume_task_budget(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    service = ExperimentService(tmp_path / "run", ResetFailureAdapter(), max_trials=1)

    evidence = service.run_controller(
        request_id="reset-failure", controller_path=controller, intent="exercise reset"
    )

    lifecycle = evidence.public["lifecycle"]
    assert lifecycle["failure_class"] == "environment_failure"
    assert lifecycle["task_budget_consumed"] is False
    assert lifecycle["controller_started"] is False
    assert service.status()["physical_attempts"] == 1
    assert service.status()["physical_trials"] == 0


def test_controller_failure_consumes_task_budget_after_episode_start(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    adapter = FakeAdapter()
    adapter.raise_during_execution = RuntimeError("controller process failed")
    service = ExperimentService(tmp_path / "run", adapter, max_trials=1)

    evidence = service.run_controller(
        request_id="controller-failure", controller_path=controller, intent="execute"
    )

    lifecycle = evidence.public["lifecycle"]
    assert lifecycle["failure_class"] == "controller_failure"
    assert lifecycle["task_budget_consumed"] is True
    assert lifecycle["controller_started"] is True
    assert service.status()["physical_attempts"] == 1
    assert service.status()["physical_trials"] == 1


def test_preflight_artifact_captures_terminal_stdout(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return robot.use('bad:v1', {'limit': 20})\n")
    result = materialize_preflight_failure(
        tmp_path / "workspace",
        {"ok": False, "errors": [{"api": "robot.use", "message": "20 > 12"}]},
        controller_path=controller,
    )
    trial = tmp_path / "workspace" / ".roboforge" / "trials" / result["trial_id"]
    assert json.loads((trial / "stdout.log").read_text()) == result


def test_first_error_is_generic_and_prefers_earliest_tool_error():
    public = {"tool_errors": [
        {"tool_id": "perception:v1", "step": 12, "type": "ToolContractError", "message": "20 > 12"},
        {"tool_id": "grasp:v1", "step": 13, "type": "RuntimeError", "message": "later"},
    ]}
    assert extract_first_error(public) == {
        "path": "$.tool_errors[0]", "error_type": "ToolContractError",
        "message": "20 > 12", "api": "perception:v1", "step": 12,
    }


def test_first_error_is_ordered_by_step_not_container_order():
    public = {"tool_errors": [
        {"tool_id": "grasp:v1", "step": 13, "type": "RuntimeError", "message": "later"},
        {"tool_id": "perception:v1", "step": 12, "type": "ToolContractError", "message": "20 > 12"},
    ]}
    assert extract_first_error(public)["api"] == "perception:v1"
    assert extract_first_error(public)["step"] == 12


def test_first_error_prefers_rpc_event_index_when_steps_are_unavailable():
    public = {"tool_errors": [
        {"index": 9, "tool_id": "grasp:v1", "type": "RuntimeError", "message": "later"},
        {"index": 2, "tool_id": "perception:v1", "type": "ToolContractError", "message": "first"},
    ]}
    assert extract_first_error(public)["api"] == "perception:v1"


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
                "action_receipts.jsonl", "keyframes", "stdout.log",
                "stderr.log", "result.json"}
    assert expected.issubset({item.name for item in trial.iterdir()})
    assert result["runner_exit_code"] == 1
    assert result["controller_status"] == "error"
    assert json.loads((trial / "first_error.json").read_text())["message"] == "20 > 12"
    assert json.loads((trial / "stdout.log").read_text()) == result
    manifest = json.loads((trial / "manifest.json").read_text())
    assert manifest["artifact_availability"]["rollout_mp4"] is True
    encoded = (trial / "trace.json").read_text()
    for private in ("sim.data", "sim.model", "hidden_evaluator", "promotion key"):
        assert private not in encoded


def test_runner_controller_environment_and_task_status_are_distinct():
    status = derive_status({"controller_termination": "completed", "task_success": False})
    assert status == {"runner_exit_code": 0, "controller_status": "completed",
                      "environment_status": "ok", "task_success": False,
                      "termination_reason": "completed", "first_error": None,
                      "trial_status": "task_failed"}


def test_campaign_result_records_openhands_failure_and_current_trial_count(tmp_path):
    campaign = _write_campaign_result(
        tmp_path,
        status={"physical_trials": 13, "max_trials": 15},
        elapsed=42.0,
        max_iterations=80,
        wall_time_budget=14400,
        latest_verified=False,
        run_error="ConversationRunError: provider unavailable",
    )
    assert campaign["termination_reason"] == "openhands_run_error"
    assert campaign["physical_trials"] == 13
    assert json.loads(
        (tmp_path / ".roboforge" / "campaign-result.json").read_text()
    ) == campaign


def test_campaign_result_distinguishes_agent_finish_from_iteration_budget(tmp_path):
    from types import SimpleNamespace

    finished = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(value="finished"), events=[]))
    limited = SimpleNamespace(state=SimpleNamespace(
        execution_status=SimpleNamespace(value="error"),
        events=[SimpleNamespace(code="MaxIterationsReached")]))
    assert _conversation_termination_reason(finished) == "agent_finished"
    assert _conversation_termination_reason(limited) == "openhands_iteration_budget_exhausted"
    campaign = _write_campaign_result(
        tmp_path,
        status={"physical_trials": 1, "max_trials": 15},
        elapsed=42.0,
        max_iterations=80,
        wall_time_budget=14400,
        latest_verified=False,
        conversation_reason="agent_finished",
    )
    assert campaign["termination_reason"] == "agent_finished"


def test_campaign_result_records_operator_interrupt(tmp_path):
    campaign = _write_campaign_result(
        tmp_path,
        status={"physical_trials": 2, "max_trials": 15},
        elapsed=12.0,
        max_iterations=80,
        wall_time_budget=14400,
        latest_verified=False,
        run_error="KeyboardInterrupt: interrupted by operator",
    )
    assert campaign["termination_reason"] == "user_interrupted"


def test_interrupted_campaign_recovers_public_trial_count(tmp_path):
    status = tmp_path / ".roboforge" / "campaign-status.json"
    status.parent.mkdir()
    status.write_text(json.dumps({
        "physical_trials": 4,
        "max_physical_trials": 15,
        "latest_physical_evidence": "experiment://physical-000004",
    }))
    assert _public_status_fallback(tmp_path, 15) == {
        "physical_trials": 4,
        "max_trials": 15,
        "latest_physical_evidence": "experiment://physical-000004",
    }
