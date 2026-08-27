import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget, ProtocolError
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.context_window import ContextWindowManager, ResourceBudgets
from embodied_codex.kernel.events import EventStore, EventStoreError
from embodied_codex.kernel.evidence import AgentEvidence, build_execution_digest
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace
from embodied_codex.kernel.sandbox import UnsafeSandboxBackend


def _call(name, arguments, identifier="call"):
    return {"content": "", "tool_calls": [{"id": identifier, "name": name,
            "arguments": json.dumps(arguments)}]}


def _loop(tmp_path, model, *, budget=None, adapter=None, resume=True,
          context_window=None):
    adapter = adapter or FakeAdapter("set marker", tmp_path / "adapter")
    workspace = PersistentWorkspace(tmp_path / "run/workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=adapter)
    loop = AgentLoop(model=model, workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace,
            initial_observation=adapter.initial_observation()),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path / "run/events"), root=tmp_path / "run",
        budget=budget or LoopBudget(max_steps=8, max_executions=4), resume=resume,
        context_window=context_window)
    return loop


def test_event_append_uses_tail_metadata_without_history_scan_and_enforces_quota(tmp_path,
                                                                                monkeypatch):
    store = EventStore(tmp_path / "events", max_bytes=512 * 1024,
                       max_record_bytes=256 * 1024)
    scans = 0
    original = store._read_locked

    def counted(*args, **kwargs):
        nonlocal scans
        scans += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "_read_locked", counted)
    for index in range(1000):
        store.commit("sample", {"index": index})
    assert scans == 0
    assert store.tail_path.is_file()
    assert len(store.events()) == 1000
    with pytest.raises(EventStoreError, match="quota"):
        store.commit("oversized", {"value": "x" * 300_000})


class _LargeEvidenceAdapter(FakeAdapter):
    def sensor_report(self, execution):
        return {"diagnostic": "x" * 300_000, "action_log": [
            {"step": index, "payload": "y" * 1000} for index in range(100)]}

    def verification_receipt(self, execution):
        receipt = super().verification_receipt(execution)
        receipt["verified"] = execution.get("completed") is True
        return receipt


def test_execution_evidence_has_one_full_copy_and_checkpoint_is_bounded(tmp_path):
    class Model:
        turn = 0
        def decide(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return _call("record_decision", {"goal": "repair", "evidence_refs": [],
                    "hypothesis": "public execution is not verified", "decision": "update controller",
                    "expected_effect": "verification succeeds", "uncertainty": None}, "decision")
            if self.turn == 2:
                return _call("write_file", {"path": "controller.py", "content":
                    "def run(robot):\n return robot.verify('target', {})\n"}, "write")
            if self.turn == 3:
                return _call("run_controller", {}, "run")
            return _call("finish", {"summary": "verified"}, "finish")

    loop = _loop(tmp_path, Model(), adapter=_LargeEvidenceAdapter(
        "large evidence", tmp_path / "adapter"))
    result = loop.run("large evidence")
    assert result["finished"] is True
    evidence_files = list((tmp_path / "run/evidence").glob("execution-*.json"))
    assert len(evidence_files) == 1
    event = next(row for row in loop.event_store.events() if row["kind"] == "execution")
    assert "execution" not in event["payload"] and "sensor_report" not in event["payload"]
    assert event["payload"]["artifact_uri"].startswith("run://evidence/")
    assert event["payload"]["artifact_sha256"] == hashlib.sha256(
        evidence_files[0].read_bytes()).hexdigest()
    checkpoint = tmp_path / "run/checkpoint/state.json"
    assert checkpoint.stat().st_size < 64 * 1024
    assert "x" * 1000 not in checkpoint.read_text()


def test_context_bounds_fixed_fields_tool_calls_and_multimodal_inputs(tmp_path, monkeypatch):
    budgets = ResourceBudgets(max_task_chars=100, max_adapter_chars=200,
        max_state_chars=200, max_assets_chars=200, max_evidence_chars=300,
        max_context_chars=4000, max_tool_calls_per_turn=2,
        max_image_bytes=1024, max_image_pixels=100, max_images_per_turn=1)
    window = ContextWindowManager(budgets=budgets)
    context = {"system": "system", "task": "t" * 1000,
        "adapter": {"sdk": "a" * 1000}, "workspace": [],
        "assets": {"items": "b" * 1000}, "latest_evidence": {"log": "e" * 1000},
        "state": {"research": "s" * 1000}}
    bounded = window.bound_context(context, artifact_root=tmp_path / "artifacts")
    assert len(json.dumps(bounded)) <= budgets.max_context_chars
    for key in ("task", "adapter", "assets", "latest_evidence", "state"):
        assert bounded[key]["truncated"] is True
        assert bounded[key]["artifact_uri"].startswith("run://artifacts/context/")

    class ManyCalls:
        def __init__(self):
            self.outputs = []
        def decide(self, **_kwargs):
            return {"content": "", "tool_calls": [
                {"id": str(index), "name": "list_files", "arguments": "{}"}
                for index in range(5)]}
        def record_tool_output(self, call_id, output, **metadata):
            self.outputs.append((call_id, json.loads(output), metadata))
    many_calls = ManyCalls()
    loop = _loop(tmp_path / "calls", many_calls,
                 budget=LoopBudget(max_steps=1, max_executions=1),
                 context_window=window)
    result = loop.run("bounded")
    tool_results = [row for row in loop.event_store.events() if row["kind"] == "tool_result"]
    assert len(tool_results) == 5
    assert [row["payload"]["skipped"] for row in tool_results] == [
        False, False, True, True, True]
    assert [call_id for call_id, _output, _metadata in many_calls.outputs] == [
        "0", "1", "2", "3", "4"]
    assert all(output["ok"] is False and metadata["failed"] is True
               for _call_id, output, metadata in many_calls.outputs[2:])
    assert result["resumable"] is True

    image = loop.workspace.root / "large.png"
    image.write_bytes(b"not-an-image" * 1000)
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (_ for _ in ()).throw(
        AssertionError("large image was read before stat")) if path == image else original(path))
    with pytest.raises(ProtocolError, match="size limit"):
        loop._view_artifact("large.png")


def test_execution_digest_is_compact_public_rpc_facts_only():
    execution = {
        "completed": True, "error": None, "program_sha256": "sha-a",
        "result": {"public": "return"},
        "rpc_events": [
            {"method": "use", "arguments": {"tool_id": "tool:v001",
             "payload": {"frame": {"rgb_path": "artifact://agent/rgb.png"}}},
             "result": {"tool_id": "tool:v001", "result": {"ok": True}}},
            {"method": "act", "arguments": {"action": {"type": "move_to_pose",
             "target_xyz": [1, 2, 3]}}, "result": {"type": "move_to_pose",
             "reached": False, "target_xyz": [1, 2, 3], "eef_after": [0, 0, 0],
             "final_position_error_m": 0.2, "final_orientation_error_rad": 0.4,
             "gripper_qpos": [0.1, 0.1]}},
            {"method": "verify", "arguments": {"verifier": "target"},
             "result": {"verified": False, "reason": "not attached"}},
        ],
    }
    digest = build_execution_digest(execution, controller_sha256="sha-a",
                                    diagnostics={"rollout_path": "artifact://agent/rollout.mp4"})
    assert digest["execution"] == {"completed": True, "error": None,
                                    "controller_sha256": "sha-a"}
    assert digest["tool_calls"][0]["tool_id"] == "tool:v001"
    assert digest["tool_calls"][0]["status"] == "success"
    assert digest["actions"][0]["result"]["reached"] is False
    assert digest["actions"][0]["requested"]["target_xyz"] == [1, 2, 3]
    assert digest["verifications"] == [{"verifier": "target", "verified": False,
                                         "reason": "not attached"}]
    assert digest["artifacts"]["rgb"] == ["artifact://agent/rgb.png"]
    assert digest["artifacts"]["rollout"] == "artifact://agent/rollout.mp4"
    assert not any(key in json.dumps(digest) for key in
                   ("reward", "done", "check_success", "hidden_evaluator"))


def test_agent_evidence_legacy_positional_reference_is_preserved():
    evidence = AgentEvidence.from_execution(
        {"completed": True, "error": None, "rpc_events": []}, {}, "evidence://legacy")
    assert evidence.evidence_ref == "evidence://legacy"
    assert evidence.digest["execution"]["completed"] is True


def test_run_controller_agent_evidence_contains_execution_digest(tmp_path):
    loop = _loop(tmp_path, object(), resume=False)
    loop.workspace.write_file("controller.py", "def run(robot):\n"
                             "    robot.act({'type': 'set_value', 'value': 0})\n"
                             "    return robot.verify('target', {})\n")
    evidence = loop._run_controller()
    digest = evidence["agent_evidence"]["digest"]
    assert digest["actions"][0]["type"] == "set_value"
    assert digest["actions"][0]["requested"]["value"] == 0
    assert digest["verifications"][0]["verified"] is False
    assert "rpc_events" not in json.dumps(evidence["agent_evidence"])
    action_event = next(event for event in evidence["execution"]["rpc_events"]
                        if event["method"] == "act")
    assert "state_before" in action_event and "state_after" in action_event
    assert action_event["state_before"]["robot"]["proprioception"]["value"] == 0
    assert action_event["state_after"]["robot"]["proprioception"]["value"] == 0


def test_execution_artifact_handles_resolve_to_immutable_scoped_snapshots(tmp_path):
    adapter = FakeAdapter("set marker", tmp_path / "adapter")
    loop = _loop(tmp_path, object(), adapter=adapter, resume=False)
    source = adapter.artifact_dir / "reused.bin"
    source.write_bytes(b"execution-one")

    loop._artifact_scope = "execution-one"
    first = loop._register_artifacts({"artifact": "artifact://adapter/reused.bin"})["artifact"]
    first_path = loop._artifact_handles[first]
    assert first.startswith("artifact://agent/")
    assert first_path.read_bytes() == b"execution-one"

    source.write_bytes(b"execution-two")
    loop._artifact_scope = "execution-two"
    second = loop._register_artifacts({"artifact": "artifact://adapter/reused.bin"})["artifact"]
    second_path = loop._artifact_handles[second]
    assert second != first
    assert second_path.read_bytes() == b"execution-two"
    assert first_path.read_bytes() == b"execution-one"
    assert str(first_path) not in json.dumps({"artifact": first})
    assert str(second_path) not in json.dumps({"artifact": second})
    assert first_path.parent != second_path.parent


def test_artifact_manifest_restores_opaque_handle_after_new_loop(tmp_path):
    adapter = FakeAdapter("artifact", tmp_path / "adapter")
    first = _loop(tmp_path, object(), adapter=adapter, resume=False)
    source = adapter.artifact_dir / "evidence.txt"
    source.write_text("public evidence")
    first._artifact_scope = "execution"
    handle = first._register_artifacts({"ref": f"artifact://adapter/{source.name}"})["ref"]
    manifest = tmp_path / "run" / "artifacts" / "manifest.json"
    assert manifest.is_file()
    second = _loop(tmp_path, object(), adapter=adapter, resume=False)
    assert handle in second._artifact_handles
    assert second._view_artifact(handle)["content"] == "public evidence"
    second._artifact_handles[handle].chmod(0o644)
    second._artifact_handles[handle].write_text("tampered")
    with pytest.raises(ProtocolError, match="checksum"):
        second._view_artifact(handle)


def test_decision_is_consumed_by_one_consequential_operation(tmp_path):
    loop = _loop(tmp_path, object(), resume=False)
    loop._active_tool_call_id = "decision"
    loop._record_decision(goal="change", evidence_refs=[], hypothesis=None,
                          decision="write", expected_effect=None, uncertainty=None)
    loop._active_tool_call_id = "write"
    assert loop._claim_decision("write_file") == "decision-decision"
    with pytest.raises(ProtocolError, match="current open Decision Record"):
        loop._claim_decision("run_controller")
    stored = loop._list_decisions()["decisions"][0]
    assert stored["status"] == "committed"
    assert stored["linked_call_ids"] == ["write"]


def test_context_includes_bounded_execution_digest_without_rpc_event_log():
    digest = build_execution_digest({"completed": True, "error": None,
        "program_sha256": "sha", "rpc_events": [
            {"method": "act", "arguments": {"action": {"type": "gripper",
             "command": "close"}}, "result": {"type": "gripper", "reached": True,
             "gripper_qpos": [0.0, 0.0]}}
        ]})
    evidence = AgentEvidence(execution={"completed": True, "error": None},
                             diagnostics={}, digest=digest, evidence_ref="evidence://1")
    context = ContextBuilder(adapter_index={}, asset_registry=None,
                             workspace=None).build(task="task", latest_evidence=evidence)
    latest = context["latest_evidence"]
    assert latest["digest"]["actions"][0]["result"]["reached"] is True
    assert "rpc_events" not in json.dumps(latest)


def test_digest_long_lists_preserve_head_and_tail_in_model_context(tmp_path):
    actions = [{"index": index, "type": f"action-{index}"} for index in range(40)]
    digest = {"execution": {"completed": True, "error": None,
                             "controller_sha256": "sha"},
              "tool_calls": [], "actions": actions, "verifications": [],
              "artifacts": {"rgb": ["artifact://agent/rgb"], "depth": [],
                            "trace": None, "rollout": None}}
    evidence = AgentEvidence(execution={"completed": True, "error": None},
                             diagnostics={}, digest=digest, evidence_ref="evidence://40")
    context = ContextBuilder(adapter_index={}, asset_registry=None,
                             workspace=None).build(task="task", latest_evidence=evidence)
    bounded = ContextWindowManager().bound_context(context, artifact_root=tmp_path / "artifacts")
    actions_view = bounded["latest_evidence"]["digest"]["actions"]
    assert actions_view["total_count"] == 40
    assert actions_view["omitted_count"] == 24
    assert actions_view["head"][0]["type"] == "action-0"
    assert actions_view["tail"][-1]["type"] == "action-39"
    assert "rpc_events" not in json.dumps(bounded)


def test_oversized_digest_remains_structured_under_evidence_budget(tmp_path):
    digest = {"execution": {"completed": True, "error": None,
                             "controller_sha256": "sha"},
              "controller_result": {"output": "x" * 100_000},
              "tool_calls": [{"tool_id": "tool:v1", "output_summary": "y" * 100_000}],
              "actions": [{"index": 1, "type": "move", "result": {"reached": False}}],
              "verifications": [{"verifier": "v", "verified": False}],
              "artifacts": {"rgb": ["artifact://agent/rgb"], "depth": [],
                            "trace": "artifact://agent/trace", "rollout": None}}
    evidence = {"execution": {"completed": True, "error": None},
                "diagnostics": {}, "digest": digest,
                "evidence_ref": "evidence://large"}
    context = {"system": "system", "task": "task", "adapter": {}, "workspace": [],
               "assets": {}, "initial_observation": {},
               "latest_evidence": evidence, "state": {}}
    manager = ContextWindowManager(budgets=ResourceBudgets(max_evidence_chars=2_000))
    bounded = manager.bound_context(context, artifact_root=tmp_path / "artifacts")
    latest = bounded["latest_evidence"]
    assert isinstance(latest, Mapping)
    assert latest.get("truncated") is not True
    assert isinstance(latest["digest"], Mapping)
    for section in ("execution", "tool_calls", "actions", "verifications", "artifacts"):
        assert section in latest["digest"]
    encoded = json.dumps(latest)
    assert "x" * 1000 not in encoded
    assert "artifact://agent/trace" in encoded
    assert "rpc_events" not in encoded


def test_routing_references_are_atomic_during_feedback_compaction(tmp_path):
    long_id = "opaque-" + "x" * 700
    artifact_uri = f"artifact://agent/{long_id}/rollout.mp4"
    evidence_ref = f"evidence://execution/{long_id}"
    run_ref = f"run://artifacts/context/{long_id}.json"
    digest = {"execution": {"completed": True, "error": None,
                             "controller_sha256": "sha"},
              "controller_result": {"routing_ref": run_ref},
              "tool_calls": [], "actions": [], "verifications": [],
              "artifacts": {"rgb": [artifact_uri] * 40,
                            "depth": [artifact_uri.replace("rollout", "depth")] * 40,
                            "trace": artifact_uri.replace("rollout.mp4", "trace.json"),
                            "rollout": artifact_uri}}
    evidence = {"execution": {"completed": True, "error": None},
                "diagnostics": {}, "digest": digest, "evidence_ref": evidence_ref}
    manager = ContextWindowManager(budgets=ResourceBudgets(max_evidence_chars=256))
    context = {"system": "system", "latest_evidence": evidence}
    bounded = manager.bound_context(context, artifact_root=tmp_path / "artifacts")
    latest = bounded["latest_evidence"]
    assert latest["evidence_ref"] == evidence_ref
    assert latest["digest"]["controller_result"]["routing_ref"] == run_ref
    artifacts = latest["digest"]["artifacts"]
    assert set(artifacts) == {"rgb", "depth", "trace", "rollout"}
    assert artifacts["trace"] == artifact_uri.replace("rollout.mp4", "trace.json")
    assert artifacts["rollout"] == artifact_uri
    for key in ("rgb", "depth"):
        sequence = artifacts[key]
        assert sequence["total_count"] == 40
        assert sequence["omitted_count"] == 38
        assert all("..." not in item for item in sequence["head"] + sequence["tail"])


def test_under_budget_latest_evidence_is_returned_without_compaction():
    evidence = {"execution": {"completed": True, "error": None},
                "diagnostics": {"reason": "kept"},
                "digest": {"execution": {"completed": True, "error": None,
                                            "controller_sha256": "sha"},
                           "controller_result": {"message": "kept"},
                           "tool_calls": [{"tool_id": "tool:v1", "output_summary": "kept"}],
                           "actions": [], "verifications": [{"verified": False}],
                           "artifacts": {"rgb": [], "depth": [], "trace": None,
                                         "rollout": None}},
                "evidence_ref": "evidence://small"}
    manager = ContextWindowManager(budgets=ResourceBudgets(max_evidence_chars=20_000))
    bounded = manager.bound_context({"latest_evidence": evidence}, artifact_root="/data/zxy/tmp")
    assert bounded["latest_evidence"] == evidence
    assert bounded["latest_evidence"] is evidence


def test_compare_executions_reports_facts_without_strategy_recommendations(tmp_path):
    loop = AgentLoop.__new__(AgentLoop)
    loop.event_store = None
    unchanged = {"controller_sha256": "sha-b", "agent_evidence": {"digest": {
        "tool_calls": [{"tool_id": "tool:v001"}],
        "actions": [{"type": "move_to_pose", "requested": {"target_xyz": [1, 2, 3], "gripper": -1},
                      "result": {"reached": True, "eef_after": [1, 2, 3],
                                 "final_position_error_m": 0.01,
                                 "final_orientation_error_rad": 0.02}}],
        "verifications": [{"verifier": "target", "verified": False}]}}}
    loop._execution_by_ref = lambda ref: unchanged
    comparison = loop._compare_executions("evidence://a", "evidence://b")
    assert comparison["controller_changed"] is False
    assert comparison["actions"]["requested_targets_changed"] is False
    assert comparison["actions"]["count_changed"] is False
    assert "recommend" not in json.dumps(comparison).lower()

    changed = {**unchanged, "controller_sha256": "sha-c"}
    changed["agent_evidence"] = {"digest": {**unchanged["agent_evidence"]["digest"],
        "actions": [{**unchanged["agent_evidence"]["digest"]["actions"][0],
                      "requested": {"target_xyz": [2, 2, 3], "gripper": -1}}]}}
    loop._execution_by_ref = lambda ref: unchanged if ref.endswith("a") else changed
    comparison = loop._compare_executions("evidence://a", "evidence://b")
    assert comparison["controller_changed"] is True
    assert comparison["actions"]["requested_targets_changed"] is True


def test_decision_record_is_external_structured_context_and_links_execution(tmp_path):
    adapter = FakeAdapter("set marker", tmp_path / "adapter")
    loop = _loop(tmp_path, object(), adapter=adapter, resume=False)
    loop._active_tool_call_id = "decision-call"
    loop._current_model_response_id = "response-1"
    recorded = loop._record_decision(
        goal="set the marker", evidence_refs=["evidence://execution-000001"],
        hypothesis="the previous public result did not verify", decision="update controller",
        expected_effect="verification receipt changes", uncertainty="tool output may be stale")
    loop._active_tool_call_id = None
    assert recorded == {"recorded": True, "decision_id": "decision-decision-call",
                        "evidence_refs": ["evidence://execution-000001"]}
    assert loop._list_decisions()["decisions"][0]["model_response_id"] == "response-1"

    loop.workspace.write_file("controller.py", "def run(robot):\n"
                              "    robot.act({'type':'set_value','value':1})\n"
                              "    return robot.verify('target', {})\n")
    evidence = loop._run_controller()
    assert evidence["decision_id"] == "decision-decision-call"
    execution_event = next(row for row in loop.event_store.events() if row["kind"] == "execution")
    assert execution_event["payload"]["decision_id"] == "decision-decision-call"
    link = next(row for row in loop.event_store.events() if row["kind"] == "decision_link")
    assert link["payload"]["evidence_ref"] == evidence["agent_evidence"]["evidence_ref"]
    assert loop._list_decisions()["decisions"][0]["links"][0]["controller_sha256"] == evidence["controller_sha256"]
    assert loop._agent_evidence(evidence)["decision_id"] == "decision-decision-call"
    encoded = json.dumps(loop._list_decisions())
    assert "hidden reasoning" not in encoded.lower()
    assert str(tmp_path) not in encoded


def test_decision_record_is_deduplicated_and_checkpointed(tmp_path):
    adapter = FakeAdapter("set marker", tmp_path / "adapter")
    loop = _loop(tmp_path, object(), adapter=adapter, resume=False)
    loop._active_tool_call_id = "same-call"
    first = loop._record_decision(goal="g", evidence_refs=[], hypothesis="h",
                                  decision="d", expected_effect="e", uncertainty="u")
    duplicate = loop._record_decision(goal="changed", evidence_refs=[], hypothesis="h",
                                      decision="d", expected_effect="e", uncertainty="u")
    assert first["recorded"] is True
    assert duplicate == {"recorded": False, "duplicate": True, "decision_id": "decision-same-call"}
    loop.current_task = adapter.instruction
    loop._checkpoint()
    resumed = _loop(tmp_path, object(), adapter=adapter, resume=True)
    assert resumed._list_decisions()["decisions"][0]["decision_id"] == "decision-same-call"
    assert resumed._pending_decision_id == "decision-same-call"


def test_decision_record_rejects_non_routing_evidence_reference(tmp_path):
    loop = _loop(tmp_path, object(), resume=False)
    with pytest.raises(ProtocolError, match="opaque routing references"):
        loop._record_decision(goal="g", evidence_refs=["/host/private.json"],
                              hypothesis=None, decision=None, expected_effect=None,
                              uncertainty=None)


def test_decision_record_bounds_and_filters_public_text(tmp_path):
    loop = _loop(tmp_path, object(), resume=False)
    loop._active_tool_call_id = "safe-text-call"
    record = loop._record_decision(
        goal=str(tmp_path / "controller.py"), evidence_refs=[],
        hypothesis="x" * 3000,
        decision="C:\\private\\controller.py",
        expected_effect="public result",
        uncertainty=None,
    )
    stored = loop._list_decisions()["decisions"][0]
    assert record["recorded"] is True
    assert stored["goal"] == "<host path omitted>"
    assert stored["decision"] == "<host path omitted>"
    assert len(stored["hypothesis"]) == 2000
    assert str(tmp_path) not in json.dumps(stored)


def test_verification_receipt_is_only_success_truth_and_one_case_skill_is_allowed(tmp_path):
    class ReceiptAdapter(FakeAdapter):
        def sensor_report(self, execution):
            return {"success": False, "sensor_success": False, "diagnostic": "receipt wins"}
        def verification_receipt(self, execution):
            receipt = super().verification_receipt(execution)
            receipt["verified"] = execution.get("completed") is True
            return receipt

    adapter = ReceiptAdapter("receipt", tmp_path / "adapter")
    class Model:
        turn = 0
        def decide(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return _call("record_decision", {"goal": "repair", "evidence_refs": [],
                    "hypothesis": "public execution is not verified", "decision": "update controller",
                    "expected_effect": "verification succeeds", "uncertainty": None})
            if self.turn == 2:
                return _call("write_file", {"path": "controller.py", "content":
                    "def run(robot):\n return robot.verify('target', {})\n"})
            if self.turn == 3: return _call("run_controller", {})
            return _call("finish", {"summary": "canonical receipt"})
    loop = _loop(tmp_path, Model(), adapter=adapter)
    result = loop.run("receipt")
    assert result["finished"] is True
    evidence_path = tmp_path / "run" / result["latest_evidence"]["artifact_uri"].removeprefix("run://")
    loop.capability_manager.skill_library = type("Skills", (), {
        "freeze": lambda self, **kwargs: {"status": "candidate", "evidence": kwargs["evidence_paths"]}})()
    saved = loop.capability_manager.register_skill(name="one-case", task="receipt",
        controller="controller.py", tool_ids=[], evidence_paths=[str(evidence_path)],
        evidence={"verified": True})
    assert saved["status"] == "candidate"


def test_new_explicit_session_gets_fresh_budget_and_preserves_research_state(tmp_path):
    class First:
        def decide(self, **_kwargs):
            return _call("record_decision", {"goal": "repair", "evidence_refs": [],
                "hypothesis": "the current public result is not verified",
                "decision": "update controller", "expected_effect": "verification succeeds",
                "uncertainty": None})
    first = _loop(tmp_path, First(), budget=LoopBudget(max_steps=1, max_executions=1))
    first.retrieved_assets = {"tools": [{"tool_id": "remembered:v001"}]}
    first_result = first.run("session task")
    assert first_result["finished"] is False and first_result["resumable"] is True
    assert first_result["session"]["steps"] == 1

    class Second:
        turn = 0
        def decide(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return _call("write_file", {"path": "controller.py", "content":
                    "def run(robot):\n    robot.act({'type':'set_value','value':1})\n"
                    "    return robot.verify('target', {})\n"})
            return (_call("run_controller", {}) if self.turn == 2
                    else _call("finish", {"summary": "resumed"}))
    resumed = _loop(tmp_path, Second(), budget=LoopBudget(max_steps=3, max_executions=2))
    result = resumed.run("session task")
    assert result["finished"] is True and result["resumable"] is False
    assert result["session"]["index"] == 2
    assert result["session"]["steps"] == 3
    assert result["cumulative"]["steps"] == 4
    assert resumed.retrieved_assets == {"tools": [{"tool_id": "remembered:v001"}]}
    assert resumed.workspace.controller.is_file()


def test_workspace_process_output_and_disk_growth_are_bounded(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "run/workspace",
        sandbox=UnsafeSandboxBackend(), require_sandbox=False,
        max_bytes=4096, max_process_output_bytes=2048)
    result = workspace.run_command(["/bin/sh", "-c",
        "python -c \"print('x' * 100000)\"; touch should-not-commit"],
        timeout_seconds=10)
    assert result["output_limited"] is True
    assert len(result["output"].encode()) <= 2048
    assert result["committed"] is False
    assert not (workspace.root / "should-not-commit").exists()
    with pytest.raises(Exception, match="file or byte limit"):
        workspace.apply([{"path": "large.bin", "content": "x" * 5000}])


def test_cli_explicit_resume_starts_a_fresh_bounded_session(tmp_path):
    common = [sys.executable, "-m", "embodied_codex",
        "--adapter", "embodied_codex.fake_adapter:FakeAdapter",
        "--model", "embodied_codex.fake_adapter:FakeModel",
        "--task", "explicit resume", "--profile", "autonomous",
        "--run-dir", str(tmp_path / "run"),
        "--asset-root", str(tmp_path / "assets")]
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    first = subprocess.run([*common[:3], "run", *common[3:], "--max-steps", "1"],
        env=environment, text=True, capture_output=True, timeout=60)
    assert first.returncode == 2, first.stdout + first.stderr
    first_result = json.loads(first.stdout)
    assert first_result["resumable"] is True
    resumed = subprocess.run([*common[:3], "resume", *common[3:], "--max-steps", "30"],
        env=environment, text=True, capture_output=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads(resumed.stdout)
    assert result["finished"] is True
    assert result["session"]["index"] == 2
    assert result["session"]["steps"] < result["cumulative"]["steps"]


def test_cli_run_auto_resumes_existing_checkpoint(tmp_path):
    command = [sys.executable, "-m", "embodied_codex", "run",
        "--adapter", "embodied_codex.fake_adapter:FakeAdapter",
        "--model", "embodied_codex.fake_adapter:FakeModel", "--task", "auto resume",
        "--profile", "autonomous", "--max-steps", "30",
        "--run-dir", str(tmp_path / "run"), "--asset-root", str(tmp_path / "assets")]
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]))
    first = subprocess.run([*command, "--max-steps", "1"], env=environment,
        text=True, capture_output=True, timeout=60)
    assert first.returncode == 2
    second = subprocess.run(command, env=environment, text=True,
        capture_output=True, timeout=60)
    assert second.returncode == 0, second.stdout + second.stderr
    assert json.loads(second.stdout)["finished"] is True
