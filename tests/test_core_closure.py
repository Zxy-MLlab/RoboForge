import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget, ProtocolError
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.context_window import ContextWindowManager, ResourceBudgets
from embodied_codex.kernel.events import EventStore, EventStoreError
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
                return _call("write_file", {"path": "controller.py", "content":
        "def run(robot):\n return robot.verify('target', {})\n"}, "write")
            if self.turn == 2:
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
        def decide(self, **_kwargs):
            return {"content": "", "tool_calls": [
                {"id": str(index), "name": "list_files", "arguments": "{}"}
                for index in range(5)]}
    loop = _loop(tmp_path / "calls", ManyCalls(),
                 budget=LoopBudget(max_steps=1, max_executions=1),
                 context_window=window)
    result = loop.run("bounded")
    tool_results = [row for row in loop.event_store.events() if row["kind"] == "tool_result"]
    assert len(tool_results) == 2
    assert result["resumable"] is True

    image = loop.workspace.root / "large.png"
    image.write_bytes(b"not-an-image" * 1000)
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (_ for _ in ()).throw(
        AssertionError("large image was read before stat")) if path == image else original(path))
    with pytest.raises(ProtocolError, match="size limit"):
        loop._view_artifact("large.png")


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
                return _call("write_file", {"path": "controller.py", "content":
                    "def run(robot):\n return robot.verify('target', {})\n"})
            if self.turn == 2: return _call("run_controller", {})
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
            return _call("write_file", {"path": "controller.py", "content":
                "def run(robot):\n robot.act({'type':'set_value','value':1})\n"
                " return robot.verify('target', {})\n"})
    first = _loop(tmp_path, First(), budget=LoopBudget(max_steps=1, max_executions=1))
    first.retrieved_assets = {"tools": [{"tool_id": "remembered:v001"}]}
    first_result = first.run("session task")
    assert first_result["finished"] is False and first_result["resumable"] is True
    assert first_result["session"]["steps"] == 1

    class Second:
        turn = 0
        def decide(self, **_kwargs):
            self.turn += 1
            return (_call("run_controller", {}) if self.turn == 1
                    else _call("finish", {"summary": "resumed"}))
    resumed = _loop(tmp_path, Second(), budget=LoopBudget(max_steps=3, max_executions=2))
    result = resumed.run("session task")
    assert result["finished"] is True and result["resumable"] is False
    assert result["session"]["index"] == 2
    assert result["session"]["steps"] == 2
    assert result["cumulative"]["steps"] == 3
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
