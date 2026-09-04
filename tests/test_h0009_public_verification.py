import json

import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.legacy.agent_loop import AgentLoop, LoopBudget, ProtocolError
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace


class _NoModel:
    def decide(self, **_kwargs):
        raise AssertionError("model is not used by direct generic regression")


class _FalseReceiptAdapter(FakeAdapter):
    def verification_receipt(self, execution):
        receipt = super().verification_receipt(execution)
        receipt["verified"] = False
        return receipt


class _TrueReceiptLocalFalseAdapter(FakeAdapter):
    def dispatch(self, method, arguments):
        result = super().dispatch(method, arguments)
        if method == "verify":
            return {**result, "verified": False, "reason": "local check disagrees"}
        return result

    def verification_receipt(self, execution):
        receipt = super().verification_receipt(execution)
        receipt["verified"] = bool(execution.get("completed") is True
                                   and not execution.get("error"))
        return receipt


def _loop(tmp_path, adapter):
    workspace = PersistentWorkspace(tmp_path / "run/workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=adapter)
    return AgentLoop(model=_NoModel(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace,
            initial_observation=adapter.initial_observation()),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path / "run/events"), root=tmp_path / "run",
        budget=LoopBudget(max_steps=4, max_executions=2, max_diagnostics=2),
        resume=False)


def _physical_controller(loop):
    loop.workspace.write_file("controller.py", """\
def run(robot):
    robot.act({"type": "set_value", "value": 1})
    return robot.verify("target", {})
""")


def _diagnostic_controller(loop):
    loop.workspace.write_file("controller.py", """\
def run(robot):
    observation = robot.observe("proprioception", {})
    robot.record({"observation": observation})
    return observation
""")


def _all_keys(value):
    keys = set()
    if isinstance(value, dict):
        keys.update(value)
        for item in value.values():
            keys.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_all_keys(item))
    return keys


def test_false_authentic_receipt_is_public_false_status(tmp_path):
    loop = _loop(tmp_path, _FalseReceiptAdapter("generic", tmp_path / "adapter"))
    _physical_controller(loop)
    evidence = loop._run_controller()

    public = loop._agent_evidence(evidence)
    assert public["physical_verification"] == {"verified": False}
    assert public["digest"]["verifications"][0]["verified"] is True


def test_true_authentic_receipt_is_public_true_despite_local_disagreement(tmp_path):
    loop = _loop(tmp_path, _TrueReceiptLocalFalseAdapter(
        "generic", tmp_path / "adapter"))
    _physical_controller(loop)
    evidence = loop._run_controller()

    public = loop._agent_evidence(evidence)
    assert public["digest"]["verifications"][0]["verified"] is False
    assert public["physical_verification"] == {"verified": True}


def test_public_physical_status_roundtrips_through_inspect_execution(tmp_path):
    loop = _loop(tmp_path, _FalseReceiptAdapter("generic", tmp_path / "adapter"))
    _physical_controller(loop)
    evidence = loop._run_controller()

    inspected = loop._inspect_execution(
        evidence["agent_evidence"]["evidence_ref"])
    assert inspected["physical_verification"] == {"verified": False}


def test_latest_context_preserves_public_physical_status(tmp_path):
    loop = _loop(tmp_path, _FalseReceiptAdapter("generic", tmp_path / "adapter"))
    _physical_controller(loop)
    loop._run_controller()

    context = loop.context_builder.build(
        task="generic", latest_evidence=loop._agent_latest_evidence, state={})
    assert context["latest_evidence"]["physical_verification"] == {
        "verified": False}


def test_diagnostic_never_fabricates_physical_verification(tmp_path):
    loop = _loop(tmp_path, _FalseReceiptAdapter("generic", tmp_path / "adapter"))
    _diagnostic_controller(loop)
    evidence = loop._run_diagnostic()

    public = loop._agent_evidence(evidence)
    assert "physical_verification" not in public
    inspected = loop._inspect_execution(
        evidence["agent_evidence"]["evidence_ref"])
    assert "physical_verification" not in inspected


def test_public_status_does_not_leak_receipt_metadata(tmp_path):
    loop = _loop(tmp_path, _FalseReceiptAdapter("generic", tmp_path / "adapter"))
    _physical_controller(loop)
    evidence = loop._run_controller()

    public = loop._agent_evidence(evidence)
    keys = _all_keys(public)
    assert "verification_receipt" not in keys
    assert "environment_identity" not in keys
    assert "resume_token" not in keys
    assert "episode_id" not in keys
    assert public["physical_verification"] == {"verified": False}


def test_finish_remains_bound_to_private_authentic_receipt(tmp_path):
    false_loop = _loop(tmp_path / "false", _FalseReceiptAdapter(
        "generic", tmp_path / "false/adapter"))
    _physical_controller(false_loop)
    false_loop._run_controller()
    with pytest.raises(ProtocolError, match="no successful Adapter verification"):
        false_loop._finish("must remain rejected")

    true_loop = _loop(tmp_path / "true", FakeAdapter(
        "generic", tmp_path / "true/adapter"))
    _physical_controller(true_loop)
    true_loop._run_controller()
    assert true_loop._finish("authentic receipt")['completion_valid'] is True
