import json

from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import LoopBudget
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.tools import ToolRegistry
from embodied_codex.kernel.campaign import CampaignAdapter


def _make_loop(tmp_path):
    from tests.test_core_closure import _loop
    return _loop(tmp_path, object(), budget=LoopBudget(max_steps=20, max_executions=2,
                                                       max_diagnostics=4))


def test_diagnostic_observation_is_not_a_physical_trial(tmp_path):
    loop = _make_loop(tmp_path)
    loop.workspace.write_file("controller.py", "def run(robot):\n    return robot.observe('rgb', {})\n")
    before = loop.adapter.generation
    evidence = loop._run_diagnostic()
    assert evidence["execution_kind"] == "diagnostic"
    assert loop.budget.executions == 0
    assert loop.adapter.generation == before
    assert loop._artifact_scope is None


def test_diagnostic_act_fails_closed_without_mutation(tmp_path):
    loop = _make_loop(tmp_path)
    loop.workspace.write_file("controller.py", "def run(robot):\n    return robot.act({'type':'set_value','value':1})\n")
    before = (loop.adapter.value, loop.budget.executions)
    try:
        loop._run_diagnostic()
    except Exception as exc:
        assert "diagnostic" in str(exc).lower()
    else:
        raise AssertionError("diagnostic act unexpectedly succeeded")
    assert (loop.adapter.value, loop.budget.executions) == before
    assert loop._artifact_scope is None


def test_two_diagnostics_have_distinct_durable_references(tmp_path):
    loop = _make_loop(tmp_path)
    loop.workspace.write_file("controller.py", "def run(robot):\n    return robot.observe('rgb', {})\n")
    first = loop._run_diagnostic(); second = loop._run_diagnostic()
    assert first["artifact_uri"] != second["artifact_uri"]
    assert first["artifact_sha256"] != ""
    assert second["artifact_sha256"] != ""
    assert len(loop._list_executions()["executions"]) == 2


def test_diagnostic_readonly_capability_is_allowed(tmp_path):
    loop = _make_loop(tmp_path)
    loop.adapter.register_capability("read:v1", lambda payload: {"ok": True},
                                    {"input_schema": {"type": "object"},
                                     "output_schema": {"type": "object"},
                                     "consequence": "READ_ONLY"})
    loop.workspace.write_file("controller.py", "def run(robot):\n    return robot.use('read:v1', {})\n")
    evidence = loop._run_diagnostic()
    assert evidence["execution"]["completed"] is True


def test_diagnostic_unknown_consequence_is_rejected(tmp_path):
    loop = _make_loop(tmp_path)
    loop.adapter.register_capability("unknown:v1", lambda payload: {"changed": True},
                                    {"input_schema": {"type": "object"},
                                     "output_schema": {"type": "object"}})
    loop.workspace.write_file("controller.py", "def run(robot):\n    return robot.use('unknown:v1', {})\n")
    try:
        loop._run_diagnostic()
    except Exception as exc:
        assert "diagnostic" in str(exc).lower()
    else:
        raise AssertionError("unknown consequence unexpectedly succeeded")


def test_diagnostic_evidence_is_not_physical_completion_evidence(tmp_path):
    loop = _make_loop(tmp_path)
    loop.workspace.write_file("controller.py", "def run(robot):\n    return robot.observe('rgb', {})\n")
    evidence = loop._run_diagnostic()
    assert evidence["execution_kind"] == "diagnostic"
    assert loop.latest_physical_evidence is None


def test_tool_registry_rejects_unknown_consequence():
    registry = ToolRegistry()
    try:
        registry.add("bad", "bad", {"type": "object"}, lambda: None, consequence="UNKNOWN")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown kernel consequence was accepted")


def test_fake_adapter_lifecycle_records_execution_kind(tmp_path):
    adapter = FakeAdapter("task", tmp_path / "adapter")
    adapter.begin_execution("diagnostic")
    assert adapter._diagnostic_state["kind"] == "diagnostic"


def test_campaign_registration_failure_rolls_back_prior_cases(tmp_path):
    first = FakeAdapter("task", tmp_path / "one")
    second = FakeAdapter("task", tmp_path / "two")
    original = second.register_capability
    def fail(*args, **kwargs):
        raise RuntimeError("injected registration failure")
    second.register_capability = fail
    campaign = CampaignAdapter([("one", first), ("two", second)])
    try:
        campaign.register_capability("new:v1", lambda payload: {},
                                     {"input_schema": {"type": "object"},
                                      "output_schema": {"type": "object"},
                                      "consequence": "READ_ONLY"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("campaign registration unexpectedly succeeded")
    assert "new:v1" not in first.capabilities
    second.register_capability = original
    campaign.register_capability("new:v1", lambda payload: {},
                                 {"input_schema": {"type": "object"},
                                  "output_schema": {"type": "object"},
                                  "consequence": "READ_ONLY"})
    assert "new:v1" in first.capabilities and "new:v1" in second.capabilities


def test_campaign_unknown_consequence_is_not_readonly(tmp_path):
    campaign = CampaignAdapter([("one", FakeAdapter("task", tmp_path / "one"))])
    campaign.register_capability("unknown:v1", lambda payload: {},
                                 {"input_schema": {"type": "object"},
                                  "output_schema": {"type": "object"}})
    assert campaign.capability_consequence("unknown:v1") == "UNKNOWN"


def test_manifest_persistence_uses_replace_and_fsync(tmp_path, monkeypatch):
    loop = _make_loop(tmp_path)
    calls = []
    import embodied_codex.kernel.agent_loop as module
    real_replace = module.os.replace
    real_fsync = module.os.fsync
    monkeypatch.setattr(module.os, "replace", lambda *args: (calls.append("replace"), real_replace(*args))[1])
    monkeypatch.setattr(module.os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    loop._persist_artifact_manifest()
    assert "replace" in calls and calls.count("fsync") >= 1


def test_diagnostic_budget_is_independent_and_bounded():
    budget = LoopBudget(max_trials=1, max_diagnostics=2)
    assert not budget.diagnostics_exhausted()
    budget.diagnostics = 2
    assert budget.diagnostics_exhausted()
    assert not budget.exhausted()


def test_sdk_exposes_numeric_and_reference_control_equally():
    point = LIBERO_ROBOT_SDK_CONTRACT["actions"]["move_to_point"]
    pose = LIBERO_ROBOT_SDK_CONTRACT["actions"]["move_to_pose"]
    assert {"target_ref"} in [set(item["required"]) for item in point["any_of"]]
    assert {"frame", "position_m"} in [set(item["required"]) for item in point["any_of"]]
    assert {"pose_ref"} in [set(item["required"]) for item in pose["any_of"]]
    assert {"frame", "position_m", "quaternion_xyzw"} in [set(item["required"]) for item in pose["any_of"]]
    assert "both first-class" in point["rule"]


def test_native_capability_search_is_strategy_neutral(tmp_path):
    adapter = FakeAdapter("task", tmp_path / "adapter")
    adapter.native_capability_index = lambda: [{"capability_id": "native:vision",
                                                "purpose": "read-only sensor inspection"}]
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=object(), adapter=adapter)
    result = manager.search("vision")
    assert result["native"][0]["source"] == "native"
    assert "grasp" not in json.dumps(result).lower()
