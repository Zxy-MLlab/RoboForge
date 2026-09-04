import hashlib
import json
from pathlib import Path

import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.legacy.agent_loop import AgentLoop, LoopBudget
from embodied_codex.legacy.campaign import CampaignAdapter, CampaignRunner
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace


def _call(name, arguments=None, index=1):
    return {"content": "", "tool_calls": [{"id": f"call-{index}",
        "name": name, "arguments": json.dumps(arguments or {})}]}


def _loop(tmp_path, model, adapter, *, budget=None, loop_type=AgentLoop):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets",
        workspace=workspace, adapter=adapter)
    return loop_type(model=model, workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace,
            initial_observation=adapter.initial_observation()),
        capability_manager=manager,
        runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path / "events"), root=tmp_path,
        budget=budget or LoopBudget(max_steps=12, max_executions=8),
        resume=False)


class _Case(FakeAdapter):
    def __init__(self, task, root, *, case, order):
        super().__init__(task, root, case=case)
        self.order = order

    def dispatch(self, method, arguments):
        if method == "act":
            self.order.append(str(self.case))
        return super().dispatch(method, arguments)


class _CaseSelectingModel:
    def __init__(self):
        self.turn = 0

    def decide(self, *, messages, tools):
        self.turn += 1
        sequence = {
            1: ("record_decision", {"goal": "repair", "evidence_refs": [],
                "hypothesis": "the current public result is not verified", "decision": "update controller",
                "expected_effect": "verification succeeds", "uncertainty": None}),
            2: ("write_file", {"path": "controller.py", "content":
                "def run(robot):\n    robot.act({'type':'set_value','value':1})\n"
                "    return robot.verify('target', {})\n"}),
            3: ("select_case", {"case_id": "case-002"}),
            4: ("run_controller", {}),
            5: ("record_decision", {"goal": "repair", "evidence_refs": [],
                "hypothesis": "the current public result is not verified", "decision": "rerun selected case",
                "expected_effect": "verification succeeds", "uncertainty": None}),
            6: ("run_controller", {}),
            7: ("record_decision", {"goal": "repair", "evidence_refs": [],
                "hypothesis": "the current public result is not verified", "decision": "switch selected case",
                "expected_effect": "verification succeeds", "uncertainty": None}),
            8: ("select_case", {"case_id": "case-001"}),
            9: ("run_controller", {}),
            10: ("finish", {"summary": "current case verified"}),
        }
        name, arguments = sequence[self.turn]
        return _call(name, arguments, self.turn)


def test_multicase_order_changes_only_after_explicit_model_selection(tmp_path):
    order = []
    first = _Case("set target", tmp_path / "cases/A", case="A", order=order)
    second = _Case("set target", tmp_path / "cases/B", case="B", order=order)
    adapter = CampaignAdapter((("A", first), ("B", second)))
    loop = _loop(tmp_path, _CaseSelectingModel(), adapter, loop_type=CampaignRunner)
    try:
        result = loop.run(adapter.instruction)
    finally:
        adapter.close()
    # Each explicit model run is a new experiment; only crash recovery may deduplicate.
    assert order == ["B", "B", "A"]
    attempts = [row for row in loop.research_state["attempts"]
                if row["tool"] == "run_controller"]
    assert len(attempts) == 3
    assert result["finished"] is True
    assert result["selected_case"] == "case-001"
    assert result["available_cases"] == ["case-001", "case-002"]
    assert "campaign" not in loop.state
    serialized = json.dumps(loop.state)
    assert "failure_focus" not in serialized
    assert '"queue"' not in serialized


class _EvidenceBoundaryAdapter(FakeAdapter):
    def sensor_report(self, execution):
        return {"diagnostic": "gripper missed", "hidden_success": True,
            "benchmark_state": 17, "case_handle": "sealed-case",
            "internal_episode_id": "secret-episode", "resume_token": "secret-token"}

    def agent_evidence(self, execution, sensor_report):
        return {"diagnostic": sensor_report["diagnostic"]}


class _EvidenceCaptureModel:
    def __init__(self):
        self.turn = 0
        self.second_messages = None

    def decide(self, *, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return _call("record_decision", {"goal": "inspect", "evidence_refs": [],
                "hypothesis": "public execution requires inspection", "decision": "inspect evidence",
                "expected_effect": "diagnostic facts available", "uncertainty": None}, self.turn)
        if self.turn == 2:
            return _call("run_controller", {}, self.turn)
        self.second_messages = json.loads(json.dumps(messages))
        return _call("read_file", {"path": "controller.py"}, self.turn)


def test_agent_evidence_excludes_harness_and_evaluator_metadata(tmp_path):
    adapter = _EvidenceBoundaryAdapter("test evidence boundary", tmp_path / "adapter")
    model = _EvidenceCaptureModel()
    loop = _loop(tmp_path, model, adapter,
        budget=LoopBudget(max_steps=3, max_executions=2))
    loop.workspace.write_file("controller.py",
        "def run(robot):\n    return robot.verify('target', {})\n")
    loop.run(adapter.instruction)
    visible = json.dumps(model.second_messages, sort_keys=True)
    assert "gripper missed" in visible
    for secret in ("case_handle", "environment_identity", "resume_token",
                   "episode_id", "verification_receipt", "hidden_success",
                   "benchmark_state", "secret-episode", "secret-token"):
        assert secret not in visible
    event = next(row for row in EventStore(tmp_path / "events").events()
                 if row["kind"] == "execution")
    assert event["payload"]["environment_identity"]["episode_id"].startswith("fake:")
    assert event["payload"]["verification_receipt"]["controller_sha256"] == hashlib.sha256(
        loop.workspace.controller.read_bytes()).hexdigest()


class _DisclosureModel:
    def __init__(self):
        self.turn = 0
        self.schema_names = []

    def decide(self, *, messages, tools):
        self.turn += 1
        names = {item["function"]["name"] for item in tools}
        self.schema_names.append(names)
        if self.turn == 1:
            return _call("acquire_capability", {"query": "robot perception"}, self.turn)
        return _call("list_files", {}, self.turn)


def test_tool_schemas_are_progressively_disclosed_after_semantic_acquisition(tmp_path):
    adapter = FakeAdapter("progressive tools", tmp_path / "adapter")
    model = _DisclosureModel()
    loop = _loop(tmp_path, model, adapter,
        budget=LoopBudget(max_steps=2, max_executions=1))
    loop.run(adapter.instruction)
    first, second = model.schema_names
    assert {"list_files", "search_assets", "inspect_asset", "run_controller",
            "inspect_execution", "view_sensor_artifact", "finish",
            "acquire_capability"}.issubset(first)
    # Legacy sessions may still see the compatibility control, but new flows
    # disclose acquisition tools from the semantic operation itself.
    assert "search_web" not in first
    assert "register_tool" not in first
    assert "load_tool_source" not in first
    assert "search_web" in second
    assert "download_public_asset" in second
    assert "register_tool" in second

    resumed_model = _DisclosureModel()
    resumed_model.turn = 1
    resumed = AgentLoop(model=resumed_model, workspace=loop.workspace,
        adapter=adapter, context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=loop.capability_manager, workspace=loop.workspace),
        capability_manager=loop.capability_manager,
        runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path / "events"), root=tmp_path,
        budget=LoopBudget(max_steps=1, max_executions=1), resume=True)
    resumed.run(adapter.instruction)
    assert "search_web" in resumed_model.schema_names[0]
    assert "register_tool" in resumed_model.schema_names[0]


class _FailedExecutionDisclosureModel:
    def __init__(self):
        self.turn = 0
        self.after_failure = None

    def decide(self, *, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return _call("run_controller", {}, self.turn)
        self.after_failure = {item["function"]["name"] for item in tools}
        return _call("list_files", {}, self.turn)


def test_failed_execution_does_not_activate_optional_tool_groups(tmp_path):
    adapter = FakeAdapter("failure does not route strategy", tmp_path / "adapter")
    model = _FailedExecutionDisclosureModel()
    loop = _loop(tmp_path, model, adapter,
        budget=LoopBudget(max_steps=2, max_executions=2))
    loop.workspace.write_file("controller.py",
        "def run(robot):\n    robot.act({'type':'set_value','value':0})\n"
        "    return robot.verify('target', {})\n")
    loop.run(adapter.instruction)
    assert "search_web" not in model.after_failure
    assert "register_tool" not in model.after_failure
    assert "load_tool_source" not in model.after_failure


def test_shared_tools_require_per_tool_inspection_and_activation(tmp_path):
    class Library:
        def __init__(self):
            self.loaded = []

        def search(self, query, limit=5, statuses=None):
            return [{"tool_id": "chosen:v001", "description": "chosen", "status": "promoted"},
                    {"tool_id": "ignored:v001", "description": "ignored", "status": "promoted"}]

        def inspect(self, tool_id, include_source=False):
            result = {"manifest": {"tool_id": tool_id, "status": "promoted",
                "input_schema": {"type": "object"}, "output_schema": {"type": "object"}},
                "manual": {"purpose": tool_id, "examples": [], "limitations": []}}
            if include_source:
                result["source"] = "def run(payload): return payload"
            return result

        def runtime_function(self, tool_id, *, artifact_resolver=None):
            self.loaded.append(tool_id)
            return lambda payload: payload

    class Adapter:
        def __init__(self):
            self.capabilities = {}

        def register_capability(self, tool_id, function, contract):
            self.capabilities[tool_id] = function

    workspace = PersistentWorkspace(tmp_path / "workspace")
    library = Library()
    adapter = Adapter()
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
        adapter=adapter, tool_library=library)
    summaries = manager.search("tool")
    assert {row["tool_id"] for row in summaries["tools"]} == {
        "chosen:v001", "ignored:v001"}
    assert adapter.capabilities == {} and library.loaded == []
    detail = manager.inspect("chosen:v001")
    assert "source" not in detail
    manager.activate_tool("chosen:v001")
    assert set(adapter.capabilities) == {"chosen:v001"}
    assert library.loaded == ["chosen:v001"]
    assert "ignored:v001" not in manager._inspected_tools


def test_plain_agent_loop_has_no_evaluation_policy_hook_parameter(tmp_path):
    adapter = FakeAdapter("no evaluation hooks", tmp_path / "adapter")
    workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets",
        workspace=workspace, adapter=adapter)
    with pytest.raises(TypeError, match="policies"):
        AgentLoop(model=object(), workspace=workspace, adapter=adapter,
            context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
                asset_registry=manager, workspace=workspace),
            capability_manager=manager, policies=[object()], resume=False)


def test_model_can_choose_fresh_episode_without_forced_reset(tmp_path):
    adapter = FakeAdapter("reset", tmp_path / "adapter")
    workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace, adapter=adapter)
    loop = AgentLoop(model=object(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index, asset_registry=manager,
                                       workspace=workspace), capability_manager=manager,
        runtime=ControllerRuntime(timeout_seconds=10), root=tmp_path, resume=False)
    adapter.value = 1
    result = loop.tools.invoke("reset_case", {})
    assert result["reset"] is True and adapter.value == 0


def test_optional_tool_group_can_be_deactivated(tmp_path):
    adapter = FakeAdapter("deactivate", tmp_path / "adapter")
    loop = _loop(tmp_path, object(), adapter)
    loop.tools.activate("web_acquisition")
    loop.tools.deactivate("web_acquisition")
    assert "search_web" not in loop.tools.names(active_only=True)
