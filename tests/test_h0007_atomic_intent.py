import json
import sys

import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace


DECISION = {
    "goal": "set the generic marker",
    "evidence_refs": [],
    "hypothesis": "the current public value is not verified",
    "decision": "execute the declared generic operation",
    "expected_effect": "the public verifier observes value one",
    "uncertainty": "the Adapter result remains authoritative",
}


def _call(name, arguments, call_id="call-1"):
    return {"content": "", "tool_calls": [{"id": call_id, "name": name,
        "arguments": json.dumps(arguments)}]}


def _loop(tmp_path, model, *, event_store=None, max_steps=1, max_trials=1):
    adapter = FakeAdapter("atomic intent", tmp_path / "adapter")
    workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
                                adapter=adapter)
    loop = AgentLoop(model=model, workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace,
            initial_observation=adapter.initial_observation()),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=event_store or EventStore(tmp_path / "events"), root=tmp_path / "run",
        budget=LoopBudget(max_steps=max_steps, max_executions=max_trials), resume=False)
    return loop


def _controller(value=1):
    return ("def run(robot):\n"
            f"    robot.act({{'type':'set_value','value':{value}}})\n"
            "    return robot.verify('target', {})\n")


def _schemas(loop):
    return {item["function"]["name"]: item["function"]
            for item in loop.tools.schemas}


def test_inline_decision_and_physical_operation_execute_once_and_link_evidence(tmp_path):
    class Model:
        def decide(self, **_kwargs):
            return _call("run_controller", {"decision_context": DECISION}, "physical-1")

    loop = _loop(tmp_path, Model())
    loop.workspace.write_file("controller.py", _controller())
    result = loop.run("atomic physical intent")

    assert result["physical_trials"] == 1
    records = [row for row in loop.event_store.events() if row["kind"] == "decision_record"]
    assert len(records) == 1
    assert records[0]["payload"]["decision_id"] == "decision-physical-1"
    execution = next(row for row in loop.event_store.events() if row["kind"] == "execution")
    assert execution["payload"]["decision_id"] == "decision-physical-1"
    link = next(row for row in loop.event_store.events() if row["kind"] == "decision_link")
    assert link["payload"]["decision_id"] == "decision-physical-1"


def test_inline_decision_and_workspace_mutation_need_one_model_call(tmp_path):
    class Model:
        def decide(self, **_kwargs):
            return _call("write_file", {"path": "controller.py", "content": _controller(),
                "decision_context": DECISION}, "workspace-1")

    loop = _loop(tmp_path, Model())
    result = loop.run("atomic workspace intent")

    assert result["steps"] == 1
    assert loop.workspace.controller.read_text() == _controller()
    tool = next(row for row in loop.event_store.events()
                if row["kind"] == "tool_result" and row["payload"]["name"] == "write_file")
    assert tool["payload"]["payload"]["ok"] is True
    assert tool["payload"]["decision_id"] == "decision-workspace-1"


def test_consequential_agent_schemas_require_atomic_decision_context(tmp_path):
    loop = _loop(tmp_path, object())
    schemas = _schemas(loop)

    for name in ("write_file", "replace_file_lines", "run_command", "run_controller",
                 "reset_case", "restore_controller_version"):
        parameters = schemas[name]["parameters"]
        assert "decision_context" in parameters["required"]
        context = parameters["properties"]["decision_context"]
        assert set(context["required"]) == set(DECISION)
        assert context["additionalProperties"] is False

    assert "decision_context" not in schemas["read_file"]["parameters"].get("required", [])
    assert "decision_context" not in schemas["run_validation"]["parameters"].get("required", [])
    assert "record_decision" not in schemas


def test_every_visible_consequential_tool_uses_the_same_atomic_contract(tmp_path):
    loop = _loop(tmp_path, object())
    for group in ("source_inspection", "web_acquisition", "asset_authoring"):
        loop.tools.activate(group)

    schemas = _schemas(loop)
    consequential = 0
    for name, schema in schemas.items():
        consequence = loop.tools.metadata(name).consequence
        parameters = schema["parameters"]
        if consequence in {"READ_ONLY", "VALIDATION"}:
            assert "decision_context" not in parameters.get("required", [])
            continue
        consequential += 1
        assert "decision_context" in parameters["required"]
        assert parameters["properties"]["decision_context"]["required"] == list(DECISION)
        assert "records and links it before execution" in schema["description"]
    assert consequential > 10


def test_missing_inline_decision_rejects_before_workspace_mutation(tmp_path):
    class Model:
        def decide(self, **_kwargs):
            return _call("write_file", {"path": "controller.py", "content": "changed"})

    loop = _loop(tmp_path, Model())
    loop.workspace.write_file("controller.py", "original")
    loop.run("missing context")

    assert loop.workspace.controller.read_text() == "original"
    assert not any(row["kind"] == "decision_record" for row in loop.event_store.events())


@pytest.mark.parametrize("context", [
    {**DECISION, "extra": "not allowed"},
    {key: value for key, value in DECISION.items() if key != "decision"},
    {**DECISION, "evidence_refs": ["/host/private.json"]},
])
def test_invalid_inline_decision_rejects_before_workspace_mutation(tmp_path, context):
    class Model:
        def decide(self, **_kwargs):
            return _call("write_file", {"path": "controller.py", "content": "changed",
                "decision_context": context})

    loop = _loop(tmp_path, Model())
    loop.workspace.write_file("controller.py", "original")
    loop.run("invalid context")

    assert loop.workspace.controller.read_text() == "original"
    assert not any(row["kind"] == "decision_record" for row in loop.event_store.events())


def test_decision_commit_failure_prevents_handler_invocation(tmp_path):
    class FailingDecisionStore(EventStore):
        def commit(self, kind, payload):
            if kind == "decision_record":
                raise OSError("injected decision durability failure")
            return super().commit(kind, payload)

    class Model:
        def decide(self, **_kwargs):
            return _call("write_file", {"path": "controller.py", "content": "changed",
                "decision_context": DECISION})

    store = FailingDecisionStore(tmp_path / "events")
    loop = _loop(tmp_path, Model(), event_store=store)
    loop.workspace.write_file("controller.py", "original")
    loop.run("durability failure")

    assert loop.workspace.controller.read_text() == "original"


def test_readonly_and_validation_calls_do_not_require_decision_context(tmp_path):
    class Model:
        turn = 0

        def decide(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return _call("read_file", {"path": "controller.py"}, "read-1")
            return _call("run_validation", {"argv": [sys.executable, "-m", "py_compile",
                "controller.py"]}, "validate-1")

    loop = _loop(tmp_path, Model(), max_steps=2)
    loop.workspace.write_file("controller.py", _controller())
    loop.run("read and validate")

    results = [row["payload"] for row in loop.event_store.events()
               if row["kind"] == "tool_result"]
    assert [(row["name"], row["payload"]["ok"]) for row in results] == [
        ("read_file", True), ("run_validation", True)]
    assert not any(row["kind"] == "decision_record" for row in loop.event_store.events())


def test_duplicate_inline_physical_call_identity_cannot_execute_twice(tmp_path):
    class Model:
        def decide(self, **_kwargs):
            return _call("run_controller", {"decision_context": DECISION}, "same-call")

    loop = _loop(tmp_path, Model(), max_steps=2, max_trials=2)
    loop.workspace.write_file("controller.py", _controller())
    result = loop.run("duplicate delivery")

    assert result["physical_trials"] == 1
    records = [row for row in loop.event_store.events() if row["kind"] == "decision_record"]
    executions = [row for row in loop.event_store.events() if row["kind"] == "execution"]
    assert len(records) == 1
    assert len(executions) == 1
