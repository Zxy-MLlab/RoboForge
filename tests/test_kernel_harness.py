import ast
import json
import sys
from pathlib import Path

from embodied_codex.kernel.agent_loop import AgentLoop
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace
from embodied_codex.kernel.assets import AssetRegistry
from embodied_codex.kernel.assets import CapabilityLibrary
from evaluation.anti_cheating import AntiCheatingPolicy
from evaluation.generalization import GeneralizationPolicy
from evaluation.provenance import ProvenancePolicy
from evaluation.sealed_evaluation import SealedEvaluationPolicy


class FakeAdapter:
    instruction = "move the marker"
    def __init__(self): self.value = 0; self.generation = "fake-generation-1"
    def dispatch(self, method, arguments):
        if method == "act": self.value = arguments["action"]["value"]; return {"ok": True}
        if method == "verify": return {"verified": self.value == 1}
        if method == "observe": return {"value": self.value}
        if method == "record": return {"recorded": True}
        if method == "use": return {"tool_id": arguments["tool_id"], "result": {"value": 1}}
        raise ValueError(method)
    def initial_observation(self): return {"value": self.value}
    def project_rpc_output(self, method, arguments, result): return dict(result)
    def sensor_report(self, execution):
        return {"sensor_success": self.value == 1, "value": self.value}
    def execution_identity(self):
        return {"episode_id": "fake-episode", "environment_generation": self.generation}
    def resume_protocol(self):
        return {"supports_resume": True, "resume_token": "fake-resume-1",
                "environment_generation": self.generation, "actions_idempotent": False,
                "replay_allowed": True}
    def verification_receipt(self, execution):
        return {"verified": bool(self.value == 1 and execution.get("completed") is True
                                 and execution.get("sensor_verification_observed") is True),
                "controller_sha256": execution.get("program_sha256"),
                "environment_identity": self.execution_identity(),
                "episode_id": "fake-episode", "environment_generation": self.generation}
    def validate_execution_receipt(self, receipt):
        return receipt.get("verified") is True and receipt.get("environment_identity") == self.execution_identity()
    def close(self): pass


class SharedAdapter(FakeAdapter):
    def __init__(self): super().__init__(); self.capabilities = {}
    def register_capability(self, tool_id, function, contract): self.capabilities[tool_id] = function
    def dispatch(self, method, arguments):
        if method == "use":
            return {"tool_id": arguments["tool_id"], "result": self.capabilities[arguments["tool_id"]](arguments["payload"])}
        return super().dispatch(method, arguments)


class SealedAdapter(FakeAdapter):
    def __init__(self): super().__init__(); self.sealed = False
    def seal_controller_execution(self): self.sealed = True
    def _sealed_check_once(self):
        assert self.sealed is True
        return self.value == 1


class CrashModel:
    def __init__(self, resume=False): self.turn = 0; self.resume = resume
    def decide(self, *, messages, tools):
        self.turn += 1
        if not self.resume and self.turn == 1:
            return call("write_file", {"path": "controller.py", "content": "def run(robot):\n    robot.act({'value': 1})\n    return robot.verify('goal', {})\n"}, self.turn)
        if not self.resume and self.turn == 2:
            return call("run_controller", {}, self.turn)
        if not self.resume: raise RuntimeError("simulated process crash")
        if self.turn == 1: return call("run_controller", {}, self.turn)
        return call("finish", {"summary": "resumed"}, self.turn)


class MemoryAssets:
    def __init__(self): self.saved = []
    def search(self, query, limit=5): return {"tools": list(self.saved)}
    def inspect(self, asset_id): return next(item for item in self.saved if item["id"] == asset_id)


def call(name, arguments, index):
    return {"tool_calls": [{"id": f"c{index}", "name": name,
                             "arguments": json.dumps(arguments)}], "content": ""}


def committed_execution(workspace, adapter, manager):
    loop = AgentLoop(model=object(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index={"protocol": "fake-v1"},
            asset_registry=manager, workspace=workspace),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(workspace.root.parent / "events"),
        root=workspace.root.parent, resume=False)
    return loop._run_controller()


class FakeModel:
    def __init__(self): self.turn = 0
    def decide(self, *, messages, tools):
        self.turn += 1
        if self.turn == 1:
            return call("write_file", {"path": "controller.py", "content": "def run(robot):\n    robot.act({'value': 0})\n    return robot.verify('goal', {})\n"}, self.turn)
        if self.turn == 2:
            return call("run_controller", {}, self.turn)
        if self.turn == 3:
            return call("read_file", {"path": "controller.py"}, self.turn)
        if self.turn == 4:
            return call("write_file", {"path": "controller.py", "content": "def run(robot):\n    robot.act({'value': 1})\n    return robot.verify('goal', {})\n"}, self.turn)
        if self.turn == 5:
            return call("run_controller", {}, self.turn)
        if self.turn == 6:
            return call("finish", {"summary": "verified"}, self.turn)
        return call("finish", {"summary": "done"}, self.turn)


def test_kernel_failure_evidence_edit_and_asset_persistence(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    assets = MemoryAssets(); adapter = FakeAdapter()
    context = ContextBuilder(adapter_index={"protocol": "fake-v1"}, asset_registry=assets,
                             workspace=workspace)
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace, adapter=adapter)
    result = AgentLoop(model=FakeModel(), workspace=workspace, adapter=adapter,
                       context_builder=context, capability_manager=manager,
                       runtime=ControllerRuntime(timeout_seconds=10),
                       event_store=EventStore(tmp_path), root=str(tmp_path)).run()
    assert result["executions"] == 2
    assert result["latest_evidence"]["verification_receipt"]["verified"] is True
    assert result["finished"] is True
    assert len(EventStore(tmp_path).events()) >= 8


def test_kernel_import_boundary_has_no_evaluation_or_adapter_dependency():
    root = Path(__file__).parents[1] / "embodied_codex" / "kernel"
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        text = "\n".join(ast.dump(node) for node in imports)
        assert "evaluation" not in text
        assert "libero" not in text.lower()


def test_asset_detail_is_manual_first_and_source_requires_explicit_escalation():
    class ToolLibrary:
        def inspect(self, asset_id):
            return {"manifest": {"tool_id": asset_id}, "source": "def run(payload): pass"}
        def manual(self, asset_id): return {"manual": {"purpose": "test"}}
    registry = AssetRegistry(tools=ToolLibrary())
    assert "source" not in registry.inspect("demo:v001")
    assert registry.load_source("demo:v001")["source"].startswith("def run")


def test_shared_tool_is_bound_and_reused_by_independent_workspace(tmp_path):
    asset_root = tmp_path / "shared-assets"
    workspace_a = PersistentWorkspace(tmp_path / "run-a" / "workspace")
    workspace_a.write_file("increment.py", "def run(payload):\n    return {'value': payload['value'] + 1}\n")
    adapter_a = SharedAdapter()
    library_a = CapabilityLibrary(asset_root / "tools", workspace_a.root, python=sys.executable, scope_id="shared01")
    manager_a = CapabilityManager(asset_root=asset_root, workspace=workspace_a, adapter=adapter_a,
                                  tool_library=library_a)
    registration = manager_a.register_tool(
        name="increment", source_path="increment.py", description="increment a value",
        input_schema={"type": "object", "properties": {"value": {"type": "integer"}},
                      "required": ["value"], "additionalProperties": False},
        output_schema={"type": "object", "properties": {"value": {"type": "integer"}},
                       "required": ["value"], "additionalProperties": False},
        source_urls=["https://example.com/increment"], trained_on_current_task=False)
    manager_a.test_tool(registration["tool_id"], [{"input": {"value": 2}, "expected": {"value": 3}}])
    workspace_a.write_file("controller.py", "def run(robot):\n"
        "    target=robot.use('increment:v001', {'value': 0})\n"
        "    robot.act(target)\n    return robot.verify('goal', {})\n")
    evidence = committed_execution(workspace_a, adapter_a, manager_a)
    manager_a.promote_asset(registration["tool_id"], [evidence["artifact_uri"]])

    workspace_b = PersistentWorkspace(tmp_path / "run-b" / "workspace")
    workspace_b.write_file("controller.py", "def run(robot):\n    return robot.use('increment:v001', {'value': 4})\n")
    adapter_b = SharedAdapter()
    library_b = CapabilityLibrary(asset_root / "tools", workspace_b.root, python=sys.executable, scope_id="shared01")
    manager_b = CapabilityManager(asset_root=asset_root, workspace=workspace_b, adapter=adapter_b,
                                  tool_library=library_b)
    assert adapter_b.capabilities == {}
    found = manager_b.search("increment a value")
    assert [row["tool_id"] for row in found["tools"]] == [registration["tool_id"]]
    detail = manager_b.inspect(registration["tool_id"])
    assert "source" not in detail
    assert manager_b.activate_tool(registration["tool_id"])["bound"] is True
    result = ControllerRuntime(timeout_seconds=10).execute(workspace_b.controller, adapter_b)
    assert result["completed"] is True
    assert result["result"] == {"value": 5}


def test_cross_task_reuse_needs_fewer_robot_executions_than_no_asset_baseline(tmp_path):
    asset_root = tmp_path / "shared"
    task_a = PersistentWorkspace(tmp_path / "task-a/workspace")
    task_a.write_file("tool.py", "def run(payload):\n    return {'value': 1}\n")
    adapter_a = SharedAdapter(); library_a = CapabilityLibrary(asset_root / "tools", task_a.root,
        python=sys.executable, scope_id="shared01")
    manager_a = CapabilityManager(asset_root=asset_root, workspace=task_a, adapter=adapter_a,
                                  tool_library=library_a)
    # Task A first executes a controller without the missing capability and fails.
    task_a.write_file("controller.py", "def run(robot):\n    robot.act({'value': 0})\n    return robot.verify('goal', {})\n")
    failed = ControllerRuntime(timeout_seconds=10).execute(task_a.controller, adapter_a)
    assert adapter_a.sensor_report(failed)["sensor_success"] is False
    registered = manager_a.register_tool(name="task_target", source_path="tool.py",
        description="returns the task target", source_urls=["https://example.com/task-target"],
        trained_on_current_task=False,
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {"value": {"type": "integer"}},
                       "required": ["value"], "additionalProperties": False})
    manager_a.test_tool(registered["tool_id"], [{"input": {}, "expected": {"value": 1}}])
    task_a.write_file("controller.py", "def run(robot):\n    target = robot.use('task_target:v001', {})\n    robot.act(target)\n    return robot.verify('goal', {})\n")
    evidence = committed_execution(task_a, adapter_a, manager_a)
    assert evidence["sensor_report"]["sensor_success"] is True
    manager_a.promote_asset(registered["tool_id"], [evidence["artifact_uri"]])

    # Task B starts in an independent workspace and reuses the shared Tool on its first execution.
    task_b = PersistentWorkspace(tmp_path / "task-b/workspace"); adapter_b = SharedAdapter()
    library_b = CapabilityLibrary(asset_root / "tools", task_b.root, python=sys.executable, scope_id="shared01")
    manager_b = CapabilityManager(asset_root=asset_root, workspace=task_b, adapter=adapter_b,
                                  tool_library=library_b)
    assert adapter_b.capabilities == {}
    manager_b.search("task target")
    manager_b.inspect("task_target:v001")
    assert manager_b.activate_tool("task_target:v001")["tool_id"] == "task_target:v001"
    task_b.write_file("controller.py", "def run(robot):\n    target = robot.use('task_target:v001', {})\n    robot.act(target)\n    return robot.verify('goal', {})\n")
    reused = ControllerRuntime(timeout_seconds=10).execute(task_b.controller, adapter_b)
    assert adapter_b.sensor_report(reused)["sensor_success"] is True
    task_b_executions = 1
    baseline = PersistentWorkspace(tmp_path / "baseline/workspace"); baseline_adapter = SharedAdapter()
    baseline.write_file("controller.py", "def run(robot):\n    robot.act({'value': 0})\n    return robot.verify('goal', {})\n")
    first_baseline = ControllerRuntime(timeout_seconds=10).execute(baseline.controller, baseline_adapter)
    assert baseline_adapter.sensor_report(first_baseline)["sensor_success"] is False
    baseline.write_file("controller.py", "def run(robot):\n    robot.act({'value': 1})\n    return robot.verify('goal', {})\n")
    second_baseline = ControllerRuntime(timeout_seconds=10).execute(baseline.controller, baseline_adapter)
    assert baseline_adapter.sensor_report(second_baseline)["sensor_success"] is True
    no_asset_baseline_executions = 2
    assert task_b_executions < no_asset_baseline_executions


def test_checkpoint_restores_snapshot_and_event_store_keeps_identical_events(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    workspace.write_file("controller.py", "def run(robot):\n    return 1\n")
    snapshot = workspace.snapshot()
    workspace.write_file("controller.py", "def run(robot):\n    return 2\n")
    workspace.restore(snapshot.snapshot_id)
    assert "return 1" in workspace.read("controller.py")
    store = EventStore(tmp_path / "events")
    first = store.commit("evidence", {"same": True})
    second = store.commit("evidence", {"same": True})
    assert first.event_id != second.event_id
    assert len(store.events()) == 2


def test_capability_manager_builds_acquired_bundle_in_workspace(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    workspace.write_file("bundle/module.py", "value = 1\n")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace, adapter=FakeAdapter())
    result = manager.build("bundle")
    assert result["build"]["exit_code"] == 0


def test_restart_resumes_after_committed_execution_without_repeating_action(tmp_path):
    adapter = FakeAdapter(); workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace, adapter=adapter)
    context = ContextBuilder(adapter_index={"protocol": "fake-v1"}, asset_registry=None, workspace=workspace)
    first = AgentLoop(model=CrashModel(), workspace=workspace, adapter=adapter,
                      context_builder=context, capability_manager=manager,
                      runtime=ControllerRuntime(timeout_seconds=10), event_store=EventStore(tmp_path), root=tmp_path)
    try: first.run()
    except RuntimeError as exc: assert "crash" in str(exc)
    assert len([event for event in EventStore(tmp_path).events() if event["kind"] == "execution"]) == 1
    resumed_adapter = FakeAdapter(); resumed_manager = CapabilityManager(asset_root=tmp_path / "assets",
        workspace=workspace, adapter=resumed_adapter)
    result = AgentLoop(model=CrashModel(resume=True), workspace=workspace, adapter=resumed_adapter,
        context_builder=ContextBuilder(adapter_index={"protocol": "fake-v1"}, asset_registry=None, workspace=workspace),
        capability_manager=resumed_manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path), root=tmp_path).run()
    assert result["finished"] is True
    assert len([event for event in EventStore(tmp_path).events() if event["kind"] == "execution"]) == 1
    assert resumed_adapter.value == 0


def test_benchmark_policies_execute_before_after_and_sealed_evaluator(tmp_path):
    from evaluation.runner import BenchmarkRunner

    adapter = SealedAdapter(); workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace, adapter=adapter)
    policies = [AntiCheatingPolicy(name="anti_cheating"), GeneralizationPolicy(name="generalization"),
                ProvenancePolicy(name="provenance"), SealedEvaluationPolicy(name="sealed_evaluation")]
    loop = AgentLoop(model=FakeModel(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index={"protocol": "fake-v1"}, asset_registry=None, workspace=workspace),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path), root=tmp_path)
    runner = BenchmarkRunner(loop, policies)
    result = runner.run()
    assert result["sealed_evaluation"] is True
    assert runner.latest_evidence is not None
    assert runner.latest_evidence.payload["verification_receipt"]["verified"] is True
    records = [event["payload"] for event in EventStore(tmp_path).events()
               if event["kind"] == "evaluation_policy"]
    assert {(item["policy"], item["phase"]) for item in records} == {
        (name, phase) for name in ("anti_cheating", "generalization", "provenance", "sealed_evaluation")
        for phase in ("before_run", "after_run")}


def test_sealed_evaluation_checks_every_campaign_case(tmp_path):
    from embodied_codex.kernel.campaign import CampaignAdapter

    first, second = SealedAdapter(), SealedAdapter()
    first.value = second.value = 1
    adapter = CampaignAdapter((("A", first), ("B", second)))
    result = {}
    loop = type("Loop", (), {"adapter": adapter,
        "event_store": EventStore(tmp_path / "events")})()
    SealedEvaluationPolicy(name="sealed_evaluation").after_run(loop, result)
    assert result["evaluation_passed"] is True
    assert result["sealed_evaluation_cases"] == [
        {"case": "A", "passed": True}, {"case": "B", "passed": True}]
    assert first.sealed is second.sealed is True
