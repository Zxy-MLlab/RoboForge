import ast
from pathlib import Path

from embodied_codex.kernel.agent_loop import AgentLoop
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace
from embodied_codex.kernel.assets import AssetRegistry


class FakeAdapter:
    instruction = "move the marker"
    def __init__(self): self.value = 0
    def dispatch(self, method, arguments):
        if method == "act": self.value = arguments["action"]["value"]
        if method == "verify": return {"verified": self.value == 1}
        if method == "observe": return {"value": self.value}
        if method == "record": return {"recorded": True}
        if method == "use": return {"tool_id": arguments["tool_id"], "result": {"value": 1}}
        raise ValueError(method)
    def project_rpc_output(self, method, arguments, result): return dict(result)
    def sensor_report(self, execution):
        return {"sensor_success": self.value == 1, "value": self.value}
    def close(self): pass


class MemoryAssets:
    def __init__(self): self.saved = []; self.reused = 0
    def search(self, query, limit=5): return {"tools": list(self.saved)}
    def inspect(self, asset_id): return next(item for item in self.saved if item["id"] == asset_id)
    def save(self, capability):
        item = {"id": f"{capability['kind']}-{len(self.saved)}", **capability}
        self.saved.append(item); return item


class FakeModel:
    def __init__(self): self.turn = 0
    def step(self, context):
        self.turn += 1
        if self.turn == 1:
            return {"changes": {"controller.py": "def run(robot):\n    robot.act({'value': 0})\n    return robot.verify('goal', {})\n"},
                    "execute": True}
        if self.turn == 2:
            assert context["latest_evidence"]["sensor_report"]["sensor_success"] is False
            return {"changes": {"controller.py": "def run(robot):\n    robot.act({'value': 1})\n    return robot.verify('goal', {})\n"},
                    "execute": True}
        if self.turn == 3:
            return {"capability": {"kind": "experience", "summary": "set marker to one"},
                    "finish": True}
        return {"finish": True}


def test_kernel_failure_evidence_edit_and_asset_persistence(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    assets = MemoryAssets(); adapter = FakeAdapter()
    context = ContextBuilder(adapter_index={"protocol": "fake-v1"}, asset_registry=assets,
                             workspace=workspace)
    result = AgentLoop(agent=FakeModel(), workspace=workspace, adapter=adapter,
                       context_builder=context, asset_registry=assets,
                       runtime=ControllerRuntime(timeout_seconds=10),
                       event_store=EventStore(tmp_path), root=str(tmp_path)).run()
    assert result["executions"] == 2
    assert result["latest_evidence"]["sensor_report"]["sensor_success"] is True
    assert len(assets.saved) == 1
    assert len(EventStore(tmp_path).events()) >= 4


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
