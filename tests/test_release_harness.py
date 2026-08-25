import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget, ProtocolError
from embodied_codex.kernel.assets import (CapabilityGapLibrary, CapabilityLibrary,
    ExperienceLibrary, SkillLibrary)
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.sandbox import BubblewrapBackend
from embodied_codex.kernel.workspace import PersistentWorkspace


class FinishModel:
    def decide(self, *, messages, tools):
        return {"content": "", "tool_calls": [{"id": "finish", "name": "finish",
            "arguments": json.dumps({"summary": "model says success"})}]}


def loop_at(tmp_path, adapter, model=None, *, resume=False, steps=2):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    tools = CapabilityLibrary(tmp_path / "assets/tools", workspace.root, python=sys.executable,
                              allowed_input_roots=[workspace.root, tmp_path / "evidence",
                                                   adapter.artifact_dir])
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
        adapter=adapter, tool_library=tools,
        skill_library=SkillLibrary(tmp_path / "assets/skills"),
        experience_library=ExperienceLibrary(tmp_path / "assets/experiences"),
        gap_library=CapabilityGapLibrary(tmp_path / "assets/gaps"))
    return AgentLoop(model=model or FinishModel(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index, asset_registry=manager,
                                       workspace=workspace), capability_manager=manager,
        runtime=ControllerRuntime(timeout_seconds=10), event_store=EventStore(tmp_path),
        root=tmp_path, budget=LoopBudget(max_steps=steps, max_executions=4), resume=resume)


def test_completion_gate_rejects_model_claim_failure_and_stale_controller(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    with pytest.raises(ProtocolError, match="not been executed"):
        loop._finish("unsupported")

    loop.workspace.write_file("controller.py",
        "def run(robot):\n    robot.act({'type':'set_value','value':0})\n    return robot.verify('target', {})\n")
    failed = loop._run_controller()
    assert failed["verification_receipt"]["verified"] is False
    with pytest.raises(ProtocolError, match="no successful Adapter verification"):
        loop._finish("unsupported")

    loop.workspace.write_file("controller.py",
        "def run(robot):\n    robot.act({'type':'set_value','value':1})\n    return robot.verify('target', {})\n")
    assert loop._run_controller()["verification_receipt"]["verified"] is True
    loop.workspace.write_file("controller.py", "def run(robot):\n    return {'changed': True}\n")
    with pytest.raises(ProtocolError, match="older Controller"):
        loop._finish("stale")


def test_completion_gate_rejects_explicit_sensor_report_failure(tmp_path):
    class FailedReportAdapter(FakeAdapter):
        def sensor_report(self, execution):
            return {"sensor_success": False, "reason": "independent sensor rejected"}
    adapter = FailedReportAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.workspace.write_file("controller.py",
        "def run(robot):\n    robot.act({'type':'set_value','value':1})\n    return robot.verify('target', {})\n")
    loop._run_controller()
    with pytest.raises(ProtocolError, match="sensor report"):
        loop._finish("receipt alone is insufficient")


def test_reset_adapter_invalidates_checkpoint_and_forces_new_execution(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.current_task = adapter.instruction
    loop.workspace.write_file("controller.py",
        "def run(robot):\n    robot.act({'type':'set_value','value':1})\n    return robot.verify('target', {})\n")
    loop._run_controller(); loop._checkpoint(); adapter.reset_generation(); adapter.close()

    reset = FakeAdapter("set target", tmp_path)
    resumed = loop_at(tmp_path, reset, resume=True)
    assert resumed.state["restored_evidence_unverified"] is True
    with pytest.raises(ProtocolError, match="current Adapter generation"):
        resumed._finish("stale physical state")
    resumed._run_controller()
    assert len([row for row in EventStore(tmp_path).events() if row["kind"] == "execution"]) == 2


def test_failed_tool_never_binds_and_internal_tool_needs_no_benchmark_policy(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.workspace.write_file("tool.py", "def run(payload):\n    return {'value': 0}\n")
    registered = loop.capability_manager.register_tool(name="internal_target", source_path="tool.py",
        description="private deterministic implementation", input_schema={"type": "object",
        "properties": {}, "additionalProperties": False}, output_schema={"type": "object",
        "properties": {"value": {"type": "integer"}}, "required": ["value"],
        "additionalProperties": False}, trained_on_current_task=True, source_urls=[])
    assert registered["status"] == "registered"
    with pytest.raises(Exception, match="tests failed"):
        loop.capability_manager.test_tool(registered["tool_id"],
            [{"input": {}, "expected": {"value": 1}}])
    assert registered["tool_id"] not in adapter.capabilities


def test_checkpoint_package_build_test_bind_and_controller_execution(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    checkpoint = b"deterministic-public-weights"
    loop.workspace.write_file("bundle/weights.bin", checkpoint.decode())
    loop.workspace.write_file("bundle/tool.py",
        "from pathlib import Path\ndef run(payload):\n    Path(__file__).with_name('weights.bin').read_bytes()\n    return {'value': 1}\n")
    build = loop.capability_manager.build("bundle",
        [sys.executable, "-c", "from pathlib import Path; Path('built.txt').write_text('ok')"])
    assert build["build"]["exit_code"] == 0
    assert (loop.workspace.root / "bundle/built.txt").read_text() == "ok"
    registered = loop.capability_manager.register_package(name="checkpoint_target",
        bundle_path="bundle", description="checkpoint-backed target model",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        output_schema={"type": "object", "properties": {"value": {"type": "integer"}},
            "required": ["value"], "additionalProperties": False},
        package_spec={"kind": "model", "entrypoint": "tool.py", "accelerator": "cpu",
            "checkpoint_sha256": {"weights.bin": hashlib.sha256(checkpoint).hexdigest()}})
    tested = loop.capability_manager.test_tool(registered["tool_id"],
        [{"input": {}, "expected": {"value": 1}}])
    assert tested["bound"] is True and registered["tool_id"] in adapter.capabilities
    loop.workspace.write_file("controller.py", "def run(robot):\n"
        f"    target=robot.use('{registered['tool_id']}', {{}})\n"
        "    robot.act({'type':'set_value','value':target['value']})\n"
        "    return robot.verify('target', {})\n")
    evidence = loop._run_controller()
    assert evidence["verification_receipt"]["verified"] is True


def test_context_bounds_large_tool_output_and_image_is_multimodal(tmp_path):
    class LargeReadModel:
        def __init__(self): self.turn = 0; self.observed = []
        def decide(self, *, messages, tools):
            self.turn += 1; self.observed.append(len(json.dumps(messages)))
            name = "read_file" if self.turn == 1 else "finish"
            arguments = {"path": "large.txt"} if self.turn == 1 else {"summary": "invalid"}
            return {"content": "", "tool_calls": [{"id": str(self.turn), "name": name,
                "arguments": json.dumps(arguments)}]}
    adapter = FakeAdapter("set target", tmp_path)
    model = LargeReadModel(); loop = loop_at(tmp_path, adapter, model=model, steps=3)
    loop.workspace.write_file("large.txt", "x" * 2_000_000)
    result = loop.run(adapter.instruction)
    assert result["finished"] is False
    assert max(model.observed) <= loop.max_context_chars + 5000
    tool_messages = [row for row in loop.messages if row.get("role") == "tool"]
    assert any(json.loads(row["content"]).get("truncated") is True for row in tool_messages)

    image = np.zeros((20, 30, 3), np.uint8)
    path = loop.workspace.root / "sensor.png"; cv2.imwrite(str(path), image)
    with pytest.raises(Exception, match="binary files cannot be read as text"):
        loop.workspace.read_file("sensor.png")
    artifact = loop._view_artifact("sensor.png")
    assert artifact["kind"] == "image" and artifact["image_url"].startswith("data:image/png;base64,")
    assert "content" not in artifact


def test_tool_test_receipt_does_not_mutate_immutable_manifest(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.workspace.write_file("tool.py", "def run(payload):\n    return {'value': 1}\n")
    registered = loop.capability_manager.register_tool(name="receipt_target", source_path="tool.py",
        description="receipt target", input_schema={"type":"object","properties":{},"additionalProperties":False},
        output_schema={"type":"object","properties":{"value":{"type":"integer"}},"required":["value"],"additionalProperties":False})
    before = json.loads((tmp_path / "assets/tools/receipt_target/v001/manifest.json").read_text())
    loop.capability_manager.test_tool(registered["tool_id"], [{"input": {}, "expected": {"value": 1}}])
    after = json.loads((tmp_path / "assets/tools/receipt_target/v001/manifest.json").read_text())
    assert before == after
    assert (tmp_path / "assets/tools/_tests/receipt_target/v001/r001.json").is_file()


def test_sandbox_probe_executes_namespace_and_rejects_nonworking_binary():
    working = BubblewrapBackend().probe()
    assert working.available is True, working.detail
    broken = BubblewrapBackend(executable="/bin/false").probe()
    assert broken.available is False


def test_canonical_tool_schemas_are_strict_and_core_has_no_legacy_or_evaluation_import(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    assert all(item["function"]["parameters"].get("additionalProperties") is False
               for item in loop.tools.schemas)
    root = Path(__file__).parents[1] / "embodied_codex/kernel"
    for path in root.glob("*.py"):
        imports = [node for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
        encoded = " ".join(ast.dump(node).casefold() for node in imports)
        assert "evaluation" not in encoded and "legacy" not in encoded and "libero" not in encoded


def test_generic_cli_end_to_end_recovery_multicase_and_cross_task_reuse(tmp_path):
    adapter = "embodied_codex.fake_adapter:FakeAdapter"
    model = "embodied_codex.fake_adapter:FakeModel"
    base = [sys.executable, "-m", "embodied_codex", "run", "--adapter", adapter,
            "--profile", "autonomous", "--model", model,
            "--max-steps", "20"]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1]),
               XDG_DATA_HOME=str(tmp_path / "shared-data"))
    assets = tmp_path / "shared-data/roboforge/assets"

    first = subprocess.run([*base, "--task", "task A", "--run-dir", str(tmp_path / "run-a")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert first.returncode == 0, first.stdout
    first_result = json.loads(first.stdout)
    assert first_result["finished"] is True and first_result["executions"] == 2
    assert (assets / "tools/fake_target/v001/manifest.json").is_file()
    assert (assets / "skills/verified_target_skill/v001/controller.py").is_file()
    assert (assets / "experiences/verified_target_repair/v001/manifest.json").is_file()

    resumed = subprocess.run([*base, "--task", "task A", "--run-dir", str(tmp_path / "run-a")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert resumed.returncode == 0, resumed.stdout
    assert json.loads(resumed.stdout)["executions"] == 2

    second = subprocess.run([*base, "--task", "task B", "--run-dir", str(tmp_path / "run-b")],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert second.returncode == 0, second.stdout
    second_result = json.loads(second.stdout)
    assert second_result["executions"] == 1 < first_result["executions"]
    search_events = [json.loads(line) for line in (tmp_path / "run-b/events.jsonl").read_text().splitlines()]
    search = next(row for row in search_events if row["kind"] == "tool_result"
                  and row["payload"]["name"] == "search_assets")
    assets = search["payload"]["payload"]["result"]
    assert assets["tools"] and assets["skills"] and assets["experiences"]

    multicase = subprocess.run([*base, "--task", "multi case", "--run-dir", str(tmp_path / "multi"),
        "--states", "0", "1"], env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=90)
    assert multicase.returncode == 0, multicase.stdout
    multi_result = json.loads(multicase.stdout)
    assert multi_result["finished"] is True and len(multi_result["cases"]) == 2
    assert len({row["latest_evidence"]["controller_sha256"] for row in multi_result["cases"]}) == 1

    doctor = subprocess.run([sys.executable, "-m", "embodied_codex", "doctor", "--adapter", adapter,
        "--model", model, "--run-dir", str(tmp_path / "doctor")], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert doctor.returncode == 0, doctor.stdout
    report = json.loads(doctor.stdout)
    assert report["sandbox"]["available"] is True
    assert report["controller_runtime"] == report["tool_runtime"] == "available"
