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

from embodied_codex.adapters.factory import adapter_preflight
from embodied_codex.capabilities.graspnet_rgbd import GraspNetRGBD
from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget, ProtocolError
from embodied_codex.kernel.assets import (CapabilityGapLibrary, CapabilityLibrary,
    ExperienceLibrary, SkillLibrary)
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.recovery import RecoveryError, load_checkpoint, save_checkpoint
from embodied_codex.kernel.sandbox import BubblewrapBackend, PosixSandboxBackend
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


def test_checkpoint_checksum_rejects_tampered_payload(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.current_task = adapter.instruction
    loop._checkpoint()
    path = tmp_path / "checkpoint/state.json"
    envelope = json.loads(path.read_text())
    envelope["payload"]["steps"] = 999
    with pytest.raises(PermissionError):
        path.write_text(json.dumps(envelope))
    path.chmod(0o600)
    path.write_text(json.dumps(envelope))
    with pytest.raises(RecoveryError, match="checksum mismatch"):
        load_checkpoint(tmp_path)


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


def test_failed_execution_cannot_be_persisted_as_skill(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.workspace.write_file("controller.py",
        "def run(robot):\n    robot.act({'type':'set_value','value':0})\n"
        "    return robot.verify('target', {})\n")
    evidence = loop._run_controller()
    with pytest.raises(Exception, match="successful Adapter evidence"):
        loop.capability_manager.register_skill(name="failed_skill", task="set target",
            controller="controller.py", tool_ids=[],
            evidence_paths=[evidence["artifact_uri"]])


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


def test_tool_runtime_rejects_missing_pinned_dependency(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.workspace.write_file("tool.py", "def run(payload):\n    return {'value': 1}\n")
    registered = loop.capability_manager.register_tool(name="dependency_target",
        source_path="tool.py", description="dependency target",
        input_schema={"type":"object","properties":{},"additionalProperties":False},
        output_schema={"type":"object","properties":{"value":{"type":"integer"}},
            "required":["value"],"additionalProperties":False},
        runtime_requirements=["package-that-cannot-exist-roboforge==1.0.0"])
    with pytest.raises(Exception, match="contract tests failed"):
        loop.capability_manager.test_tool(registered["tool_id"],
            [{"input": {}, "expected": {"value": 1}}])
    assert registered["tool_id"] not in adapter.capabilities


def test_default_sandbox_is_rootless_and_bubblewrap_is_only_optional():
    working = PosixSandboxBackend().probe()
    assert working.available is True, working.detail
    assert working.features["requires_root"] is False
    assert working.features["uses_user_namespace"] is False
    assert working.features["no_new_privs"] is True
    assert working.features["seccomp"] is True
    broken = BubblewrapBackend(executable="/bin/false").probe()
    assert broken.available is False


def test_controller_and_tool_network_syscalls_are_denied(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    loop.workspace.write_file("controller.py", "import socket\n"
        "def run(robot):\n    socket.socket()\n")
    execution = loop.runtime.execute(loop.workspace.controller, adapter)
    assert execution["completed"] is False
    assert "PermissionError" in execution["error"]

    loop.workspace.write_file("network_tool.py", "import socket\n"
        "def run(payload):\n    socket.socket()\n    return {'unexpected': True}\n")
    registered = loop.capability_manager.register_tool(name="network_tool",
        source_path="network_tool.py", description="must be denied network access",
        input_schema={"type":"object","properties":{},"additionalProperties":False},
        output_schema={"type":"object","properties":{"unexpected":{"type":"boolean"}},
            "required":["unexpected"],"additionalProperties":False})
    with pytest.raises(Exception, match="contract tests failed"):
        loop.capability_manager.test_tool(registered["tool_id"],
            [{"input": {}, "expected": {"unexpected": True}}])
    assert registered["tool_id"] not in adapter.capabilities


def test_workspace_command_is_staged_and_cannot_modify_run_state(tmp_path):
    run_root = tmp_path / "run"
    workspace = PersistentWorkspace(run_root / "workspace")
    shared_assets = tmp_path / "shared-assets"
    shared_assets.mkdir()
    shared_manifest = shared_assets / "manifest.json"
    shared_manifest.write_text('{"immutable": true}\n')
    workspace.add_protected_path(shared_assets)
    workspace.write_file("controller.py", "def run(robot):\n    return 1\n")
    controller_sha = hashlib.sha256(workspace.controller.read_bytes()).hexdigest()
    workspace.lock_file("controller.py", controller_sha)
    checkpoint = save_checkpoint(run_root, {"step": 1})
    events = EventStore(run_root / "events", protect=True)
    events.commit("before", {"value": 1})
    evidence_dir = run_root / "evidence"; evidence_dir.mkdir()
    evidence = evidence_dir / "receipt.json"; evidence.write_text('{"verified": true}\n')
    evidence.chmod(0o400); evidence_dir.chmod(0o500)
    protected = [checkpoint, events.path, evidence, workspace.controller,
                 shared_manifest]
    script = ("from pathlib import Path\n"
        f"paths={repr([str(path) for path in protected])}\n"
        "for value in paths:\n"
        " try: Path(value).write_text('tampered'); raise RuntimeError('write escaped stage')\n"
        " except PermissionError: pass\n"
        "Path('generated.txt').write_text('committed')\n")
    result = workspace.run_command([sys.executable, "-c", script], timeout_seconds=20)
    assert result["exit_code"] == 0 and result["committed"] is True
    assert workspace.read("generated.txt") == "committed"
    assert hashlib.sha256(workspace.controller.read_bytes()).hexdigest() == controller_sha
    assert load_checkpoint(run_root) == {"step": 1}
    assert events.events()[0]["payload"] == {"value": 1}
    assert json.loads(evidence.read_text()) == {"verified": True}
    assert json.loads(shared_manifest.read_text()) == {"immutable": True}
    assert workspace.stage_root != workspace.root
    assert workspace.snapshot_root != workspace.root

    with pytest.raises(Exception, match="immutable file"):
        workspace.run_command([sys.executable, "-c",
            "from pathlib import Path; Path('controller.py').write_text('changed')"])
    assert hashlib.sha256(workspace.controller.read_bytes()).hexdigest() == controller_sha


def test_controller_runtime_cannot_mutate_harness_or_shared_asset_state(tmp_path):
    run_root = tmp_path / "run"
    workspace = PersistentWorkspace(run_root / "workspace")
    adapter = FakeAdapter("set target", run_root)
    checkpoint = save_checkpoint(run_root, {"step": 1})
    asset_root = tmp_path / "shared-assets"
    asset_root.mkdir()
    manifest = asset_root / "manifest.json"
    manifest.write_text('{"immutable": true}\n')
    source = ("from pathlib import Path\n"
        "def run(robot):\n"
        f"    protected={repr([str(checkpoint), str(manifest)])}\n"
        "    for value in protected:\n"
        "        try:\n"
        "            Path(value).write_text('tampered')\n"
        "            raise RuntimeError('write escaped sandbox')\n"
        "        except PermissionError:\n"
        "            pass\n"
        "    robot.act({'type':'set_value','value':1})\n"
        "    return robot.verify('target', {})\n")
    workspace.write_file("controller.py", source)
    runtime = ControllerRuntime(timeout_seconds=20,
                                protected_paths=[asset_root])
    try:
        execution = runtime.execute(workspace.controller, adapter)
    finally:
        adapter.close()
    assert execution["completed"] is True and execution["error"] is None
    assert load_checkpoint(run_root) == {"step": 1}
    assert json.loads(manifest.read_text()) == {"immutable": True}
    assert json.loads((run_root / "adapter/environment_state.json").read_text())["value"] == 1


def test_failed_stage_is_not_committed_and_timeout_kills_process_group(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "run/workspace")
    failed = workspace.run_command([sys.executable, "-c",
        "from pathlib import Path; Path('leak.txt').write_text('no'); raise SystemExit(3)"])
    assert failed["exit_code"] == 3 and failed["committed"] is False
    assert not (workspace.root / "leak.txt").exists()

    timed = workspace.run_command([sys.executable, "-c",
        "import subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c',"
        "'import time;time.sleep(30)']); print(child.pid,flush=True); time.sleep(30)"],
        timeout_seconds=0.5)
    assert timed["timed_out"] is True and timed["committed"] is False
    child_pid = int(timed["output"].strip().splitlines()[0])
    for _ in range(50):
        if not Path(f"/proc/{child_pid}").exists(): break
        import time
        time.sleep(0.02)
    assert not Path(f"/proc/{child_pid}").exists()


def test_unsafe_backend_is_rejected_outside_dev(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "embodied_codex", "run",
        "--adapter", "embodied_codex.fake_adapter:FakeAdapter", "--model",
        "embodied_codex.fake_adapter:FakeModel", "--task", "unsafe",
        "--profile", "autonomous", "--sandbox", "unsafe",
        "--run-dir", str(tmp_path / "run")], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=30,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1])))
    assert completed.returncode != 0
    assert "unsafe sandbox is permitted only" in completed.stdout


def test_libero_preflight_reports_structured_missing_runtime(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("ROBOFORGE_DEVICE", "cpu")
    monkeypatch.setenv("ROBOFORGE_GROUNDINGDINO_ROOT", str(missing / "dino"))
    monkeypatch.setenv("ROBOFORGE_SAM_ROOT", str(missing / "sam"))
    monkeypatch.setenv("ROBOFORGE_GRASPNET_ROOT", str(missing / "graspnet"))
    monkeypatch.setenv("ROBOFORGE_GROUNDINGDINO_CHECKPOINT", str(missing / "dino.pth"))
    monkeypatch.setenv("ROBOFORGE_SAM_CHECKPOINT", str(missing / "sam.pth"))
    monkeypatch.setenv("ROBOFORGE_GRASPNET_CHECKPOINT", str(missing / "graspnet.tar"))
    report = adapter_preflight("libero")
    assert report["ok"] is False
    assert set(report) == {"ok", "modules", "sources", "checkpoints", "accelerator"}
    assert all(item["available"] is False for item in report["checkpoints"].values())
    assert all(item["valid"] is False for item in report["checkpoints"].values())
    assert all(value is False for value in report["sources"].values())
    assert report["accelerator"] == {"requested": "cpu", "available": True}


def test_graspnet_wrapper_passes_external_source_root_to_real_backend(tmp_path):
    source_root = tmp_path / "vendor/graspnet"
    (source_root / "models").mkdir(parents=True)
    (source_root / "models/graspnet.py").write_text("# contract marker\n")
    checkpoint = tmp_path / "weights.tar"
    checkpoint.write_bytes(b"weights")
    backend = tmp_path / "backend.py"
    backend.write_text('''
import argparse
import json
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--source-root", required=True)
parser.add_argument("--downward-min")
parser.add_argument("--preferred-downward-min")
args = parser.parse_args()
Path(args.output).write_text(json.dumps({
    "grasps": [], "orientation_override_grasps": [],
    "filter_thresholds": {}, "filter_diagnostics": {"source_root": args.source_root}
}))
''')
    frame_dir = tmp_path / "frame"
    mask_dir = frame_dir / "masks"
    mask_dir.mkdir(parents=True)
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    depth = np.ones((20, 20), dtype=np.float32)
    mask = np.full((20, 20), 255, dtype=np.uint8)
    rgb_path = frame_dir / "rgb.png"
    depth_path = frame_dir / "depth.npy"
    mask_path = mask_dir / "target.png"
    assert cv2.imwrite(str(rgb_path), rgb)
    np.save(depth_path, depth)
    assert cv2.imwrite(str(mask_path), mask)
    capability = GraspNetRGBD(backend_script=backend, checkpoint=checkpoint,
                              source_root=source_root, python=sys.executable)
    result = capability.infer({"frame": {"frame_id": "frame-1", "cameras": {
        "agentview": {"rgb_path": str(rgb_path), "depth_path": str(depth_path),
            "intrinsic": [[10, 0, 10], [0, 10, 10], [0, 0, 1]],
            "camera_to_world": np.eye(4).tolist()}}},
        "detection": {"mask_path": str(mask_path), "box_xyxy": [0, 0, 19, 19]}})
    assert result["filter_diagnostics"]["source_root"] == str(source_root.resolve())


def test_canonical_tool_schemas_are_strict_and_core_has_no_legacy_or_evaluation_import(tmp_path):
    adapter = FakeAdapter("set target", tmp_path)
    loop = loop_at(tmp_path, adapter)
    assert all(item["function"]["parameters"].get("additionalProperties") is False
               for item in loop.tools.schemas)
    from embodied_codex.adapters.libero import _grasp_contract, _perception_contract
    def objects(value):
        if isinstance(value, dict):
            yield value
            for item in value.values():
                yield from objects(item)
        elif isinstance(value, list):
            for item in value:
                yield from objects(item)
    seed_contracts = [_perception_contract(), _grasp_contract()]
    assert not any(item.get("additionalProperties") is True
                   for contract in seed_contracts for item in objects(contract))
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
    search_events = [json.loads(line) for line in
                     (tmp_path / "run-b/events/events.jsonl").read_text().splitlines()]
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
