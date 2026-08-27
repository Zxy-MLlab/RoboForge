import json
import math
import sys

import pytest

from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
from embodied_codex.deployments.libero import LiberoDeployment
from embodied_codex.kernel.agent_loop import AgentLoop, LoopBudget, ProtocolError
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.embodied_state import Frame, Pose, relative_pose_in_frames
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.workspace import PersistentWorkspace


def _loop(tmp_path, model):
    adapter = FakeAdapter("invariant", tmp_path / "adapter")
    workspace = PersistentWorkspace(tmp_path / "workspace")
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace, adapter=adapter)
    return AgentLoop(model=model, workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace,
            initial_observation=adapter.initial_observation()),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path / "events"), root=tmp_path,
        budget=LoopBudget(max_steps=1, max_executions=1), resume=False)


def _call(name, arguments):
    return {"content": "", "tool_calls": [{"id": "call-1", "name": name,
        "arguments": json.dumps(arguments)}]}


def test_harness_never_fabricates_model_decision_record(tmp_path):
    class Model:
        def decide(self, **_kwargs):
            return _call("write_file", {"path": "controller.py", "content": "x"})

    loop = _loop(tmp_path, Model())
    loop.run("invariant")
    assert not any(row.get("kind") == "decision_record" for row in loop.event_store.events())
    result = [row for row in loop.event_store.events() if row.get("kind") == "tool_result"][-1]
    assert "Decision Record is required" in json.dumps(result)


def test_validation_command_discards_filesystem_side_effects(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    workspace.write_file("controller.py", "original")
    result = workspace.run_validation(["python", "-c",
        "from pathlib import Path; Path('controller.py').write_text('changed'); Path('new.txt').write_text('x')"])
    assert result["exit_code"] == 0
    assert workspace.controller.read_text() == "original"
    assert not (workspace.root / "new.txt").exists()


def test_mutating_run_command_requires_decision_record(tmp_path):
    class Model:
        def decide(self, **_kwargs):
            return _call("run_command", {"argv": [sys.executable, "-c",
                "from pathlib import Path; Path('created.txt').write_text('x')"]})

    loop = _loop(tmp_path, Model())
    loop.run("invariant")
    assert not (loop.workspace.root / "created.txt").exists()
    assert not any(row.get("kind") == "decision_record" for row in loop.event_store.events())


def test_explicit_decision_allows_mutating_run_command(tmp_path):
    class Model:
        turn = 0
        def decide(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return _call("record_decision", {"goal": "mutate", "evidence_refs": [],
                    "hypothesis": None, "decision": "create file", "expected_effect": None,
                    "uncertainty": None})
            return _call("run_command", {"argv": [sys.executable, "-c",
                "from pathlib import Path; Path('created.txt').write_text('x')"]})

    loop = _loop(tmp_path, Model())
    loop.budget.max_steps = 2
    loop.run("invariant")
    assert (loop.workspace.root / "created.txt").read_text() == "x"


def test_missing_canonical_projection_fails_closed_for_arbitrary_native_names(tmp_path):
    loop = AgentLoop.__new__(AgentLoop)
    loop.adapter = object()
    with pytest.raises(ProtocolError, match="canonical_observation"):
        loop._canonical_observation({"tcp_state": [0, 0, 0], "joint_positions": [1],
                                     "rgb_frames": []})


def test_relative_pose_in_frames_separates_common_and_result_frames():
    quarter_turn = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))
    frames = {
        "world": Frame("world"),
        "camera": Frame("camera", "world", ((0, -1, 0, 2), (1, 0, 0, 0),
                                                (0, 0, 1, 0), (0, 0, 0, 1))),
    }
    parent = Pose("world", (1, 0, 0), quarter_turn)
    child = Pose("camera", (0, 0, 0), quarter_turn)
    relative = relative_pose_in_frames(parent, child, frames,
                                       common_frame="world", result_frame="parent_local")
    assert relative.frame == "parent_local"
    assert relative.position == pytest.approx((0.0, -1.0, 0.0))


def test_sdk_contract_is_authoritative_for_runtime_projection():
    contract = LIBERO_ROBOT_SDK_CONTRACT["methods"]
    assert {name: set(spec["output_fields"])
            for name, spec in contract.items()} == LiberoDeployment._OUTPUT_FIELDS


def test_one_model_decision_spans_workspace_validation_and_execution(tmp_path):
    class Model:
        turn = 0
        def decide(self, **_kwargs):
            self.turn += 1
            if self.turn == 1:
                return _call("record_decision", {"goal": "repair", "evidence_refs": [],
                    "hypothesis": "public check failed", "decision": "update controller",
                    "expected_effect": "verification changes", "uncertainty": None})
            if self.turn == 2:
                return _call("write_file", {"path": "controller.py",
                    "content": "def run(robot):\n    robot.act({'type':'set_value','value':1})\n    return robot.verify('target', {})\n"})
            if self.turn == 3:
                return _call("run_validation", {"argv": [sys.executable, "-m", "py_compile", "controller.py"]})
            if self.turn == 4:
                return _call("read_file", {"path": "controller.py"})
            return _call("run_controller", {})

    loop = _loop(tmp_path, Model())
    loop.budget.max_steps = 5
    loop.run("invariant")
    links = [row["payload"] for row in loop.event_store.events()
             if row.get("kind") == "tool_result" and row["payload"].get("decision_id")]
    assert len(links) == 4
    assert {item["decision_id"] for item in links} == {"decision-call-1"}
    record = loop._decision_records["decision-call-1"]
    assert record["status"] == "committed"
    assert record["hypothesis"] == "public check failed"
