import ast
from pathlib import Path

import pytest

from embodied_harness.errors import GraphCompileError, NodeCompileError
from embodied_harness.graph_store import GraphStore
from embodied_harness.node_store import NodeStore


PYTHON = "/data/zxy/envs/vla-report/bin/python"


class FakeAdapter:
    def __init__(self, *, move_reaches=False):
        self.x = 0.0; self.move_reaches = move_reaches; self.closed = False
        self.calls = []

    @property
    def initial_context(self): return {"task_instruction": "move right"}

    def dispatch(self, method, arguments):
        self.calls.append(method)
        if method == "instruction": return "move right"
        if method == "sense": return {"frame_id": "frame-1", "x": self.x}
        if method == "verify":
            verifier = arguments["verifier"]
            return {"verified": verifier == "scene_visible" or (
                verifier == "goal_reached" and self.x >= 1.0
            )}
        if method == "act":
            if self.move_reaches: self.x = float(arguments["action"]["target_x"])
            return {"reached": self.move_reaches, "x": self.x}
        if method == "record": return {"recorded": True}
        raise AssertionError(method)

    def sensor_report(self, execution):
        return {"completed": execution.get("completed"), "x": self.x}

    def close(self): self.closed = True


def _stores(tmp_path):
    nodes = NodeStore(tmp_path / "nodes", python=PYTHON, timeout_seconds=5)
    observe = nodes.create(
        name="observe_scene", kind="observation", description="live observation",
        requires=["task_instruction"],
        provides_by_outcome={"observed": ["frame_id", "start_x"]},
        source='''def run_stage(adapter, context):
    frame = adapter.sense("rgbd", {})
    return {"outcome": "observed", "updates": {
        "frame_id": frame["frame_id"], "start_x": frame["x"]}}
''',
    )["node_id"]
    verify = nodes.create(
        name="verify_scene", kind="verification", description="adapter proof",
        requires=["frame_id"],
        provides_by_outcome={"verified": ["scene_verified"],
                             "rejected": ["failure_reason"]},
        checkpoint_outcomes=["verified"],
        source='''def run_stage(adapter, context):
    result = adapter.verify("scene_visible", {"frame_id": context["frame_id"]})
    if result.get("verified") is True:
        return {"outcome": "verified", "updates": {"scene_verified": True}}
    return {"outcome": "rejected", "updates": {"failure_reason": "not visible"}}
''',
    )["node_id"]
    move_v1 = nodes.create(
        name="execute_move", kind="motion", description="first attempt",
        requires=["start_x", "scene_verified"],
        provides_by_outcome={"moved": ["action_result"],
                             "failed": ["failure_reason"]},
        source='''def run_stage(adapter, context):
    result = adapter.act({"target_x": context["start_x"] + 1.0})
    if result.get("reached") is True:
        return {"outcome": "moved", "updates": {"action_result": result}}
    return {"outcome": "failed", "updates": {"failure_reason": "unreached"}}
''',
    )["node_id"]
    goal = nodes.create(
        name="verify_goal", kind="verification", description="final sensor proof",
        requires=["action_result"],
        provides_by_outcome={"reached": ["goal_evidence"],
                             "missed": ["failure_reason"]},
        checkpoint_outcomes=["reached"],
        source='''def run_stage(adapter, context):
    result = adapter.verify("goal_reached", {})
    if result.get("verified") is True:
        return {"outcome": "reached", "updates": {"goal_evidence": result}}
    return {"outcome": "missed", "updates": {"failure_reason": "not reached"}}
''',
    )["node_id"]
    graphs = GraphStore(
        tmp_path / "graphs", nodes=nodes,
        available_initial_fields={"task_instruction"},
    )
    return nodes, graphs, observe, verify, move_v1, goal


def _graph(graphs, *, name, observe, verify, move, goal, base=None, frozen=None):
    return graphs.create(
        name=name, description="observe verify then move", entry="observe",
        bindings={"observe": observe, "verify": verify, "move": move, "goal": goal},
        edges=[
            {"from": "observe", "outcome": "observed", "to": "verify"},
            {"from": "verify", "outcome": "verified", "to": "move"},
            {"from": "verify", "outcome": "rejected", "to": "$failure"},
            {"from": "move", "outcome": "moved", "to": "goal"},
            {"from": "move", "outcome": "failed", "to": "$failure"},
            {"from": "goal", "outcome": "reached", "to": "$success"},
            {"from": "goal", "outcome": "missed", "to": "$failure"},
        ],
        initial_fields=["task_instruction"], base_graph_id=base,
        frozen_aliases=frozen,
    )


def test_persistent_adapter_and_node_level_revision(tmp_path):
    nodes, graphs, observe, verify, move_v1, goal = _stores(tmp_path)
    base = _graph(graphs, name="base_graph", observe=observe,
                  verify=verify, move=move_v1, goal=goal)
    assert graphs.compile(base["graph_id"])["adapter_launched"] is False
    first_adapter = FakeAdapter(move_reaches=False)
    first = graphs.execute(base["graph_id"], first_adapter)
    assert first["completed"] is True
    assert first["graph_outcome"] == "failure"
    assert first["verified_prefix_aliases"] == ["observe", "verify"]
    assert first_adapter.calls == ["sense", "verify", "act"]

    move_v2 = nodes.create(
        name="execute_move", kind="motion", description="revised attempt",
        requires=["start_x", "scene_verified"],
        provides_by_outcome={"moved": ["action_result"],
                             "failed": ["failure_reason"]},
        source='''def run_stage(adapter, context):
    result = adapter.act({"target_x": context["start_x"] + 1.0})
    if result.get("reached") is True:
        return {"outcome": "moved", "updates": {"action_result": result}}
    return {"outcome": "failed", "updates": {"failure_reason": "unreached"}}
''',
    )["node_id"]
    revised = _graph(
        graphs, name="revised_graph", observe=observe, verify=verify,
        move=move_v2, goal=goal, base=base["graph_id"],
        frozen=["observe", "verify"],
    )
    manifest = graphs.inspect(revised["graph_id"])["manifest"]
    assert manifest["bindings"]["observe"] == observe
    assert manifest["bindings"]["verify"] == verify
    assert manifest["bindings"]["move"] == move_v2
    second_adapter = FakeAdapter(move_reaches=True)
    second = graphs.execute(revised["graph_id"], second_adapter)
    assert second["graph_outcome"] == "success"
    assert second_adapter.x == 1.0
    assert second_adapter.calls == ["sense", "verify", "act", "verify"]


def test_compiler_rejects_privilege_bad_returns_and_fake_initial_fields(tmp_path):
    nodes = NodeStore(tmp_path / "nodes", python=PYTHON)
    with pytest.raises(NodeCompileError, match="forbidden_name"):
        nodes.create(
            name="cheating_node", kind="observation", description="bad",
            requires=[], provides_by_outcome={"done": []},
            source='''def run_stage(adapter, context):
    reward = 1
    return {"outcome": "done", "updates": {}}
''',
        )
    with pytest.raises(NodeCompileError, match="outcome_must_be_literal"):
        nodes.create(
            name="dynamic_result", kind="observation", description="bad",
            requires=[], provides_by_outcome={"done": [], "failed": []},
            source='''def run_stage(adapter, context):
    outcome = "done"
    return {"outcome": outcome, "updates": {}}
''',
        )
    good = nodes.create(
        name="simple_node", kind="observation", description="good",
        requires=["fabricated_pose"], provides_by_outcome={"done": []},
        source='''def run_stage(adapter, context):
    return {"outcome": "done", "updates": {}}
''',
    )["node_id"]
    graphs = GraphStore(tmp_path / "graphs", nodes=nodes,
                       available_initial_fields={"task_instruction"},
                       require_verified_success=False)
    with pytest.raises(GraphCompileError, match="does not supply"):
        graphs.create(
            name="bad_initial_graph", description="bad", entry="only",
            bindings={"only": good},
            edges=[{"from": "only", "outcome": "done", "to": "$success"}],
            initial_fields=["fabricated_pose"],
        )


def test_standalone_core_has_no_legacy_runtime_imports():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("Thea", "capability_library", "controller_program_workspace",
                 "libero_robot_sdk", "run_groundingdino_controller")
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(str(node.module or ""))
        assert not any(
            module == token or module.startswith(token + ".")
            for module in imports for token in forbidden
        ), path
