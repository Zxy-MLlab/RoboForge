import pytest

from controller_graph_workspace import (
    ControllerGraphValidationError,
    ControllerGraphWorkspace,
)
from stage_node_workspace import StageNodeWorkspace


PYTHON = "/data/zxy/envs/vla-report/bin/python"


def _node(workspace, name, requires, outcome, provides, body):
    return workspace.create(
        name=name, stage_kind=name, description=name, requires=requires,
        provides_by_outcome={outcome: provides}, source=body,
    )["node_id"]


def _workspaces(tmp_path):
    nodes = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON, timeout_sec=5)
    observe = _node(
        nodes, "observe_scene", [], "observed", ["instruction", "frame_id", "eef_xyz"],
        '''def run_stage(robot, context):
    instruction = robot.instruction()
    frame = robot.observe()
    return {"outcome": "observed", "updates": {
        "instruction": instruction, "frame_id": frame["frame_id"],
        "eef_xyz": frame["eef_xyz"]}}
''',
    )
    plan_v1 = _node(
        nodes, "plan_motion", ["frame_id", "eef_xyz"], "planned", ["target_xyz"],
        '''def run_stage(robot, context):
    robot.record({"frame_id": context["frame_id"]})
    eef = context["eef_xyz"]
    return {"outcome": "planned", "updates": {
        "target_xyz": [eef[0] + 0.1, eef[1], eef[2]]}}
''',
    )
    execute = _node(
        nodes, "execute_motion", ["target_xyz"], "completed", [],
'''def run_stage(robot, context):
    result = robot.act({"target_eef_xyz": context["target_xyz"], "gripper": -1})
    return {"outcome": "completed", "updates": {}}
''',
    )
    graphs = ControllerGraphWorkspace(tmp_path / "graphs", nodes=nodes)
    return nodes, graphs, observe, plan_v1, execute


def _graph(graphs, *, name, observe, plan, execute, base=None, frozen=None):
    return graphs.create(
        name=name, description="observe, plan, execute", entry_node="observe",
        bindings={"observe": observe, "plan": plan, "execute": execute},
        edges=[
            {"from": "observe", "outcome": "observed", "to": "plan"},
            {"from": "plan", "outcome": "planned", "to": "execute"},
            {"from": "execute", "outcome": "completed", "to": "$success"},
        ],
        base_graph_id=base, frozen_node_aliases=frozen or [],
    )


def test_graph_executes_typed_nodes_on_one_persistent_adapter(tmp_path):
    nodes, graphs, observe, plan, execute = _workspaces(tmp_path)
    created = _graph(
        graphs, name="pick_graph", observe=observe, plan=plan, execute=execute,
    )
    calls = []

    def dispatch(method, arguments):
        calls.append(method)
        if method == "instruction":
            return "pick up the bowl"
        if method == "observe":
            return {"frame_id": "frame-1", "eef_xyz": [0.1, 0.2, 0.3]}
        if method == "record":
            return {"recorded": True}
        if method == "act":
            return {"reached_target": True}
        raise AssertionError(method)

    report = graphs.execute(created["graph_id"], dispatch)
    assert report["execution_completed"] is True
    assert report["graph_outcome"] == "success"
    assert [item["alias"] for item in report["node_trace"]] == [
        "observe", "plan", "execute",
    ]
    assert calls == ["instruction", "observe", "record", "act"]


def test_graph_revision_freezes_node_id_and_incident_topology(tmp_path):
    nodes, graphs, observe, plan_v1, execute = _workspaces(tmp_path)
    base = _graph(
        graphs, name="base_graph", observe=observe, plan=plan_v1, execute=execute,
    )
    plan_v2 = _node(
        nodes, "plan_motion", ["frame_id", "eef_xyz"], "planned", ["target_xyz"],
        '''def run_stage(robot, context):
    eef = context["eef_xyz"]
    return {"outcome": "planned", "updates": {
        "target_xyz": [eef[0], eef[1] + 0.1, eef[2]]}}
''',
    )
    revised = _graph(
        graphs, name="revised_graph", observe=observe, plan=plan_v2,
        execute=execute, base=base["graph_id"], frozen=["observe"],
    )
    manifest = graphs.inspect(revised["graph_id"])["manifest"]
    assert manifest["bindings"]["observe"] == observe
    assert manifest["bindings"]["plan"] == plan_v2
    assert manifest["revision"]["frozen_node_ids"] == {"observe": observe}

    replacement_observe = _node(
        nodes, "observe_scene", [], "observed", ["instruction", "frame_id", "eef_xyz"],
        '''def run_stage(robot, context):
    frame = robot.observe()
    return {"outcome": "observed", "updates": {
        "instruction": robot.instruction(), "frame_id": frame["frame_id"],
        "eef_xyz": frame["eef_xyz"]}}
''',
    )
    with pytest.raises(ControllerGraphValidationError, match="frozen_node_replaced"):
        _graph(
            graphs, name="illegal_graph", observe=replacement_observe,
            plan=plan_v2, execute=execute, base=base["graph_id"], frozen=["observe"],
        )


def test_graph_rejects_missing_routes_and_context_contracts(tmp_path):
    nodes, graphs, observe, plan, execute = _workspaces(tmp_path)
    with pytest.raises(ControllerGraphValidationError, match="no routes"):
        graphs.create(
            name="missing_route_graph", description="bad", entry_node="observe",
            bindings={"observe": observe}, edges=[],
        )
    with pytest.raises(ControllerGraphValidationError, match="not guaranteed"):
        graphs.create(
            name="bad_context_graph", description="bad", entry_node="plan",
            bindings={"plan": plan, "execute": execute},
            edges=[
                {"from": "plan", "outcome": "planned", "to": "execute"},
                {"from": "execute", "outcome": "completed", "to": "$success"},
            ],
        )


def test_graph_visit_budget_bounds_cycles(tmp_path):
    nodes = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON, timeout_sec=5)
    loop = nodes.create(
        name="retry_stage", stage_kind="retry_stage", description="retry",
        requires=[], provides_by_outcome={"retry": [], "done": []},
        source='''def run_stage(robot, context):
    if context.get("finish"):
        return {"outcome": "done", "updates": {}}
    return {"outcome": "retry", "updates": {}}
''',
    )["node_id"]
    graphs = ControllerGraphWorkspace(tmp_path / "graphs", nodes=nodes)
    graph = graphs.create(
        name="bounded_retry_graph", description="bounded", entry_node="retry",
        bindings={"retry": loop},
        edges=[
            {"from": "retry", "outcome": "retry", "to": "retry"},
            {"from": "retry", "outcome": "done", "to": "$success"},
        ],
        max_node_visits=2,
    )
    report = graphs.execute(graph["graph_id"], lambda method, args: {})
    assert report["execution_completed"] is False
    assert "visit budget" in report["error"]


def test_visual_checkpoint_exports_structured_verified_prefix(tmp_path):
    nodes = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON, timeout_sec=5)
    observe = _node(
        nodes, "checkpoint_observe", [], "observed", ["frame_id"],
        '''def run_stage(robot, context):
    frame = robot.observe()
    return {"outcome": "observed", "updates": {"frame_id": frame["frame_id"]}}
''',
    )
    verify = nodes.create(
        name="attachment_verification", stage_kind="visual_verification",
        description="adapter-owned visual verification", requires=["frame_id"],
        provides_by_outcome={"verified": [], "rejected": []},
        checkpoint_outcomes=["verified"],
    source='''def run_stage(robot, context):
    result = robot.call_tool("verify_attachment", {"frame_id": context["frame_id"]})
    if result.get("verified") is True:
        return {"outcome": "verified", "updates": {}}
    return {"outcome": "rejected", "updates": {}}
''',
    )["node_id"]
    graphs = ControllerGraphWorkspace(tmp_path / "graphs", nodes=nodes)
    graph = graphs.create(
        name="checkpoint_graph", description="checkpoint", entry_node="observe",
        bindings={"observe": observe, "verify": verify},
        edges=[
            {"from": "observe", "outcome": "observed", "to": "verify"},
            {"from": "verify", "outcome": "verified", "to": "$success"},
            {"from": "verify", "outcome": "rejected", "to": "$failure"},
        ],
    )

    def dispatch(method, arguments):
        if method == "observe":
            return {"frame_id": "frame-1"}
        if method == "call_tool":
            return {"verified": True}
        raise AssertionError(method)

    report = graphs.execute(graph["graph_id"], dispatch)
    assert report["verified_prefix_aliases"] == ["observe", "verify"]


def test_checkpoint_cannot_ignore_failed_adapter_verification(tmp_path):
    nodes = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON, timeout_sec=5)
    dishonest = nodes.create(
        name="dishonest_verification", stage_kind="visual_verification",
        description="must not freeze without adapter proof", requires=[],
        provides_by_outcome={"verified": []}, checkpoint_outcomes=["verified"],
        source='''def run_stage(robot, context):
    robot.call_tool("verify_attachment", {"frame_id": "live-frame"})
    return {"outcome": "verified", "updates": {}}
''',
    )["node_id"]
    graphs = ControllerGraphWorkspace(tmp_path / "graphs", nodes=nodes)
    graph = graphs.create(
        name="dishonest_checkpoint_graph", description="must fail",
        entry_node="verify", bindings={"verify": dishonest},
        edges=[{"from": "verify", "outcome": "verified", "to": "$success"}],
    )

    report = graphs.execute(
        graph["graph_id"],
        lambda method, arguments: {"verified": False},
    )
    assert report["execution_completed"] is False
    assert report["verified_prefix_aliases"] == []
    assert "lacks adapter-owned verified:true" in report["error"]


def test_required_graph_revision_cannot_omit_frozen_nodes(tmp_path):
    nodes, graphs, observe, plan, execute = _workspaces(tmp_path)
    base = _graph(
        graphs, name="required_base", observe=observe, plan=plan, execute=execute,
    )
    constrained = ControllerGraphWorkspace(
        tmp_path / "graphs", nodes=nodes,
        required_revision={
            "base_graph_id": base["graph_id"],
            "frozen_node_aliases": ["observe"],
        },
    )
    with pytest.raises(ControllerGraphValidationError, match="required_base_graph"):
        _graph(
            constrained, name="unbased_revision", observe=observe,
            plan=plan, execute=execute,
        )


def test_task_graph_success_path_requires_verified_checkpoint(tmp_path):
    nodes, _, observe, plan, execute = _workspaces(tmp_path)
    graphs = ControllerGraphWorkspace(
        tmp_path / "task_graphs", nodes=nodes,
        require_checkpoint_success=True,
    )
    with pytest.raises(ControllerGraphValidationError, match="requires.*checkpoint"):
        _graph(
            graphs, name="unverified_success_graph", observe=observe,
            plan=plan, execute=execute,
        )


def test_graph_preflight_compiles_all_node_contracts_without_adapter(tmp_path):
    nodes, graphs, observe, plan, execute = _workspaces(tmp_path)
    graph = _graph(
        graphs, name="compiled_graph", observe=observe, plan=plan, execute=execute,
    )
    report = graphs.preflight(graph["graph_id"])
    assert report["eligible"] is True
    assert report["adapter_launched"] is False
    assert report["node_count"] == 3
    assert {row["alias"] for row in report["nodes"]} == {
        "observe", "plan", "execute",
    }


def test_graph_compiler_rejects_initial_fields_absent_from_adapter(tmp_path):
    nodes, _, observe, plan, execute = _workspaces(tmp_path)
    graphs = ControllerGraphWorkspace(
        tmp_path / "adapter_graphs", nodes=nodes,
        available_initial_context_fields={"task_instruction"},
    )
    with pytest.raises(ControllerGraphValidationError, match="not supplied.*Adapter"):
        graphs.create(
            name="fabricated_initial_graph", description="bad adapter input",
            entry_node="observe",
            bindings={"observe": observe, "plan": plan, "execute": execute},
            edges=[
                {"from": "observe", "outcome": "observed", "to": "plan"},
                {"from": "plan", "outcome": "planned", "to": "execute"},
                {"from": "execute", "outcome": "completed", "to": "$success"},
            ],
            initial_context_fields=["bowl_xyz"],
        )
