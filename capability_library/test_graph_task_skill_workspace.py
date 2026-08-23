import json

import pytest

from controller_graph_workspace import ControllerGraphWorkspace
from graph_task_skill_workspace import (
    GraphTaskSkillValidationError,
    GraphTaskSkillWorkspace,
)
from stage_node_workspace import StageNodeWorkspace


PYTHON = "/data/zxy/envs/vla-report/bin/python"


def _candidate(tmp_path):
    capabilities = tmp_path / "tools"
    capabilities.mkdir()
    nodes = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON)
    observe = nodes.create(
        name="observe_live_scene", stage_kind="observation", description="observe",
        requires=[], provides_by_outcome={"observed": ["frame_id"]},
        source='''def run_stage(robot, context):
    frame = robot.observe()
    return {"outcome": "observed", "updates": {"frame_id": frame["frame_id"]}}
''',
    )["node_id"]
    verify = nodes.create(
        name="visual_task_verification", stage_kind="visual_verification",
        description="verify", requires=["frame_id"],
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
        name="frozen_graph", description="graph", entry_node="observe",
        bindings={"observe": observe, "verify": verify},
        edges=[
            {"from": "observe", "outcome": "observed", "to": "verify"},
            {"from": "verify", "outcome": "verified", "to": "$success"},
            {"from": "verify", "outcome": "rejected", "to": "$failure"},
        ],
    )
    skills = GraphTaskSkillWorkspace(
        tmp_path / "skills", graph_workspace=graphs,
        capability_workspace=capabilities, python=PYTHON,
    )
    created = skills.create_candidate(
        name="learned_graph_skill", description="sensor graph",
        semantic_task="verify a live object", graph_id=graph["graph_id"],
        development_evidence={
            "sensor_only_conclusion": "sensor_verification_passed",
            "attachment_verified": True, "placement_verified": True,
            "controller_graph": {
                "verified_prefix_aliases": ["observe", "verify"],
                "node_trace": [{"alias": "observe"}, {"alias": "verify"}],
            },
        },
        development_context={
            "environment": "libero_spatial",
            "state_key": "task-5:state-23:seed-7",
        },
    )
    return skills, created


def test_graph_task_skill_freezes_and_executes_full_dependency_closure(tmp_path):
    skills, created = _candidate(tmp_path)
    inspected = skills.inspect(created["skill_id"])
    assert inspected["manifest"]["protocol"] == "embodied-graph-task-skill-v1"
    assert len(inspected["manifest"]["nodes"]) == 2

    def dispatch(method, arguments):
        if method == "observe":
            return {"frame_id": "frame-1"}
        if method == "call_tool":
            return {"verified": True}
        raise AssertionError(method)

    report = skills.execute(created["skill_id"], dispatch)
    assert report["graph_outcome"] == "success"
    assert report["verified_prefix_aliases"] == ["observe", "verify"]


def test_graph_task_skill_requires_three_distinct_unseen_states(tmp_path):
    skills, created = _candidate(tmp_path)
    with pytest.raises(GraphTaskSkillValidationError, match="development state"):
        skills.record_unseen_validation(
            created["skill_id"], environment="libero_spatial",
            state_key="task-5:state-23:seed-7",
            sensor_evidence={"sensor_only_conclusion": "sensor_verification_passed"},
        )
    for state in (4, 11, 18):
        result = skills.record_unseen_validation(
            created["skill_id"], environment="libero_spatial",
            state_key=f"task-5:state-{state}:seed-7",
            sensor_evidence={"sensor_only_conclusion": "sensor_verification_passed"},
        )
    assert result["status"] == "sensor_validated"


def test_graph_task_skill_rejects_partially_verified_graph(tmp_path):
    capabilities = tmp_path / "tools"
    capabilities.mkdir()
    nodes = StageNodeWorkspace(tmp_path / "nodes", python=PYTHON)
    node = nodes.create(
        name="only_node", stage_kind="observation", description="node",
        requires=[], provides_by_outcome={"done": []},
        source='''def run_stage(robot, context):
    return {"outcome": "done", "updates": {}}
''',
    )["node_id"]
    graphs = ControllerGraphWorkspace(tmp_path / "graphs", nodes=nodes)
    graph = graphs.create(
        name="partial_graph", description="graph", entry_node="node",
        bindings={"node": node},
        edges=[{"from": "node", "outcome": "done", "to": "$success"}],
    )
    skills = GraphTaskSkillWorkspace(
        tmp_path / "skills", graph_workspace=graphs,
        capability_workspace=capabilities,
    )
    with pytest.raises(GraphTaskSkillValidationError, match="every executed"):
        skills.create_candidate(
            name="partial_skill", description="partial", semantic_task="task",
            graph_id=graph["graph_id"],
            development_evidence={
                "sensor_only_conclusion": "sensor_verification_passed",
                "controller_graph": {
                    "verified_prefix_aliases": [],
                    "node_trace": [{"alias": "node"}],
                },
            },
        )
