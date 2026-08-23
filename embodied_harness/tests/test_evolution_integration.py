import json
import fcntl
from pathlib import Path

import pytest

from embodied_harness.agent import Agent
from embodied_harness.evolution import EvolutionEngine
from embodied_harness.model import OpenAICompatibleModel
from embodied_harness.tool_registry import ToolRegistry
from embodied_harness.tool_store import ToolStore


PYTHON = "/data/zxy/envs/vla-report/bin/python"


class EpisodeAdapter:
    def __init__(self, reaches):
        self.reaches = reaches
        self.x = 0.0
        self.closed = False

    @property
    def initial_context(self):
        return {"task_instruction": "move one unit right"}

    def dispatch(self, method, arguments):
        if method == "sense":
            return {"frame_id": "sensor-frame", "x": self.x}
        if method == "verify":
            name = arguments["verifier"]
            return {"verified": name == "scene_visible" or (
                name == "goal_reached" and self.x >= 1.0
            ), "measured_x": self.x}
        if method == "act":
            if self.reaches:
                self.x = float(arguments["action"]["target_x"])
            return {"reached": self.reaches, "measured_x": self.x}
        if method == "record":
            return {"recorded": True}
        raise AssertionError(method)

    def sensor_report(self, execution):
        return {
            "sensor_verification_passed": (
                execution.get("graph_outcome") == "success" and self.x >= 1.0
            ),
            "measured_x": self.x,
        }

    def close(self):
        self.closed = True


class AdapterFactory:
    def __init__(self):
        self.instances = []

    def __call__(self):
        adapter = EpisodeAdapter(reaches=len(self.instances) >= 1)
        self.instances.append(adapter)
        return adapter


def _call(number, name, arguments):
    return {"content": "", "tool_calls": [{
        "id": f"call-{number}", "name": name,
        "arguments": json.dumps(arguments),
    }]}


class ScriptedEngineeringModel:
    """Deterministic model double exercising the real Agent/tool protocol."""
    def __init__(self):
        self.steps = {1: 0, 2: 0}

    def decide(self, *, messages, tools):
        round_index = json.loads(messages[1]["content"])["round"]
        step = self.steps[round_index]
        self.steps[round_index] += 1
        observe = '''def run_stage(adapter, context):
    frame = adapter.sense("rgbd", {})
    return {"outcome": "observed", "updates": {"frame_id": frame["frame_id"], "start_x": frame["x"]}}
'''
        precheck = '''def run_stage(adapter, context):
    proof = adapter.verify("scene_visible", {"frame_id": context["frame_id"]})
    if proof.get("verified") is True:
        return {"outcome": "verified", "updates": {"scene_verified": True}}
    return {"outcome": "rejected", "updates": {"failure_reason": "scene unavailable"}}
'''
        move = '''def run_stage(adapter, context):
    result = adapter.act({"target_x": context["start_x"] + 1.0})
    if result.get("reached") is True:
        return {"outcome": "moved", "updates": {"action_result": result}}
    return {"outcome": "failed", "updates": {"failure_reason": "motion did not reach"}}
'''
        goal = '''def run_stage(adapter, context):
    proof = adapter.verify("goal_reached", {})
    if proof.get("verified") is True:
        return {"outcome": "verified", "updates": {"goal_evidence": proof}}
    return {"outcome": "rejected", "updates": {"failure_reason": "goal absent"}}
'''
        if round_index == 1:
            sequence = [
                ("create_stage_node", {"name": "observe_scene", "kind": "observation",
                 "description": "live sensing", "source": observe,
                 "requires": ["task_instruction"],
                 "provides_by_outcome": {"observed": ["frame_id", "start_x"]},
                 "checkpoint_outcomes": []}),
                ("create_stage_node", {"name": "verify_scene", "kind": "verification",
                 "description": "pre-action sensor checkpoint", "source": precheck,
                 "requires": ["frame_id"],
                 "provides_by_outcome": {"verified": ["scene_verified"],
                                         "rejected": ["failure_reason"]},
                 "checkpoint_outcomes": ["verified"]}),
                ("create_stage_node", {"name": "execute_move", "kind": "motion",
                 "description": "first motion version", "source": move,
                 "requires": ["start_x", "scene_verified"],
                 "provides_by_outcome": {"moved": ["action_result"],
                                         "failed": ["failure_reason"]},
                 "checkpoint_outcomes": []}),
                ("create_stage_node", {"name": "verify_goal", "kind": "verification",
                 "description": "final sensor verification", "source": goal,
                 "requires": ["action_result"],
                 "provides_by_outcome": {"verified": ["goal_evidence"],
                                         "rejected": ["failure_reason"]},
                 "checkpoint_outcomes": ["verified"]}),
                ("create_controller_graph", self._graph_args(version=1)),
                ("execute_controller_graph", {"graph_id": "move_controller:v001"}),
            ]
        else:
            sequence = [
                ("create_stage_node", {"name": "execute_move", "kind": "motion",
                 "description": "revised from sensor failure", "source": move,
                 "requires": ["start_x", "scene_verified"],
                 "provides_by_outcome": {"moved": ["action_result"],
                                         "failed": ["failure_reason"]},
                 "checkpoint_outcomes": []}),
                ("create_controller_graph", self._graph_args(version=2)),
                ("execute_controller_graph", {"graph_id": "move_controller:v002"}),
            ]
        if step >= len(sequence):
            return {"content": "round complete", "tool_calls": []}
        name, args = sequence[step]
        return _call(round_index * 100 + step, name, args)

    @staticmethod
    def _graph_args(version):
        bindings = {
            "observe": "observe_scene:v001", "precheck": "verify_scene:v001",
            "move": f"execute_move:v{version:03d}", "goal": "verify_goal:v001",
        }
        return {
            "name": "move_controller", "description": "sensor closed loop",
            "entry": "observe", "bindings": bindings,
            "edges": [
                {"from": "observe", "outcome": "observed", "to": "precheck"},
                {"from": "precheck", "outcome": "verified", "to": "move"},
                {"from": "precheck", "outcome": "rejected", "to": "$failure"},
                {"from": "move", "outcome": "moved", "to": "goal"},
                {"from": "move", "outcome": "failed", "to": "$failure"},
                {"from": "goal", "outcome": "verified", "to": "$success"},
                {"from": "goal", "outcome": "rejected", "to": "$failure"},
            ],
            "initial_fields": ["task_instruction"], "max_visits": 3,
            "base_graph_id": None if version == 1 else "move_controller:v001",
            "frozen_aliases": [] if version == 1 else ["observe", "precheck"],
        }


def test_agent_revises_failed_node_and_freezes_complete_skill(tmp_path):
    factory = AdapterFactory()
    engine = EvolutionEngine(
        root=tmp_path / "run", model=ScriptedEngineeringModel(),
        adapter_factory=factory, available_initial_fields={"task_instruction"},
        python=PYTHON, max_agent_turns=12,
    )
    state = engine.run(
        task="move one unit right", skill_name="move_right_skill", max_rounds=2,
    )
    assert state["status"] == "sensor_success"
    assert len(state["rounds"]) == 2
    assert state["rounds"][0]["graph_outcome"] == "failure"
    assert state["rounds"][0]["verified_prefix_aliases"] == ["observe", "precheck"]
    assert state["rounds"][1]["graph_outcome"] == "success"
    assert all(adapter.closed for adapter in factory.instances)

    graph = json.loads((tmp_path / "run/graphs/move_controller/v002/manifest.json").read_text())
    assert graph["bindings"]["observe"] == "observe_scene:v001"
    assert graph["bindings"]["precheck"] == "verify_scene:v001"
    assert graph["bindings"]["move"] == "execute_move:v002"
    skill_path = Path(state["skill"]["path"])
    assert (skill_path / "graph/manifest.json").is_file()
    assert len(list((skill_path / "nodes").glob("*/manifest.json"))) == 4

    # Resume is idempotent: no new robot episode or asset version is created.
    resumed = engine.run(
        task="move one unit right", skill_name="move_right_skill", max_rounds=2,
    )
    assert resumed["skill"]["skill_id"] == state["skill"]["skill_id"]
    assert len(factory.instances) == 2


def test_previous_evidence_summary_keeps_geometry_but_drops_camera_payload():
    previous = {
        "round": 3, "graph_id": "controller:v003",
        "graph_outcome": "failure", "failure_signature": "verify/rejected",
        "sensor_report": {
            "sensor_verification_passed": False, "final_step": 91,
            "rollout_path": "/tmp/rollout.mp4",
            "final_proprioception": {"huge": list(range(1000))},
        },
        "node_trace": [{"alias": "verify", "node_id": "verify:v001",
                        "outcome": "rejected", "updates": {"huge": "x" * 10000}}],
        "rpc_evidence": [{
            "method": "sense", "result": {
                "frame_id": "frame-2", "step": 91,
                "cameras": {"agentview": {"rgb_path": "x", "intrinsic": list(range(1000))}},
                "proprioception": {"robot0_eef_pos": [0.1, 0.2, 1.0]},
            },
        }, {
            "method": "verify",
            "arguments": {"verifier": "visual_support_relation", "payload": {
                "frame": {"cameras": {"huge": "x" * 10000}},
                "object_query": "black bowl", "source_ref": "source",
                "target_ref": "target",
            }},
            "result": {"verified": False, "target_xy_error_m": 0.02,
                       "vertical_offset_m": -0.06,
                       "object": {"world_xyz": [0.0, 0.2, 0.91],
                                  "mask_path": "/tmp/mask.png"}},
        }],
    }
    summary = EvolutionEngine._previous_evidence_summary(previous)
    encoded = json.dumps(summary)
    assert len(encoded) < 3000
    assert "cameras" not in encoded
    assert "intrinsic" not in encoded
    assert "final_proprioception" not in encoded
    assert summary["rpc_evidence"][1]["result"]["vertical_offset_m"] == -0.06
    assert summary["rpc_evidence"][1]["result"]["object"]["world_xyz"] == [0.0, 0.2, 0.91]


def test_agent_retries_model_transport_without_replaying_tools(tmp_path):
    class FlakyModel:
        def __init__(self): self.calls = 0
        def decide(self, *, messages, tools):
            self.calls += 1
            if self.calls < 3: raise ConnectionError("temporary disconnect")
            return {"content": "recovered", "tool_calls": []}

    model = FlakyModel()
    result = Agent(
        model=model, tools=ToolRegistry(), system_prompt="test",
        trace_path=tmp_path / "trace.jsonl", max_turns=1,
        max_model_attempts=3, model_retry_delay_seconds=0,
    ).run("continue")
    assert result["completed"] is True
    assert result["final_text"] == "recovered"
    assert model.calls == 3
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    errors = [event for event in events if event["type"] == "model_error"]
    assert [event["will_retry"] for event in errors] == [True, True]


def test_streaming_model_reassembles_fragmented_tool_call():
    class Object:
        def __init__(self, **items): self.__dict__.update(items)

    chunks = [
        Object(choices=[Object(delta=Object(
            content=None, tool_calls=[Object(
                index=0, id="call-1", function=Object(
                    name="create_stage_", arguments='{\"name\":\"ground'),
            )],
        ))]),
        Object(choices=[Object(delta=Object(
            content=None, tool_calls=[Object(
                index=0, id=None, function=Object(
                    name="node", arguments='\",\"kind\":\"sensing\"}'),
            )],
        ))]),
        Object(choices=[Object(delta=Object(content="done", tool_calls=None))]),
    ]
    result = OpenAICompatibleModel._collect_stream(chunks)
    assert result == {
        "content": "done",
        "tool_calls": [{"id": "call-1", "name": "create_stage_node",
                        "arguments": '{"name":"ground","kind":"sensing"}'}],
    }


def test_agent_resumes_transport_failure_without_replaying_completed_tool(tmp_path):
    registry = ToolRegistry()
    calls = []
    registry.register(
        name="remember", description="record once",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        function=lambda: calls.append({}) or {"recorded": True},
    )

    class FirstProcessModel:
        def __init__(self): self.step = 0
        def decide(self, *, messages, tools):
            self.step += 1
            if self.step == 1:
                return _call(1, "remember", {})
            raise ConnectionError("outage")

    trace = tmp_path / "agent_trace.jsonl"
    first = Agent(
        model=FirstProcessModel(), tools=registry, system_prompt="test",
        trace_path=trace, max_turns=4, max_model_attempts=1,
        model_retry_delay_seconds=0,
    ).run("do it")
    assert first["completed"] is False
    assert len(calls) == 1
    # A process may be killed immediately after announcing a resume and before
    # receiving another model response; a later process must still recover.
    with trace.open("a") as stream:
        stream.write(json.dumps({"unix": 0, "type": "resume", "turn": 2,
                                 "recovered_tool_results": 1}) + "\n")

    class ResumedModel:
        def decide(self, *, messages, tools):
            assert any(message.get("role") == "tool" for message in messages)
            return {"content": "continued", "tool_calls": []}

    second = Agent(
        model=ResumedModel(), tools=registry, system_prompt="test",
        trace_path=trace, max_turns=4, max_model_attempts=1,
        model_retry_delay_seconds=0,
    ).run("do it")
    assert second["completed"] is True
    assert second["final_text"] == "continued"
    assert len(calls) == 1
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(event["type"] == "resume" and event["turn"] == 2 for event in events)


def test_evolution_run_directory_rejects_concurrent_writer(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    lock_path = root / ".evolution.lock"
    with lock_path.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = EvolutionEngine(
            root=root, model=ScriptedEngineeringModel(),
            adapter_factory=AdapterFactory(),
            available_initial_fields={"task_instruction"}, python=PYTHON,
        )
        with pytest.raises(RuntimeError, match="another evolution process"):
            engine.run(task="move one unit right", skill_name="skill", max_rounds=1)


def test_tested_tool_is_automatically_installable_in_robot_adapter(tmp_path):
    tools = ToolStore(tmp_path / "tools")
    created = tools.create(
        name="add_one", description="deterministic reusable transform",
        source="def run(payload):\n    return {'value': payload.get('value', 0) + 1}\n",
        input_schema={"type": "object"}, output_schema={"type": "object"},
        source_urls=["https://example.org/algorithm"],
    )
    tools.test(created["tool_id"], [
        {"input": {"value": 2}, "expected": {"value": 3}},
    ])

    class InstallableAdapter:
        def __init__(self): self.capabilities = {}
        def register_capability(self, tool_id, function):
            self.capabilities[tool_id] = function

    adapter = InstallableAdapter()
    assert tools.install_runtime_capabilities(adapter) == ["add_one:v001"]
    assert adapter.capabilities["add_one:v001"]({"value": 9}) == {"value": 10}
