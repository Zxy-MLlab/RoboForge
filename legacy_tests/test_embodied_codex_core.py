from pathlib import Path
import hashlib
import json
import os
import sys

import pytest

from embodied_codex.legacy.runtime import ControllerRuntime
from embodied_codex.examples.evaluate_libero_skill import inspect_skill, _sensor_success
from embodied_codex.capabilities.open_vocab_rgbd import OpenVocabularyRGBD
from embodied_codex.legacy.agent import CodingAgent
from embodied_codex.legacy.assets import (CapabilityLibrary, ExperienceLibrary, SkillLibrary,
                                   CapabilityGapLibrary, AssetError)
from embodied_codex.legacy.engineering import (EngineeringSurface, _controller_semantic_sha256,
                                         _controller_strategy_sha256,
                                         _controller_strategy_prefix_sha256,
                                         _controller_tool_ids,
                                         _controller_tool_ids_before_robot_event,
                                         transient_infrastructure_failure)
from embodied_codex.legacy.evolution import (EvolutionEngine,
                                      _post_action_transient_replay_source)
from embodied_codex.legacy.registry import FunctionRegistry
from embodied_codex.legacy.workspace import TaskWorkspace, WorkspaceError
from embodied_codex.web import _bing_results
from embodied_codex.legacy.conformance import audit_run
from embodied_codex.legacy.task_model import TaskModelError, review_capability_integration


PYTHON = sys.executable


class CapabilityReviewModel:
    def __init__(self, verdict):
        self.verdict=dict(verdict);self.called=False

    def decide(self, *, messages, tools):
        if self.called:return {"content":"review complete","tool_calls":[]}
        self.called=True
        return _call(1,"submit_capability_integration_review",self.verdict)


def test_coding_agent_default_context_budget_stays_below_unstable_api_size(tmp_path):
    agent=CodingAgent(model=object(),registry=FunctionRegistry(),system_prompt="system",
        trace_path=tmp_path/"trace.jsonl")
    assert agent.max_context_characters==70000


def test_agent_global_context_compaction_preserves_tool_call_boundaries(tmp_path):
    import json
    registry=FunctionRegistry()
    agent=CodingAgent(model=object(),registry=registry,system_prompt="system",
        trace_path=tmp_path/"trace.jsonl",max_context_characters=40000,
        context_tail_messages=8)
    messages=[{"role":"system","content":"system"},{"role":"user","content":"task"}]
    for index in range(20):
        messages.extend([
            {"role":"assistant","content":"","tool_calls":[{"id":f"c{index}"}]},
            {"role":"tool","tool_call_id":f"c{index}","content":"x"*5000},
            {"role":"user","content":f"evidence {index}"},
        ])
    compacted=agent._compact_context_window(messages)
    assert len(json.dumps(compacted))<len(json.dumps(messages))
    assert json.loads(compacted[2]["content"])["working_memory_compacted"] is True
    assert compacted[3]["role"] in {"user","assistant"}
    assert not (compacted[3]["role"]=="assistant" and compacted[3].get("tool_calls"))
    assert agent._tool_call_links_valid(compacted)

    uninterrupted=messages[:2]
    for index in range(20):
        uninterrupted.extend([
            {"role":"assistant","content":"","tool_calls":[{"id":f"u{index}"}]},
            {"role":"tool","tool_call_id":f"u{index}","content":"y"*5000},
        ])
    compacted=agent._compact_context_window(uninterrupted)
    assert agent._tool_call_links_valid(compacted)
    # The newest complete Tool chain is the only place where its results can be
    # delivered to the model.  It must survive compaction rather than being
    # replaced before the model has seen it.
    assert [message["role"] for message in compacted][-2:]==["assistant","tool"]
    assert compacted[-1]["tool_call_id"]=="u19"


def test_context_compaction_delivers_latest_multi_tool_chain_before_receipting_it(tmp_path):
    agent=CodingAgent(model=object(),registry=FunctionRegistry(),system_prompt="system",
        trace_path=tmp_path/"trace.jsonl",max_context_characters=40000,
        context_tail_messages=8)
    messages=[{"role":"system","content":"system"},
              {"role":"user","content":"task"+"q"*23000},
              {"role":"assistant","content":"","tool_calls":[
                  {"id":"source"},{"id":"manual"},{"id":"evidence"}]},
              {"role":"tool","tool_call_id":"source","content":json.dumps({
                  "ok":True,"result":{"path":"controller.py","start_line":1,
                  "end_line":180,"total_lines":360,"content":"def run(robot):\n"+"x"*18000}})},
              {"role":"tool","tool_call_id":"manual","content":json.dumps({
                  "ok":True,"result":{"tool_id":"vision:v1","manual":"m"*12000}})},
              {"role":"tool","tool_call_id":"evidence","content":json.dumps({
                  "ok":True,"result":{"path":"failure.json","content":"e"*12000}})}]

    compacted=agent._compact_context_window(messages)

    assert len(json.dumps(compacted,separators=(",",":")))<=40000
    assert agent._tool_call_links_valid(compacted)
    assert any(message.get("role")=="assistant" and message.get("tool_calls")
               for message in compacted)
    delivered_ids={message.get("tool_call_id") for message in compacted
                   if message.get("role")=="tool"}
    assert delivered_ids=={"source","manual","evidence"}
    # Large secondary artifacts may become reloadable receipts, but the latest
    # chain itself is not silently discarded.
    assert any("working_memory_compacted" in message.get("content","")
               for message in compacted if message.get("role")=="tool")


def test_agent_global_context_compaction_is_a_character_hard_limit_for_short_histories(tmp_path):
    import json
    agent=CodingAgent(model=object(),registry=FunctionRegistry(),system_prompt="system",
        trace_path=tmp_path/"trace.jsonl",max_context_characters=40000,
        context_tail_messages=28)
    messages=[{"role":"system","content":"system"},{"role":"user","content":"task"},
        {"role":"assistant","content":"","tool_calls":[{"id":"bulk"}]},
        {"role":"tool","tool_call_id":"bulk","content":"x"*90000},
        {"role":"user","content":"continue from immutable evidence"}]
    compacted=agent._compact_context_window(messages)
    assert len(json.dumps(compacted,separators=(",",":")))<=40000
    assert agent._tool_call_links_valid(compacted)
    assert [message["role"] for message in compacted]==["system","user","user","user"]

    recent_chain=[{"role":"system","content":"system"},{"role":"user","content":"task"},
        {"role":"user","content":"recent evidence follows"},
        {"role":"assistant","content":"","tool_calls":[
            {"id":f"bulk-{index}"} for index in range(12)]}]
    recent_chain.extend({"role":"tool","tool_call_id":f"bulk-{index}",
                         "content":"z"*12000} for index in range(12))
    compacted=agent._compact_context_window(recent_chain)
    assert len(json.dumps(compacted,separators=(",",":")))<=40000
    assert agent._tool_call_links_valid(compacted)


def test_post_robot_context_compaction_preserves_current_evidence_aliases(tmp_path):
    agent=CodingAgent(model=object(),registry=FunctionRegistry(),system_prompt="system",
        trace_path=tmp_path/"trace.jsonl",max_context_characters=40000,
        context_tail_messages=8)
    messages=[{"role":"system","content":"system"},{"role":"user","content":"task"}]
    for index in range(12):
        messages.extend([
            {"role":"assistant","content":"","tool_calls":[{"id":f"c{index}"}]},
            {"role":"tool","tool_call_id":f"c{index}","content":"x"*6000},
            {"role":"user","content":f"continue {index}"},
        ])

    compacted=agent._compact_context_window(
        messages,latest_execution_available=True)
    receipt=json.loads(compacted[2]["content"])

    assert receipt["robot_executed_in_current_pass"] is True
    assert receipt["authoritative_current_execution"]=="latest_robot_execution"
    assert receipt["authoritative_current_rollout"]=="latest_rollout"
    assert "prior episode only" in receipt["previous_alias_scope"]
    assert agent._tool_call_links_valid(compacted)


def test_evolution_workspace_index_is_bounded_recent_and_content_free(tmp_path):
    workspace=TaskWorkspace(tmp_path/"workspace")
    workspace.write_file("controller.py","def run(robot):\n    return {}\n")
    workspace.write_file("old_note.md","old private diagnosis")
    workspace.write_file("latest_handoff.md","latest private diagnosis")
    os.utime(workspace.root/"controller.py",(10,10))
    os.utime(workspace.root/"old_note.md",(20,20))
    os.utime(workspace.root/"latest_handoff.md",(30,30))
    engine=EvolutionEngine.__new__(EvolutionEngine)
    engine.workspace=workspace

    index=engine._workspace_index(limit=2)

    assert [item["path"] for item in index]==["latest_handoff.md","old_note.md"]
    assert all(set(item)=={"path","bytes","modified_unix"} for item in index)
    assert "diagnosis" not in json.dumps(index)


def test_workspace_temporal_warning_marks_preexecution_postrun_note(tmp_path):
    engine=EvolutionEngine.__new__(EvolutionEngine)
    engine.root=tmp_path/"run"
    engine.workspace=TaskWorkspace(engine.root/"workspace")
    engine.workspace.write_file("iteration_008_postrun_diagnosis.md","premature conclusion")
    note=engine.workspace.root/"iteration_008_postrun_diagnosis.md"
    os.utime(note,(10,10))
    execution=engine.root/"iterations"/"iteration_008"/"robot_execution.json"
    execution.parent.mkdir(parents=True);execution.write_text("{}\n")
    os.utime(execution,(20,20))
    engine.workspace.write_file("iteration_007_postrun_diagnosis.md","valid conclusion")
    valid=engine.workspace.root/"iteration_007_postrun_diagnosis.md"
    valid_execution=engine.root/"iterations"/"iteration_007"/"robot_execution.json"
    valid_execution.parent.mkdir(parents=True);valid_execution.write_text("{}\n")
    os.utime(valid_execution,(20,20));os.utime(valid,(30,30))

    warnings=engine._workspace_temporal_warnings()

    assert [item["path"] for item in warnings]==["iteration_008_postrun_diagnosis.md"]
    assert "before_execution_commit" in warnings[0]["warning"]


class FakeDeployment:
    instruction = "move the cube"
    def __init__(self): self.x = 0; self.closed = False
    def dispatch(self, method, arguments):
        if method == "observe": return {"frame_id": "f1", "x": self.x}
        if method == "use":
            return {"tool_id": arguments["tool_id"], "result": {"target_x": 1}}
        if method == "act":
            self.x = arguments["action"]["target_x"]; return {"reached": True, "x": self.x}
        if method == "verify": return {"verified": self.x == 1, "measured_x": self.x}
        if method == "record": return {"recorded": True}
        raise AssertionError(method)
    def sensor_report(self, execution): return {"x": self.x}
    def project_rpc_output(self,method,arguments,result):return dict(result)
    def close(self): self.closed = True


def test_arbitrary_controller_gets_direct_tool_result(tmp_path):
    program = tmp_path / "controller.py"
    program.write_text('''def run(robot):
    frame = robot.observe("rgbd", {})
    target = robot.use("target_finder:v001", {"frame": frame})
    action = robot.act({"target_x": target["target_x"]})
    proof = robot.verify("goal", {"frame": robot.observe("rgbd", {})})
    return {"action": action, "proof": proof}
''')
    deployment = FakeDeployment()
    result = ControllerRuntime(python=PYTHON, timeout_seconds=5).execute(program, deployment)
    assert result["completed"] is True
    assert result["result"]["proof"]["verified"] is True
    assert result["sensor_verification_observed"] is True
    assert [event["method"] for event in result["rpc_events"]] == [
        "observe", "use", "act", "observe", "verify",
    ]
    assert result["rpc_output_defense"]=="kernel-evaluator-field-deny-v1"


def test_kernel_rejects_evaluator_field_even_when_adapter_projector_leaks_it(tmp_path):
    class LeakyDeployment(FakeDeployment):
        def dispatch(self,method,arguments):
            result=super().dispatch(method,arguments)
            if method=="observe":result["nested"]={"reward":1.0}
            return result
    program=tmp_path/"controller.py"
    program.write_text("def run(robot): return robot.observe('rgbd', {})\n")
    result=ControllerRuntime(python=PYTHON,timeout_seconds=5).execute(program,LeakyDeployment())
    assert result["completed"] is False
    assert "forbidden evaluator field" in result["rpc_events"][0]["error"]


def test_sensor_success_does_not_require_redundant_magic_controller_status(tmp_path):
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
''')
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(run/"capabilities",workspace.root,python=PYTHON),
        runtime=ControllerRuntime(python=PYTHON,timeout_seconds=5),
        deployment_factory=ClosedLoopDeployment,
        artifact_dir=run/"iterations"/"iteration_001")
    result=surface.run_robot_controller("controller.py")
    assert result["controller_result"]=={"verified":True,"measured_x":1}
    assert result["sensor_success_candidate"] is True
    execution={"completed":True,"result":{"verified":True},
        "rpc_events":[{"method":"verify","result":{"verified":True}}]}
    assert _sensor_success(execution,{"sensor_verification_passed":True}) is True


def test_failed_controller_ast_cannot_consume_an_unchanged_robot_episode(tmp_path):
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    source='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
    workspace.write_file("controller.py",source)
    rejected=_controller_semantic_sha256(workspace.root/"controller.py")
    deployments=[]
    def deployment_factory():
        deployments.append(True);return ClosedLoopDeployment()
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(run/"capabilities",workspace.root,python=PYTHON),
        runtime=ControllerRuntime(python=PYTHON,timeout_seconds=5),
        deployment_factory=deployment_factory,
        artifact_dir=run/"iterations"/"iteration_002",
        rejected_controller_semantic_sha256=rejected)
    with pytest.raises(RuntimeError,match="unchanged_controller_after_failed_episode"):
        surface.run_robot_controller("controller.py")
    workspace.write_file("controller.py",source+"# formatting-only change\n")
    with pytest.raises(RuntimeError,match="unchanged_controller_after_failed_episode"):
        surface.run_robot_controller("controller.py")
    workspace.write_file("controller.py",source.replace(
        'robot.act({"target_x": 1})','robot.record({"strategy": "changed"})\n    robot.act({"target_x": 1})'))
    with pytest.raises(RuntimeError,match="unchanged_controller_after_failed_episode"):
        surface.run_robot_controller("controller.py")
    workspace.write_file("controller.py",source.replace(
        'robot.act({"target_x": 1})',
        'unused_release_height = 0.044\n    robot.record({"strategy": "changed"})\n    robot.act({"target_x": 1})'))
    with pytest.raises(RuntimeError,match="unchanged_controller_after_failed_episode"):
        surface.run_robot_controller("controller.py")
    workspace.write_file("controller.py",source.replace(
        'robot.act({"target_x": 1})','robot.act({"target_x": 1.0})'))
    result=surface.run_robot_controller("controller.py")
    assert result["sensor_success_candidate"] is True
    assert deployments==[True]


def test_strategy_fingerprint_ignores_parameter_tuning_but_tracks_new_control_strategy(tmp_path):
    first=tmp_path/"first.py";second=tmp_path/"second.py";retry=tmp_path/"retry.py"
    override=tmp_path/"override.py"
    first.write_text('''def run(robot):
    frame = robot.observe(channel="rgbd", request={})
    target = robot.use("detector:v001", {"frame": frame})
    robot.act({"type": "move_to_point", "target_ref": target, "offset": [0, 0, 0.02], "max_steps": 80})
    return robot.verify("visual_attachment", {"frame": frame})
''')
    second.write_text(first.read_text().replace("0.02", "0.00").replace("80", "150")+
                      "# formatting and tuning only\n")
    retry.write_text('''def run(robot):
    frame = robot.observe(channel="rgbd", request={})
    target = robot.use("detector:v001", {"frame": frame})
    for candidate in range(3):
        robot.act({"type": "move_to_point", "target_ref": target})
        result = robot.verify("visual_attachment", {"frame": frame})
        if result.get("verified"):
            return result
    return result
''')
    override.write_text(first.read_text().replace(
        '"offset": [0, 0, 0.02]',
        '"offset": [0, 0, 0.02], "quaternion_xyzw": live_quaternion'))
    assert _controller_strategy_sha256(first)==_controller_strategy_sha256(second)
    assert _controller_strategy_sha256(first)!=_controller_strategy_sha256(retry)
    assert _controller_strategy_sha256(first)!=_controller_strategy_sha256(override)


def test_failure_prefix_tracks_explicit_orientation_control_mode(tmp_path):
    inherited=tmp_path/"inherited.py";overridden=tmp_path/"overridden.py"
    inherited.write_text('''def run(robot):
    robot.observe(channel="rgbd", request={})
    robot.act({"type": "move_to_pose", "pose_ref": pose,
               "offset": [0, 0, 0.10], "max_steps": 150})
''')
    overridden.write_text('''def run(robot):
    robot.observe(channel="rgbd", request={})
    robot.act({"type": "move_to_pose", "pose_ref": pose,
               "quaternion_xyzw": live_quaternion,
               "offset": [0, 0, 0.10], "max_steps": 150})
''')
    assert (_controller_strategy_prefix_sha256(inherited,2)!=
            _controller_strategy_prefix_sha256(overridden,2))


def test_failure_prefix_ignores_unreached_downstream_changes_but_tracks_retry(tmp_path):
    failed=tmp_path/"failed.py";downstream=tmp_path/"downstream.py";retry=tmp_path/"retry.py"
    failed.write_text('''def run(robot):
    frame = robot.observe(channel="rgbd", request={})
    robot.use("detector:v001", {"frame": frame})
    robot.act({"type": "move_to_point"})
    robot.act({"type": "gripper", "command": "close"})
    return robot.verify("visual_attachment", {})
''')
    downstream.write_text(failed.read_text().replace(
        'return robot.verify("visual_attachment", {})',
        'result = robot.verify("visual_attachment", {})\n'
        '    for target in range(3):\n'
        '        robot.use("downstream_target_tool:v001", {"target": target})\n'
        '        robot.act({"type": "move_to_point"})\n'
        '    return result'))
    retry.write_text(failed.read_text().replace(
        'robot.act({"type": "move_to_point"})',
        'for candidate in range(3):\n'
        '        robot.act({"type": "move_to_point"})'))
    count=4
    assert (_controller_strategy_prefix_sha256(failed,count)==
            _controller_strategy_prefix_sha256(downstream,count))
    assert (_controller_strategy_prefix_sha256(failed,count)!=
            _controller_strategy_prefix_sha256(retry,count))
    assert _controller_tool_ids_before_robot_event(downstream,count)=={"detector:v001"}


def test_controller_tool_ids_resolve_only_unambiguous_string_bindings(tmp_path):
    controller=tmp_path/"controller.py"
    controller.write_text('''DETECTOR = "detector:v001"
def run(robot):
    return robot.use(DETECTOR, {})
''')
    assert _controller_tool_ids(controller)=={"detector:v001"}
    assert _controller_tool_ids_before_robot_event(controller,1)=={"detector:v001"}

    controller.write_text('''DETECTOR = "detector:v001"
DETECTOR = "detector:v002"
def run(robot):
    return robot.use(DETECTOR, {})
''')
    assert _controller_tool_ids(controller)==set()
    assert _controller_tool_ids_before_robot_event(controller,1)==set()


def test_controller_tool_ids_do_not_resolve_dynamic_bindings(tmp_path):
    controller=tmp_path/"controller.py"
    controller.write_text('''VERSION = "v001"
DETECTOR = "detector:" + VERSION
def run(robot):
    return robot.use(DETECTOR, {})
''')
    assert _controller_tool_ids(controller)==set()
    assert _controller_tool_ids_before_robot_event(controller,1)==set()


def test_repeated_failed_strategy_is_rejected_before_robot_start(tmp_path):
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    source='''def run(robot):
    robot.act({"type": "move_stage", "heading_deg": 1.0})
    return robot.verify("goal", {})
'''
    workspace.write_file("controller.py",source)
    strategy=_controller_strategy_sha256(workspace.root/"controller.py")
    deployments=[]
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(run/"capabilities",workspace.root,python=PYTHON),
        runtime=ControllerRuntime(python=PYTHON,timeout_seconds=5),
        deployment_factory=lambda:deployments.append(True) or ClosedLoopDeployment(),
        artifact_dir=run/"iterations"/"iteration_003",
        rejected_controller_strategy_failures={strategy:{
            "failures":["attachment failed (2x)"],"prior_tool_ids":[]}})
    with pytest.raises(RuntimeError,match="repeated_strategy_after_failed_episodes"):
        surface.run_robot_controller("controller.py")
    assert deployments==[]

    # Changing an unreachable downstream stage, including adding a Tool there,
    # cannot disguise the already-failed physical prefix as a new strategy.
    prefix=_controller_strategy_prefix_sha256(workspace.root/"controller.py",2)
    surface.rejected_controller_strategy_failures={prefix:{
        "strategy_prefix_sha256":prefix,"robot_event_count":2,
        "failures":["attachment failed (2x)"],"prior_tool_ids":[]}}
    workspace.write_file("controller.py",'''def run(robot):
    robot.act({"type": "move_stage", "heading_deg": 2.0})
    result = robot.verify("goal", {})
    robot.use("unreached_target_selector:v001", {"candidates": []})
    for target in range(3):
        robot.act({"type": "move_stage", "heading_deg": target})
    return result
''')
    with pytest.raises(RuntimeError,match="repeated_strategy_after_failed_episodes"):
        surface.run_robot_controller("controller.py")
    assert deployments==[]

    # A newly integrated capability gets one causal task-level trial even when
    # the surrounding action family is unchanged.
    workspace.write_file("controller.py",'''def run(robot):
    result = robot.use("new_grasp_ranker:v001", {"candidates": []})
    robot.act({"type": "move_stage", "heading_deg": result.get("heading_deg", 1.0)})
    return robot.verify("goal", {})
''')
    strategy_with_tool=_controller_strategy_sha256(workspace.root/"controller.py")
    surface.rejected_controller_strategy_failures={strategy_with_tool:{
        "failures":["attachment failed (2x)"],"prior_tool_ids":[]}}
    # The fake deployment does not implement this Tool, but reaching the
    # deployment proves the repeated-strategy gate allowed the integration
    # trial before the controller failed closed.
    result=surface.run_robot_controller("controller.py")
    assert result["completed"] is False
    assert deployments==[True]


def test_controller_supports_helpers_but_cannot_read_host_files(tmp_path):
    import os
    os.environ["APEX_API_KEY"]="must-not-enter-controller"
    secret=tmp_path.parent/"sealed_benchmark_secret.txt";secret.write_text("hidden")
    (tmp_path/"helper.py").write_text("VALUE = 1\n")
    program=tmp_path/"controller.py"
    program.write_text(f'''from pathlib import Path
import os
from helper import VALUE
def run(robot):
    proof=robot.verify("goal", {{}})
    return {{"helper":VALUE,"host_secret_visible":Path({str(secret)!r}).exists(),
            "host_api_key":os.environ.get("APEX_API_KEY"),"proof":proof}}
''')
    deployment=FakeDeployment();deployment.x=1
    result=ControllerRuntime(python=PYTHON,timeout_seconds=5).execute(program,deployment)
    assert result["completed"] is True
    assert result["result"]["helper"]==1
    assert result["result"]["host_secret_visible"] is False
    assert result["result"]["host_api_key"] is None


def test_workspace_supports_free_code_edits_and_commands(tmp_path):
    import os
    workspace = TaskWorkspace(tmp_path / "task")
    assert workspace.read_file("not_created.py")=={
        "path":"not_created.py","exists":False,"start_line":1,"end_line":0,
        "total_lines":0,"content":""}
    workspace.write_file("controller.py", "print('old')\n")
    workspace.replace_in_file("controller.py", "old", "new")
    assert "new" in workspace.read_file("controller.py")["content"]
    assert workspace.read_file("controller.py")["exists"] is True
    result = workspace.run_command([PYTHON, "controller.py"], timeout_seconds=5)
    assert result["exit_code"] == 0 and "new" in result["output"]
    secret=tmp_path/"benchmark_secret.txt";secret.write_text("privileged")
    isolated=workspace.run_command([PYTHON,"-c",
        f"from pathlib import Path; print(Path({str(secret)!r}).exists())"],timeout_seconds=5)
    assert isolated["output"].strip()=="False"
    assert isolated["sandbox"]=="bubblewrap-workspace-v1"
    os.environ["APEX_API_KEY"]="must-not-enter-agent-code"
    secret_env=workspace.run_command([PYTHON,"-c",
        "import os; print(os.environ.get('APEX_API_KEY', 'hidden'))"],timeout_seconds=5)
    assert secret_env["output"].strip()=="hidden"
    network=workspace.run_command([PYTHON,"-c",
        "import socket; s=socket.socket(); s.settimeout(.2); print(s.connect_ex(('1.1.1.1',80)))"],
        timeout_seconds=5)
    assert network["exit_code"]==0 and network["output"].strip()!="0"
    with pytest.raises(WorkspaceError): workspace.write_file("../escape.py", "bad")


def test_line_range_edit_is_bounded_atomic_and_semantically_audited(tmp_path):
    workspace=TaskWorkspace(tmp_path/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    value = 1
    return value
''')
    old="    value = 1\n"
    digest=hashlib.sha256(old.encode()).hexdigest()
    edited=workspace.replace_file_lines("controller.py",2,2,
        "    value = 2",expected_old_sha256=digest)
    assert edited["old_sha256"]==digest
    assert workspace.read_file("controller.py")["content"]==(
        "def run(robot):\n    value = 2\n    return value")
    with pytest.raises(WorkspaceError,match="changed since inspection"):
        workspace.replace_file_lines("controller.py",2,2,"    value = 3",
                                     expected_old_sha256=digest)
    with pytest.raises(WorkspaceError,match="invalid inclusive line range"):
        workspace.replace_file_lines("controller.py",0,1,"bad")

    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(tmp_path/"tools",workspace.root),
        runtime=ControllerRuntime(python=PYTHON),
        deployment_factory=FakeDeployment,artifact_dir=tmp_path/"artifact")
    receipt=surface.registry().invoke("replace_file_lines",{
        "path":"controller.py","start_line":2,"end_line":2,
        "new_content":"    value = 4"})
    assert receipt["controller_semantic_progress"] is True
    assert "replace_file_lines" in surface.registry().items


def test_workspace_source_read_is_real_200_line_pagination(tmp_path):
    workspace=TaskWorkspace(tmp_path/"workspace")
    workspace.write_file("controller.py","\n".join(f"line_{i}" for i in range(1,506)))
    first=workspace.read_file("controller.py",1,500)
    assert first["start_line"]==1 and first["end_line"]==200
    assert first["next_start_line"]==201 and first["content_truncated"] is True
    second=workspace.read_file("controller.py",first["next_start_line"],500)
    assert second["start_line"]==201 and second["end_line"]==400
    assert second["next_start_line"]==401


def _call(number, name, arguments):
    import json
    return {"content":"","tool_calls":[{"id":f"c{number}","name":name,
            "arguments":json.dumps(arguments)}]}


def test_engineering_function_registry_enforces_its_published_schema():
    registry=FunctionRegistry();registry.add("strict","strict contract",{
        "type":"object","properties":{"value":{"type":"number"}},
        "required":["value"],"additionalProperties":False},lambda value:value)
    assert registry.invoke("strict",{"value":2})==2
    with pytest.raises(ValueError,match="required property"):
        registry.invoke("strict",{})
    with pytest.raises(ValueError,match="Additional properties"):
        registry.invoke("strict",{"value":2,"typo":1})


def _capability_review_inputs():
    return {
        "gap":{"gap_id":"adaptive_grasp:v001",
               "required_capability":{"kind":"sensor-verified adaptive grasp"}},
        "tools":[{"manifest":{"tool_id":"adaptive_grasp_tool:v001",
                               "status":"tested"},
                  "manual":{"purpose":"Generate a sensor-anchored grasp."}}],
        "controller_source":(
            "def run(robot):\n"
            "    return robot.use('adaptive_grasp_tool:v001', {})\n"),
    }


def test_capability_critic_allows_nonblocking_limitations(tmp_path):
    payload=_capability_review_inputs()
    verdict=review_capability_integration(
        model=CapabilityReviewModel({
            "approved":True,
            "approved_tool_ids":["adaptive_grasp_tool:v001"],
            "covered_requirements":["sensor-verified adaptive grasp"],
            "blocking_issues":[],
            "limitations":["Physical efficacy still requires a robot rollout"],
        }),trace_path=tmp_path/"critic.jsonl",**payload)
    assert verdict["approved"] is True
    assert verdict["issues"]==[]
    assert verdict["limitations"]==[
        "Physical efficacy still requires a robot rollout"]


def test_capability_critic_rejects_approval_with_blocking_issues(tmp_path):
    payload=_capability_review_inputs()
    with pytest.raises(TaskModelError,match="internally inconsistent approval"):
        review_capability_integration(
            model=CapabilityReviewModel({
                "approved":True,
                "approved_tool_ids":["adaptive_grasp_tool:v001"],
                "covered_requirements":["sensor-verified adaptive grasp"],
                "blocking_issues":["Controller never invokes the Tool"],
                "limitations":[],
            }),trace_path=tmp_path/"critic.jsonl",**payload)


def test_capability_critic_rejects_unknown_tool_id(tmp_path):
    payload=_capability_review_inputs()
    with pytest.raises(TaskModelError,match="approved an unknown Tool"):
        review_capability_integration(
            model=CapabilityReviewModel({
                "approved":True,
                "approved_tool_ids":["unregistered_tool:v999"],
                "covered_requirements":["sensor-verified adaptive grasp"],
                "blocking_issues":[],
                "limitations":[],
            }),trace_path=tmp_path/"critic.jsonl",**payload)


def test_engineering_tool_schemas_do_not_require_python_default_arguments(tmp_path):
    import inspect
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(run/"tools",workspace.root,python=PYTHON),
        runtime=ControllerRuntime(python=PYTHON),deployment_factory=lambda:FakeDeployment(),
        artifact_dir=run/"iterations"/"iteration_001")
    for function in surface.registry().items.values():
        signature=inspect.signature(function.call)
        required=set(function.parameters.get("required") or [])
        properties=set((function.parameters.get("properties") or {}))
        for name in properties.intersection(signature.parameters):
            parameter=signature.parameters[name]
            if parameter.default is not inspect.Parameter.empty:
                assert name not in required, (function.name,name)


class ScriptedCodingModel:
    def __init__(self): self.steps={1:0,2:0}
    def decide(self, *, messages, tools):
        import json
        instruction=json.loads(messages[1]["content"])
        assert "retrieved_tool_index" in instruction
        iteration=instruction["iteration"]
        step=self.steps[iteration];self.steps[iteration]+=1
        bad='''def run(robot):
    robot.observe("rgbd", {})
    robot.act({"target_x": 0})
    proof = robot.verify("goal", {})
    return {"status": "sensor_success" if proof["verified"] else "failure", "proof": proof}
'''
        good=bad.replace('"target_x": 0','"target_x": 1')
        sequence=[
            ("write_file",{"path":"controller.py","content":bad if iteration==1 else good}),
            ("run_command",{"argv":[PYTHON,"-m","py_compile","controller.py"],"timeout_seconds":5}),
            ("run_robot_controller",{"path":"controller.py"}),
        ]
        if step>=len(sequence):return {"content":"iteration complete","tool_calls":[]}
        name,args=sequence[step];return _call(iteration*10+step,name,args)


class ClosedLoopDeployment(FakeDeployment):
    def sensor_report(self, execution):
        return {"sensor_verification_passed": self.x == 1, "x": self.x,
                "benchmark_signal_exposed": False}


def test_embodied_codex_freely_rewrites_controller_and_freezes_skill(tmp_path):
    instances=[]
    def factory():
        value=ClosedLoopDeployment();instances.append(value);return value
    engine=EvolutionEngine(root=tmp_path/"run",model=ScriptedCodingModel(),
        deployment_factory=factory,python=PYTHON,
        deployment_guidance={"actions":"target_x for fake test"})
    state=engine.run(task="move the cube",skill_name="move_cube_skill",max_iterations=2)
    assert state["status"]=="sensor_success"
    assert len(state["iterations"])==2
    assert state["iterations"][0]["evidence"]["sensor_success_candidate"] is False
    assert state["iterations"][1]["evidence"]["sensor_success_candidate"] is True
    for index, row in enumerate(state["iterations"], 1):
        snapshot=tmp_path/"run"/"iterations"/f"iteration_{index:03d}"/"controller.py"
        assert snapshot.is_file()
        assert row["evidence"]["controller_snapshot"]==str(snapshot.resolve())
    skill=Path(state["skill"]["path"])
    assert (skill/"controller.py").is_file() and (skill/"manifest.json").is_file()
    assert (skill/"experience.json").is_file()
    controller, manifest, tools = inspect_skill(skill)
    assert controller == (skill/"controller.py").resolve()
    assert manifest["skill_id"] == "move_cube_skill:v001"
    assert set(manifest["interface"]["required_robot_operations"])=={
        "observe","act","verify"}
    assert manifest["interface"]["required_sensors"]==["rgbd"]
    assert manifest["interface"]["composition_notes"]
    assert tools == {}
    assert all(instance.closed for instance in instances)
    successor=EvolutionEngine(root=tmp_path/"successor",model=object(),
        deployment_factory=lambda:None,python=PYTHON)
    bootstrap=successor.bootstrap_skill(skill)
    assert bootstrap["skill_id"]=="move_cube_skill:v001"
    assert Path(bootstrap["experience_path"]).is_file()
    assert (tmp_path/"successor"/"workspace"/"controller.py").read_bytes()==controller.read_bytes()


def test_success_freezes_executed_snapshot_and_skips_post_success_edit(tmp_path):
    class EditAfterSuccessModel:
        def __init__(self):self.step=0
        def decide(self,*,messages,tools):
            executed='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
            unexecuted='''def run(robot):
    return {"sensor_failure": "unexecuted edit"}
'''
            sequence=[("write_file",{"path":"controller.py","content":executed}),
                      ("run_robot_controller",{"path":"controller.py"}),
                      ("write_file",{"path":"controller.py","content":unexecuted})]
            if self.step>=len(sequence):return {"content":"done","tool_calls":[]}
            name,args=sequence[self.step];self.step+=1
            return _call(self.step,name,args)

    root=tmp_path/"snapshot_freeze"
    state=EvolutionEngine(root=root,model=EditAfterSuccessModel(),
        deployment_factory=ClosedLoopDeployment,python=PYTHON).run(
            task="move the cube",skill_name="snapshot_skill",max_iterations=1)
    frozen=Path(state["skill"]["path"])/"controller.py"
    snapshot=root/"iterations"/"iteration_001"/"controller.py"
    assert frozen.read_bytes()==snapshot.read_bytes()
    assert frozen.read_bytes()==(root/"workspace"/"controller.py").read_bytes()
    assert state["iterations"][0]["agent_completed"] is True


def test_robot_execution_is_committed_before_post_rollout_model_crash(tmp_path):
    import json
    class CrashAfterRobotModel:
        def __init__(self):self.step=0
        def decide(self,*,messages,tools):
            program='''def run(robot):
    robot.act({"target_x": 0})
    return {"status": "sensor_failure"}
'''
            sequence=[
                ("write_file",{"path":"controller.py","content":program}),
                ("run_robot_controller",{"path":"controller.py"}),
            ]
            if self.step>=len(sequence):raise KeyboardInterrupt("simulated model stream loss")
            name,args=sequence[self.step];self.step+=1
            return _call(self.step,name,args)

    root=tmp_path/"transactional"
    engine=EvolutionEngine(root=root,model=CrashAfterRobotModel(),
        deployment_factory=ClosedLoopDeployment,python=PYTHON)
    with pytest.raises(KeyboardInterrupt,match="stream loss"):
        engine.run(task="move the cube",skill_name="transactional_skill",max_iterations=1)
    state=json.loads((root/"state.json").read_text())
    assert len(state["iterations"])==1
    row=state["iterations"][0]
    assert row["robot_execution_transaction_committed"] is True
    assert row["agent_error"]=="post_execution_agent_pending"
    assert row["evidence"]["completed"] is True
    assert (root/"iterations"/"iteration_001"/"robot_execution.json").is_file()


def test_committed_success_stops_without_repeating_robot_episode(tmp_path):
    class CrashAfterSuccessfulRobotModel:
        def __init__(self):self.step=0
        def decide(self,*,messages,tools):
            executed='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
            unexecuted='''def run(robot):
    return {"sensor_failure": "unexecuted post-rollout edit"}
'''
            sequence=[
                ("write_file",{"path":"controller.py","content":executed}),
                ("run_robot_controller",{"path":"controller.py"}),
                ("write_file",{"path":"controller.py","content":unexecuted}),
            ]
            if self.step>=len(sequence):
                raise KeyboardInterrupt("simulated post-success stream loss")
            name,args=sequence[self.step];self.step+=1
            return _call(self.step,name,args)

    root=tmp_path/"recover-success";deployments=[]
    def factory():
        deployment=ClosedLoopDeployment();deployments.append(deployment);return deployment
    first=EvolutionEngine(root=root,model=CrashAfterSuccessfulRobotModel(),
        deployment_factory=factory,python=PYTHON)
    recovered=first.run(task="move the cube",skill_name="recover_success",max_iterations=1)
    executed_snapshot=root/"iterations"/"iteration_001"/"controller.py"
    assert recovered["status"]=="sensor_success"
    assert len(deployments)==1
    assert len(recovered["iterations"])==1
    assert recovered["iterations"][0]["agent_completed"] is True
    frozen=Path(recovered["skill"]["path"])/"controller.py"
    assert frozen.read_bytes()==executed_snapshot.read_bytes()
    assert (root/"workspace"/"controller.py").read_bytes()==executed_snapshot.read_bytes()


def test_committed_generalization_success_resumes_with_locked_next_case(tmp_path):
    program='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
    class CrashAfterFirstCaseModel:
        def __init__(self):self.step=0
        def decide(self,*,messages,tools):
            unexecuted='''def run(robot):
    return {"sensor_failure": "must never run"}
'''
            sequence=[
                ("write_file",{"path":"controller.py","content":program}),
                ("run_robot_controller",{"path":"controller.py"}),
                ("write_file",{"path":"controller.py","content":unexecuted}),
            ]
            if self.step>=len(sequence):raise KeyboardInterrupt("case-a stream loss")
            name,args=sequence[self.step];self.step+=1
            return _call(self.step,name,args)
    class CaseDeployment(ClosedLoopDeployment):
        def __init__(self,case):super().__init__();self.case=case
        def sensor_report(self,execution):
            return {**super().sensor_report(execution),"_harness_case_id":self.case}

    root=tmp_path/"recover-gate";created=[]
    def factory_a():
        value=CaseDeployment("case_a");created.append(value);return value
    first=EvolutionEngine(root=root,model=CrashAfterFirstCaseModel(),
        deployment_factory=factory_a,python=PYTHON,
        required_success_cases=["case_a","case_b"]).run(
            task="move the cube",skill_name="recover_gate",max_iterations=1)
    assert first["status"]=="evolving"
    assert len(created)==1

    def factory_b():
        value=CaseDeployment("case_b");created.append(value);return value
    state=EvolutionEngine(root=root,model=object(),deployment_factory=factory_b,
        python=PYTHON,required_success_cases=["case_a","case_b"]).run(
            task="move the cube",skill_name="recover_gate",max_iterations=2)
    assert state["status"]=="sensor_success"
    assert len(created)==2
    assert len(state["iterations"])==2
    assert state["iterations"][1]["locked_generalization_validation"] is True
    coverage=state["generalization_gate"]["successes_by_program"]
    assert len(coverage)==1
    assert list(coverage.values())==[["case_a","case_b"]]
    first_snapshot=root/"iterations"/"iteration_001"/"controller.py"
    second_snapshot=root/"iterations"/"iteration_002"/"controller.py"
    frozen=Path(state["skill"]["path"])/"controller.py"
    assert first_snapshot.read_bytes()==second_snapshot.read_bytes()==frozen.read_bytes()
    assert b"must never run" not in frozen.read_bytes()


def test_generalization_gate_requires_same_controller_on_every_case(tmp_path):
    program='''def run(robot):
    robot.act({"target_x": 1})
    proof = robot.verify("goal", {})
    return {"status": "sensor_success" if proof["verified"] else "failure"}
'''
    class GateModel:
        def __init__(self):self.steps={}
        def decide(self,*,messages,tools):
            import json
            instruction=json.loads(messages[1]["content"])
            assert "required_cases" not in instruction["generalization_gate"]
            assert instruction["generalization_gate"]["required_case_count"]==2
            iteration=instruction["iteration"]
            step=self.steps.get(iteration,0);self.steps[iteration]=step+1
            sequence=[]
            if iteration==1:sequence.append(("write_file",{"path":"controller.py","content":program}))
            sequence.append(("run_robot_controller",{"path":"controller.py"}))
            if step>=len(sequence):return {"content":"case complete","tool_calls":[]}
            name,args=sequence[step];return _call(iteration*10+step,name,args)
    class CaseDeployment(ClosedLoopDeployment):
        def __init__(self,case):super().__init__();self.case=case
        def sensor_report(self,execution):
            return {**super().sensor_report(execution),"_harness_case_id":self.case}
    cases=iter(["case_a","case_b"])
    gate_model=GateModel()
    engine=EvolutionEngine(root=tmp_path/"gated",model=gate_model,
        deployment_factory=lambda:CaseDeployment(next(cases)),python=PYTHON,
        required_success_cases=["case_a","case_b"])
    state=engine.run(task="move the cube",skill_name="gated_skill",max_iterations=2)
    assert state["status"]=="sensor_success"
    hashes={row["evidence"]["sensor_report"]["benchmark_signal_exposed"]
            for row in state["iterations"]}
    assert hashes=={False}
    coverage=state["generalization_gate"]["successes_by_program"]
    assert list(coverage.values())==[["case_a","case_b"]]
    assert all("_harness_case_id" not in row["evidence"]["sensor_report"]
               for row in state["iterations"])
    assert state["iterations"][1]["locked_generalization_validation"] is True
    assert state["iterations"][1]["coding_passes"]==0
    assert 2 not in gate_model.steps


def test_generalization_coverage_is_invalidated_when_evidence_protocol_changes(tmp_path):
    import json
    from embodied_codex.legacy.evolution import EvolutionEngine
    root=tmp_path/"protocol-change";root.mkdir()
    (root/"state.json").write_text(json.dumps({
        "task":"task","skill_name":"skill","status":"evolving","iterations":[],
        "generalization_gate":{
            "required_cases":["a","b"],
            "successes_by_program":{"old-sha":["a","b"]},
            "evidence_protocol":"old-verifier"}}))
    engine=EvolutionEngine(root=root,model=object(),deployment_factory=lambda:None,
        required_success_cases=["a","b"],success_evidence_protocol="new-verifier")
    # Exercise the protocol migration before the loop needs a deployment.
    engine._run("task","skill",0)
    state=json.loads((root/"state.json").read_text())
    assert state["generalization_gate"]["successes_by_program"]=={}
    assert state["generalization_gate"]["evidence_protocol"]=="new-verifier"
    assert state["invalidated_generalization_gates"][0]["gate"][
        "successes_by_program"]=={"old-sha":["a","b"]}


def test_infrastructure_retry_replays_current_controller_without_model(tmp_path):
    run_root=tmp_path/"retry"
    first=EvolutionEngine(root=run_root,model=ScriptedCodingModel(),
        deployment_factory=ClosedLoopDeployment,python=PYTHON)
    failed=first.run(task="move the cube",skill_name="retry_skill",max_iterations=1)
    assert failed["status"]=="evolving"
    assert failed["iterations"][0]["evidence"]["sensor_success_candidate"] is False
    # Model a verifier contradiction discovered after the episode, followed by
    # a stale coding edit. The corrected infrastructure must replay the exact
    # immutable controller that produced the positive independent observation.
    failed["iterations"][0]["evidence"]["sensor_report"][
        "independent_task_outcome"]={"verified":True}
    first._save(failed)
    execution_path=run_root/"iterations"/"iteration_001"/"robot_execution.json"
    persisted_execution=json.loads(execution_path.read_text())
    persisted_execution["sensor_report"]["_harness_case_id"]="opaque-case-a"
    execution_path.write_text(json.dumps(persisted_execution))
    (run_root/"workspace"/"controller.py").write_text(
        "def run(robot):\n    raise RuntimeError('stale reactive edit')\n")

    class CorrectedVerifierDeployment(ClosedLoopDeployment):
        def dispatch(self,method,arguments):
            if method=="verify":return {"verified":True,"corrected_infrastructure":True}
            return super().dispatch(method,arguments)
        def sensor_report(self,execution):
            return {"sensor_verification_passed":True,"benchmark_signal_exposed":False}

    class ModelMustNotRun:
        def decide(self,**unused):raise AssertionError("model must not run during replay")

    class ReplayFactory:
        def __init__(self):self.selected=[]
        def select_case(self,case_handle):self.selected.append(case_handle)
        def __call__(self):return CorrectedVerifierDeployment()

    replay_factory=ReplayFactory()
    resumed=EvolutionEngine(root=run_root,model=ModelMustNotRun(),
        deployment_factory=replay_factory,python=PYTHON,
        retry_locked_validation_once=True)
    state=resumed.run(task="move the cube",skill_name="retry_skill",max_iterations=2)
    assert state["status"]=="sensor_success"
    replay=state["iterations"][1]
    assert replay["coding_passes"]==0
    assert replay["infrastructure_replay_without_model"] is True
    assert replay["locked_validation_retry_after_infrastructure_change"] is True
    assert replay["infrastructure_replay_source_iteration"]==1
    assert replay_factory.selected==["opaque-case-a"]


def test_libero_factory_partial_episode_does_not_consume_case(tmp_path,monkeypatch):
    import embodied_codex.examples.run_libero as run_libero

    root=tmp_path/"run";(root/"episodes").mkdir(parents=True)
    for index in range(1,4):(root/"episodes"/f"episode_{index:03d}").mkdir()
    (root/"state.json").write_text(json.dumps({"iterations":[
        {"evidence":{"committed":True}},
        {"evidence":{"committed":True}},
    ]}))
    episodes=[type("Episode",(),{"case_handle":handle})()
              for handle in ("a","b","c")]
    monkeypatch.setattr(run_libero,"LiberoDeployment",lambda **kwargs:kwargs)
    factory=run_libero.Factory(episodes=episodes,run_root=root,
        capabilities={},capability_contracts={},verifiers={})
    deployment=factory()
    assert deployment["episode"].case_handle=="c"
    assert deployment["artifact_dir"].name=="episode_004"


def test_libero_runtime_model_roles_are_migrated_once_and_then_immutable(tmp_path):
    import embodied_codex.examples.run_libero as run_libero

    root=tmp_path/"run";root.mkdir()
    configuration=root/"harness_configuration.json"
    configuration.write_text(json.dumps({"protocol":"test"}))
    roles={
        "coding_agent":{"model":"gpt-5.6-sol","reasoning_effort":"high"},
        "visual_relation_grounder":{"model":"gpt-5.6-sol",
                                    "reasoning_effort":"high"},
        "independent_task_outcome_verifier":{
            "model":"gpt-5.6-sol","reasoning_effort":"low",
            "consensus_rounds":3,"total_timeout_seconds":90.0},
    }
    run_libero._bind_runtime_model_configuration(root,roles)
    saved=json.loads(configuration.read_text())
    assert saved["runtime_model_configuration"]==roles
    assert saved["configuration_migrations"][-1]["kind"]==(
        "bind_runtime_model_roles_v1")
    run_libero._bind_runtime_model_configuration(root,roles)
    changed={**roles,"independent_task_outcome_verifier":{
        **roles["independent_task_outcome_verifier"],"reasoning_effort":"high"}}
    with pytest.raises(RuntimeError,match="runtime model configuration mismatch"):
        run_libero._bind_runtime_model_configuration(root,changed)


def test_libero_factory_marks_nontransactional_physical_episode_as_aborted(tmp_path,monkeypatch):
    import embodied_codex.examples.run_libero as run_libero

    root=tmp_path/"run";episodes_root=root/"episodes"
    committed=episodes_root/"episode_001";orphan=episodes_root/"episode_002"
    committed.mkdir(parents=True);orphan.mkdir(parents=True)
    trace=orphan/"adapter_trace.json"
    trace.write_text(json.dumps([{"event":"observe"},{"event":"act"}]))
    (orphan/"rollout.mp4").write_bytes(b"partial video")
    (root/"state.json").write_text(json.dumps({"iterations":[
        {"evidence":{"sensor_report":{"trace_path":str(committed/"adapter_trace.json")}}}
    ]}))
    cases=[type("Episode",(),{"case_handle":handle})() for handle in ("a","b")]
    monkeypatch.setattr(run_libero,"LiberoDeployment",lambda **kwargs:kwargs)

    factory=run_libero.Factory(episodes=cases,run_root=root,
        capabilities={},capability_contracts={},verifiers={})

    marker=json.loads((orphan/"aborted_infrastructure.json").read_text())
    assert marker["status"]=="aborted_infrastructure"
    assert marker["robot_action_count"]==1
    assert marker["benchmark_signal_exposed"] is False
    deployment=factory()
    assert deployment["episode"].case_handle=="b"
    assert deployment["artifact_dir"].name=="episode_003"


def test_libero_factory_forced_replay_does_not_advance_normal_case(tmp_path,monkeypatch):
    import embodied_codex.examples.run_libero as run_libero

    root=tmp_path/"run";(root/"episodes").mkdir(parents=True)
    (root/"state.json").write_text(json.dumps({"iterations":[
        {"evidence":{"committed":True}},
    ]}))
    episodes=[type("Episode",(),{"case_handle":handle})() for handle in ("a","b","c")]
    monkeypatch.setattr(run_libero,"LiberoDeployment",lambda **kwargs:kwargs)
    factory=run_libero.Factory(episodes=episodes,run_root=root,
        capabilities={},capability_contracts={},verifiers={})
    factory.select_case("a")
    assert factory()["episode"].case_handle=="a"
    assert factory()["episode"].case_handle=="b"


def test_frozen_skill_success_requires_last_sensor_verification():
    execution={"completed":True,"result":{"status":"sensor_success"},"rpc_events":[
        {"method":"verify","result":{"verified":True}}]}
    assert _sensor_success(execution,{"sensor_verification_passed":True}) is True
    assert _sensor_success({**execution,"rpc_events":execution["rpc_events"]+
                            [{"method":"record","result":{}}]},
                           {"sensor_verification_passed":True}) is False
    assert _sensor_success(execution,{"sensor_verification_passed":False}) is False


def test_visual_support_verifier_requires_fresh_target_overlap():
    detector=object.__new__(OpenVocabularyRGBD)
    target={"world_xyz":[.05,.05,.915],
            "world_bounds_10_90":[[0.,0.,.91],[.10,.10,.92]]}
    centered={"world_xyz":[.05,.05,.95],
              "world_bounds_10_90":[[.03,.03,.915],[.07,.07,.98]]}
    edge={"world_xyz":[.10,.05,.95],
          "world_bounds_10_90":[[.08,.03,.915],[.12,.07,.98]]}
    payload={"frame":{},"object_query":"bowl","target_query":"plate",
             "source_world_xyz":[-.2,.05,.95],"target_world_xyz":[.05,.05,.915]}
    detector.detect=lambda unused:{"detections":{"bowl":[centered],"plate":[target]}}
    proof=detector.verify_support_relation(payload)
    assert proof["verified"] is True and proof["support_overlap_fraction"] == 1.0
    detector.detect=lambda unused:{"detections":{"bowl":[edge],"plate":[target]}}
    proof=detector.verify_support_relation(payload)
    assert proof["verified"] is False
    assert proof["support_overlap_fraction"] < proof["minimum_support_overlap"]

    # When an occluding bowl is also labeled as a plate, independent support
    # height and cross-query alias rejection must select the real plate.
    real_target={"world_xyz":[.08,.05,.915],
                 "world_bounds_10_90":[[0.,0.,.91],[.10,.10,.92]]}
    alias_target={"world_xyz":centered["world_xyz"],
                  "world_bounds_10_90":centered["world_bounds_10_90"]}
    detector.detect=lambda unused:{"detections":{
        "bowl":[centered],"plate":[alias_target,real_target]}}
    proof=detector.verify_support_relation(payload)
    assert proof["target"]["world_xyz"]==real_target["world_xyz"]
    assert proof["verified"] is True

    # A same-category distractor can remain near the old source. A prior
    # attachment receipt for this exact source plus displacement of the selected
    # placed object proves the manipulated instance transitioned away.
    distractor={"world_xyz":[-.195,.05,.95],
                "world_bounds_10_90":[[-.22,.03,.915],[-.17,.07,.98]]}
    detector.detect=lambda unused:{"detections":{
        "bowl":[centered,distractor],"plate":[target]}}
    proof=detector.verify_support_relation(payload)
    assert proof["verified"] is False
    proof=detector.verify_support_relation({**payload,"source_transport_verified":True})
    assert proof["verified"] is True
    assert proof["geometric_source_vacated"] is False
    assert proof["source_vacancy_method"]==(
        "prior_attachment_and_selected_object_displacement")

    far_target={"world_xyz":[.05,-.10,.915],
                "world_bounds_10_90":[[0.,-.15,.91],[.10,-.05,.92]]}
    anchored={**payload,"target_world_bounds_10_90":target["world_bounds_10_90"]}
    detector.detect=lambda unused:{"detections":{"bowl":[centered],"plate":[far_target]}}
    proof=detector.verify_support_relation(anchored)
    assert proof["verified"] is True
    assert proof["target_geometry_source"]=="pre_action_sensor_anchor"

    # A nearby post-placement bowl mask may also be labeled as the support.
    # Even when its combined association rank is below the broad rank limit,
    # its surface is far above the independent pre-action support height and
    # must not replace the anchored plate geometry.
    raised_alias={"world_xyz":[.025,.05,.946],
                  "world_bounds_10_90":[[-.01,.01,.919],[.07,.09,.960]]}
    detector.detect=lambda unused:{"detections":{"bowl":[centered],
                                                   "plate":[raised_alias]}}
    proof=detector.verify_support_relation(anchored)
    assert proof["verified"] is True
    assert proof["target_geometry_source"]=="pre_action_sensor_anchor"
    assert proof["target_surface_height_error_m"] > \
        proof["maximum_target_surface_height_error_m"]


def test_libero_adapter_chains_attachment_receipt_to_same_source_support_verifier():
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment=object.__new__(LiberoDeployment)
    deployment.references={
        "source":{"world_xyz":[0,0,1],
                  "world_bounds_10_90":[[-.05,-.05,.95],[.05,.05,1.05]]},
        "target":{"world_xyz":[.2,0,.9],
                  "world_bounds_10_90":[[.15,-.05,.89],[.25,.05,.91]]},
    }
    captured=[]
    deployment.verifiers={
        "visual_attachment":lambda payload:{"verified":True},
        "visual_support_relation":lambda payload:(
            captured.append(dict(payload)) or {"verified":True}),
    }
    deployment.trace=[];deployment.last_verify=False;deployment.verified_attachments=set()
    attachment=deployment._verify("visual_attachment",{
        "frame":{},"object_query":"bowl","source_ref":"source"})
    assert attachment["verified"] is True
    support=deployment._verify("visual_support_relation",{
        "frame":{},"object_query":"bowl","target_query":"plate",
        "source_ref":"source","target_ref":"target"})
    assert support["verified"] is True
    assert captured[0]["source_transport_verified"] is True
    assert captured[0]["source_world_xyz"]==[0,0,1]
    assert captured[0]["target_world_bounds_10_90"]==[
        [.15,-.05,.89],[.25,.05,.91]]


def test_libero_adapter_separates_original_source_from_retry_attachment_receipt():
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment=object.__new__(LiberoDeployment)
    deployment.references={
        "original":{"world_xyz":[0,0,1]},
        "retry":{"world_xyz":[.03,0,1]},
        "target":{"world_xyz":[.2,0,.9]},
    }
    captured=[]
    deployment.verifiers={
        "visual_attachment":lambda payload:{"verified":True},
        "visual_support_relation":lambda payload:(
            captured.append(dict(payload)) or {"verified":True}),
    }
    deployment.trace=[];deployment.last_verify=False;deployment.verified_attachments=set()
    deployment._verify("visual_attachment",{
        "frame":{},"object_query":"bowl","source_ref":"retry"})
    deployment._verify("visual_support_relation",{
        "frame":{},"object_query":"bowl","target_query":"plate",
        "source_ref":"original","transport_ref":"retry","target_ref":"target"})
    assert captured[0]["source_world_xyz"]==[0,0,1]
    assert captured[0]["source_transport_verified"] is True

    deployment.verified_attachments.clear();captured.clear()
    deployment._verify("visual_support_relation",{
        "frame":{},"object_query":"bowl","target_query":"plate",
        "source_ref":"original","transport_ref":"retry","target_ref":"target"})
    assert captured[0]["source_transport_verified"] is False


def test_engineering_agent_can_see_only_current_run_sensor_images(tmp_path):
    run_root=tmp_path/"run"; artifact=run_root/"iterations"/"iteration_001"
    surface=EngineeringSurface(workspace=TaskWorkspace(run_root/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    image_path=run_root/"episodes"/"episode_001"/"sensors"/"frame-000001"/"frame.png"
    image_path.parent.mkdir(parents=True)
    # Small lossless images are transported directly.
    import base64
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
    result=surface.view_sensor_image(str(image_path))["_embodied_codex_image"]
    assert result["mime_type"]=="image/png" and result["data_base64"]
    assert str(image_path) in surface.list_sensor_artifacts("episodes/**/*.png")
    log_path=run_root/"episodes"/"episode_001"/"adapter_trace.json"
    log_path.write_text('{"ok": true}\n')
    assert '"ok": true' in surface.read_run_artifact(str(log_path))["content"]
    workspace_image=run_root/"workspace"/"montage.png"
    workspace_image.write_bytes(image_path.read_bytes())
    assert surface.view_sensor_image("montage.png")["_embodied_codex_image"]["path"]==str(workspace_image)
    artifact_ref=result["artifact_ref"]
    assert surface.view_sensor_image(artifact_ref=artifact_ref)["_embodied_codex_image"][
        "source_sha256"]==result["source_sha256"]
    with pytest.raises(RuntimeError,match="unknown image artifact_ref"):
        surface.view_sensor_image(artifact_ref="image-not-registered")
    with pytest.raises(RuntimeError,match="exactly one"):
        surface.view_sensor_image()
    outside=tmp_path/"outside.png";outside.write_bytes(image_path.read_bytes())
    with pytest.raises(RuntimeError):surface.view_sensor_image(str(outside))
    with pytest.raises(RuntimeError):surface.list_sensor_artifacts("../**/*")
    deployment_metadata=run_root/"episodes"/"episode_001"/"deployment.json"
    deployment_metadata.write_text('{"task_index": 3, "state_index": 9}\n')
    assert str(deployment_metadata) not in surface.list_sensor_artifacts("episodes/**/*")
    with pytest.raises(RuntimeError,match="controller-visible"):
        surface.read_run_artifact(str(deployment_metadata),1,10)


def test_sensor_image_delivery_compresses_large_png_without_mutating_evidence(tmp_path):
    import cv2
    import hashlib
    import numpy as np

    run=tmp_path/"run";episode=run/"episodes"/"episode_001"
    episode.mkdir(parents=True)
    image_path=episode/"sensors"/"frame-000001"/"camera.png"
    image_path.parent.mkdir(parents=True)
    generator=np.random.default_rng(7)
    image=generator.integers(0,256,size=(512,512,3),dtype=np.uint8)
    assert cv2.imwrite(str(image_path),image)
    original=image_path.read_bytes();assert len(original)>64*1024
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=run/"iterations"/"iteration_001")
    delivered=surface.view_sensor_image(str(image_path))["_embodied_codex_image"]
    assert delivered["mime_type"]=="image/jpeg"
    assert delivered["source_mime_type"]=="image/png"
    assert delivered["transport_preview"] is True
    assert delivered["delivery_bytes"]<delivered["source_bytes"]*0.8
    assert delivered["source_sha256"]==hashlib.sha256(original).hexdigest()
    assert image_path.read_bytes()==original


def test_libero_outcome_evidence_montages_external_and_wrist_views(tmp_path):
    import hashlib
    import numpy as np
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.artifact_dir=tmp_path/"episode"
    external=np.zeros((8,10,3),dtype=np.uint8);external[...,0]=200
    wrist=np.zeros((8,10,3),dtype=np.uint8);wrist[...,2]=180
    deployment.obs={"agentview_image":external,
                    "robot0_eye_in_hand_image":wrist}
    receipt=deployment._capture_outcome_rgb("after")
    path=Path(receipt["rgb_path"])
    assert path.is_file() and receipt["shape"]==[8,20,3]
    assert receipt["views"]==["agentview","robot0_eye_in_hand"]
    assert receipt["layout"]=="external_left_wrist_right"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt["rgb_sha256"]


def test_large_run_artifact_is_readable_in_bounded_line_chunks(tmp_path):
    run_root=tmp_path/"run";artifact=run_root/"iterations"/"iteration_001"
    surface=EngineeringSurface(workspace=TaskWorkspace(run_root/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    log=run_root/"episodes"/"episode_001"/"sensors"/"diagnostics"/"large.json"
    log.parent.mkdir(parents=True)
    line="x"*4096
    log.write_text("\n".join(f"{index:04d}:{line}" for index in range(700))+"\n")
    assert log.stat().st_size>2*1024*1024
    chunk=surface.read_run_artifact(str(log),start_line=501,end_line=510)
    assert chunk["start_line"]==501 and chunk["end_line"]==510
    assert chunk["total_lines"]==700 and "0500:" in chunk["content"]
    assert chunk["content_truncated"] is False
    latest=artifact/"robot_execution.json";latest.write_text('{"latest": true}\n')
    assert '"latest": true' in surface.read_run_artifact(
        "latest_robot_execution")["content"]


def test_conformance_audit_separates_harness_health_from_task_success(tmp_path):
    import json
    root=tmp_path/"run";iteration=root/"iterations"/"iteration_001"
    episode=root/"episodes"/"episode_001"
    iteration.mkdir(parents=True);episode.mkdir(parents=True)
    controller=iteration/"controller.py"
    controller.write_text("def run(robot):\n    return {'status':'sensor_failure'}\n")
    (episode/"adapter_trace.json").write_text("[]\n")
    (episode/"rollout.mp4").write_bytes(b"video")
    report={"controller_path":"controller.py","controller_snapshot":str(controller),
            "execution":{"completed":True,"error":None,"result":{"status":"sensor_failure"},
                             "runtime_isolation":"bubblewrap-controller-v1",
                             "rpc_output_projection":"adapter-explicit-allowlist-v1",
                             "rpc_output_defense":"kernel-evaluator-field-deny-v1",
                             "rpc_events":[]},
        "sensor_report":{"benchmark_signal_exposed":False,
            "trace_path":str(episode/"adapter_trace.json"),
            "rollout_path":str(episode/"rollout.mp4")},
        "sensor_success_candidate":False,
        "robot_contract_preflight":{"passed":True}}
    (iteration/"robot_execution.json").write_text(json.dumps(report))
    events=[
        {"type":"task","instruction":json.dumps({"previous_sensor_evidence":None})},
        {"type":"model_request","turn":1,"attempt":1,"message_count":2,
         "messages_sha256":"a"*64,"tool_schema_sha256":"b"*64,
         "system_prompt_sha256":"c"*64},
        {"type":"model_error","turn":1,"attempt":1,
         "error":"APIConnectionError: temporary transport failure"},
        {"type":"model","turn":1,"tool_calls":[{"name":"write_file"}]},
        {"type":"tool_result","turn":1,"name":"replace_in_file","ok":False,
         "error":"WorkspaceError: concurrent edit mismatch"},
        {"type":"tool_result","turn":1,"name":"replace_in_file","ok":True},
        {"type":"tool_result","turn":1,"name":"write_file","ok":True},
        {"type":"tool_result","turn":2,"name":"run_robot_controller","ok":True},
    ]
    (iteration/"agent_trace.jsonl").write_text(
        "".join(json.dumps(event)+"\n" for event in events))
    (root/"state.json").write_text(json.dumps({"task":"impossible task",
        "status":"evolving","iterations":[{"iteration":1,"evidence":{"failed":True}}]}))
    (root/"harness_configuration.json").write_text(json.dumps({
        "protocol":"embodied-codex-run-configuration-v2",
        "capability_root":str(root/"capabilities"),
        "experience_root":str(root/"experiences"),
        "isolation":{"engineering":"bubblewrap-workspace-v1",
            "controller":"bubblewrap-controller-v1","generated_tool":"bubblewrap-tool-v1"}}))
    audit=audit_run(root)
    assert audit["conformant"] is True
    for gate in ("engineering_workspace_isolated","controller_runtime_isolated",
                 "generated_tool_runtime_isolated","tool_manual_present",
                 "manual_source_separation","experience_asset_integrity",
                 "dependency_reproducibility","tool_contract_validated"):
        assert audit["gates"][gate] is True
    assert audit["metrics"]["sensor_successes"]==0
    assert len(audit["recovered_model_errors"])==1
    assert len(audit["recovered_tool_errors"])==1
    assert audit["interface_errors"]==[]

    events.append({"type":"tool_result","turn":3,"name":"robot.act","ok":False,
                   "error":"unsupported action"})
    (iteration/"agent_trace.jsonl").write_text(
        "".join(json.dumps(event)+"\n" for event in events))
    audit=audit_run(root)
    assert audit["conformant"] is False
    assert audit["gates"]["clean_engineering_interfaces"] is False

    # One malformed path among a same-turn batch is recovered when another
    # call to the exact Tool succeeds and the agent continues.
    events[-1]={"type":"tool_result","turn":3,"name":"view_sensor_image",
                "ok":True,"result":{"vision_delivered":True}}
    events.append({"type":"tool_result","turn":3,"name":"view_sensor_image",
                   "ok":False,"error":"RuntimeError: bad duplicated path"})
    (iteration/"agent_trace.jsonl").write_text(
        "".join(json.dumps(event)+"\n" for event in events))
    audit=audit_run(root)
    assert audit["conformant"] is True
    assert audit["recovered_tool_errors"][-1]["tool"]=="view_sensor_image"


def test_conformance_recovers_engineering_tool_error_across_resumed_iteration(tmp_path):
    import json

    root=tmp_path/"run"
    first=root/"iterations"/"iteration_001"
    second=root/"iterations"/"iteration_002"
    first.mkdir(parents=True);second.mkdir(parents=True)
    (first/"agent_trace.jsonl").write_text("".join(json.dumps(event)+"\n" for event in [
        {"type":"task","instruction":"iteration one"},
        {"type":"tool_result","turn":1,"name":"query_run_json","ok":False,
         "error":"RuntimeError: JSON pointer not found: /events"},
    ]))
    (second/"agent_trace.jsonl").write_text("".join(json.dumps(event)+"\n" for event in [
        {"type":"task","instruction":"resumed iteration"},
        {"type":"tool_result","turn":1,"name":"query_run_json","ok":True,
         "result":{"rows":[]}},
    ]))
    audit=audit_run(root)
    assert audit["interface_errors"]==[]
    assert audit["recovered_tool_errors"]==[{
        "type":"tool_error","tool":"query_run_json",
        "error":"RuntimeError: JSON pointer not found: /events"}]


def test_conformance_records_retired_skill_lookup_exception_as_contract_upgrade(tmp_path):
    import json
    root=tmp_path/"run";iteration=root/"iterations"/"iteration_001"
    iteration.mkdir(parents=True)
    (iteration/"agent_trace.jsonl").write_text(json.dumps({
        "type":"tool_result","turn":1,"name":"read_skill_source","ok":False,
        "error":"AssetError: invalid Skill id"})+"\n")
    audit=audit_run(root)
    assert audit["interface_errors"]==[]
    assert audit["recovered_tool_errors"]==[{
        "type":"tool_error","tool":"read_skill_source",
        "error":"AssetError: invalid Skill id",
        "recovered_by_contract_upgrade":"read-skill-source-soft-not-found-v1"}]


def test_execution_aliases_keep_current_and_previous_iterations_distinct(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    capabilities=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root)
    previous=tmp_path/"run"/"iterations"/"iteration_001"/"robot_execution.json"
    previous.parent.mkdir(parents=True);previous.write_text('{"iteration": 1}\n')
    current=tmp_path/"run"/"iterations"/"iteration_002"
    surface=EngineeringSurface(workspace=workspace,capabilities=capabilities,
        runtime=ControllerRuntime(python=PYTHON),deployment_factory=lambda:None,
        artifact_dir=current)
    result=surface.read_run_artifact("latest_robot_execution")
    assert result["path"]==str((current/"robot_execution.json").resolve())
    assert result["exists"] is False
    prior=surface.read_run_artifact("previous_robot_execution")
    assert prior["path"]==str(previous.resolve())
    assert '"iteration": 1' in prior["content"]
    with pytest.raises(FileNotFoundError,match="latest_robot_execution"):
        surface.inspect_execution_event("latest_robot_execution",0)
    current_execution=current/"robot_execution.json"
    current_execution.write_text('{"iteration": 2}\n')
    result=surface.read_run_artifact("latest_robot_execution")
    assert result["path"]==str(current_execution.resolve())
    prior=surface.read_run_artifact("previous_robot_execution")
    assert prior["path"]==str(previous.resolve())
    snapshot=previous.parent/"controller.py";snapshot.write_text("def run(robot): pass\n")
    # Before the next rollout commits its snapshot, the symbolic controller
    # evidence reference means the most recent immutable executed program.
    assert surface._evidence_reference("controller.py")==snapshot
    assert surface._evidence_reference("executed_controller")==snapshot
    current_snapshot=current/"controller.py";current_snapshot.write_text("def run(robot): return None\n")
    assert surface._evidence_reference("controller.py")==current_snapshot
    assert surface._evidence_reference("executed_controller")==current_snapshot
    source=surface.read_file(str(snapshot),1,20)
    assert source["exists"] is True and "def run" in source["content"]
    with pytest.raises(RuntimeError,match="inside the current run"):
        surface.read_file(str(tmp_path/"outside.py"),1,20)


def test_optional_robot_execution_symbols_return_missing_receipts_on_first_iteration(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    for reference in ("latest_robot_execution","previous_robot_execution",
                      "robot_execution.json"):
        result=surface.read_run_artifact(reference)
        assert result["exists"] is False and result["content"]==""
        assert result["total_lines"]==0 and result["next_start_line"] is None


def test_execution_event_inspection_and_asset_search_are_context_bounded(tmp_path):
    import json

    class SearchLibrary:
        def __init__(self):self.limits=[]
        def search(self,query,limit):
            self.limits.append(limit)
            return [{"tool_id":f"tool:{index}"} for index in range(limit)]

    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    execution=artifact/"robot_execution.json"
    execution.write_text(json.dumps({"execution":{"rpc_events":[{
        "method":"use","arguments":{"tool_id":"ranker:v001","payload":{"frame":{
            "frame_id":"frame-1","step":7,"cameras":{"agentview":{
                "rgb_path":"/run/frame.png","camera_to_world":[[1,0],[0,1]]}}}}},
        "result":{"result":{"candidates":[{"rank":index,"score":1/index}
            for index in range(1,101)]}}}]}}))
    library=SearchLibrary()
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=library,runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    bounded=surface.search_assets("candidate",["tool"],limit=50)
    assert bounded["requested_limit"]==50 and bounded["limit"]==20
    assert library.limits==[20] and len(bounded["tools"])==20
    assert set(bounded["tools"][0]) >= {"tool_id","description","input_fields",
        "required_inputs","output_fields","retrieval_score"}
    event=surface.inspect_execution_event("latest_robot_execution",0,3)
    assert len(json.dumps(event))<20_000
    candidates=event["event"]["result"]["result"]["candidates"]
    assert candidates["count"]==100 and len(candidates["head"])==3
    assert candidates["head"][0]=={"rank":1,"score":1.0}
    assert event["event"]["arguments"]["payload"]["frame"]=={
        "frame_id":"frame-1","step":7,
        "camera_rgb_paths":{"agentview":"/run/frame.png"}}
    with pytest.raises(RuntimeError,match="event_index out of range"):
        surface.inspect_execution_event("latest_robot_execution",1,3)


def test_execution_event_inspection_bounds_deep_model_candidate_trees(tmp_path):
    import json

    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    candidate={"rank_score":0.9,"world_xyz":[0.1,0.2,0.3],
        "rotation_world":[[float(row*3+column) for column in range(3)]
                          for row in range(3)],
        "diagnostics":{"samples":[{"matrix":[float(value)]*64,
                                     "scores":[float(value)]*200}
                                    for value in range(100)]}}
    execution=artifact/"robot_execution.json"
    execution.write_text(json.dumps({"execution":{"rpc_events":[{
        "method":"use","arguments":{"tool_id":"planner:v001"},
        "result":{"result":{"candidates":[candidate for _ in range(100)]}}}]}}))
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    event=surface.inspect_execution_event("latest_robot_execution",0,8)
    encoded=json.dumps(event)
    assert len(encoded)<20_000
    candidates=event["event"]["result"]["result"]["candidates"]
    assert candidates["count"]==100 and len(candidates["head"])==8
    assert candidates["head"][0]["rank_score"]==0.9
    assert candidates["head"][0]["world_xyz"]==[0.1,0.2,0.3]
    assert candidates["head"][0]["rotation_world"]=={
        "type":"list","truncated":True}


def test_run_json_query_filters_sorts_and_projects_large_candidate_arrays(tmp_path):
    import json

    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    evidence=run/"episodes"/"episode_001"/"sensors"/"planner.json"
    artifact.mkdir(parents=True);evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({"candidates":[
        {"id":index,"score":index/10,"collision":index%2==0,
         "pose":{"xyz":[index,0,1]},"large":[float(index)]*1000}
        for index in range(100)]}))
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    result=surface.query_run_json(str(evidence),"/candidates",
        filters=[{"field":"/collision","op":"eq","value":False},
                 {"field":"/score","op":"gte","value":5.0}],
        sort_by="/score",descending=True,fields=["/id","/score","/pose/xyz"],limit=3)
    assert result["source_count"]==100 and result["matched_count"]==25
    assert result["rows"]==[
        {"/id":99,"/score":9.9,"/pose/xyz":[99,0,1]},
        {"/id":97,"/score":9.7,"/pose/xyz":[97,0,1]},
        {"/id":95,"/score":9.5,"/pose/xyz":[95,0,1]}]
    assert len(json.dumps(result))<2000
    outside=tmp_path/"outside.json";outside.write_text('{"candidates": []}')
    with pytest.raises(RuntimeError,match="inside the current run"):
        surface.query_run_json(str(outside),"/candidates")


def test_run_json_query_compacts_broad_projected_tool_results(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    candidates=[{"score":index/100,"matrix":[float(index)]*100}
                for index in range(100)]
    (artifact/"robot_execution.json").write_text(json.dumps({"execution":{
        "rpc_events":[{"method":"use","arguments":{"tool_id":"planner:v1"},
                       "result":{"result":{"candidates":candidates}}}]}}))
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)

    result=surface.query_run_json("latest_robot_execution","/execution/rpc_events",
        fields=["method","arguments","result"],limit=10)

    assert len(json.dumps(result))<12_000
    compacted=result["rows"][0]["result"]["result"]["candidates"]
    assert compacted["count"]==100 and len(compacted["head"])==8


def test_run_json_query_transparently_unwraps_rpc_result_receipts(tmp_path):
    import json

    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    execution=artifact/"robot_execution.json"
    execution.write_text(json.dumps({"execution":{"rpc_events":[{
        "method":"use","result":{"tool_id":"planner:v1","result":{
            "candidates":[{"score":0.4},{"score":0.9}]}}}]}}))
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    for pointer in ("/events/0/result/candidates",
                    "/rpc_events/0/result/candidates",
                    "/execution/events/0/result/candidates",
                    "/execution/rpc_events/0/result/candidates"):
        result=surface.query_run_json("latest_robot_execution",pointer,
            sort_by="/score",descending=True,limit=1)
        assert result["rows"]==[{"score":0.9}]


def test_run_json_query_accepts_top_level_field_shorthand_and_reports_sparse_fields(tmp_path):
    import json
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    (artifact/"robot_execution.json").write_text(json.dumps({"execution":{"rpc_events":[
        {"method":"act","arguments":{"action":{"type":"settle"}},
         "result":{"reached":True}}
    ]}}))
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    result=surface.query_run_json("latest_robot_execution","/execution/rpc_events",
        fields=["method","arguments","result"])
    assert result["rows"]==[{"method":"act",
        "arguments":{"action":{"type":"settle"}},"result":{"reached":True}}]
    sparse=surface.query_run_json("latest_robot_execution","/execution/rpc_events",
        fields=["method","tool_id"])
    assert sparse["rows"]==[{"method":"act","tool_id":None}]
    assert sparse["projection_warnings"]==[{
        "field":"tool_id","missing_rows":1,"returned_rows":1,
        "note":"null denotes a field absent from this heterogeneous row; use filters or a nested pointer to narrow the projection"}]


def test_structured_queries_accept_hash_validated_experience_evidence(tmp_path):
    import json

    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    source=tmp_path/"source_execution.json"
    source.write_text(json.dumps({"execution":{"rpc_events":[
        {"method":"act","result":{"reached":False}},
        {"method":"act","result":{"reached":True}},
    ]}}))
    experiences=ExperienceLibrary(tmp_path/"experiences")
    registered=experiences.register(name="staged_motion",summary="Stage long moves.",
        applicability="Pose-controlled robots.",keywords=["motion"],
        evidence_paths=[source])
    experience_id=registered["experience_id"]
    manifest=experiences.inspect(experience_id)
    asset_ref=f"{experience_id}#{manifest['evidence'][0]['path']}"
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact,experiences=experiences)

    queried=surface.query_run_json(asset_ref,"/execution/events",
        filters=[{"field":"/result/reached","op":"eq","value":True}],limit=4)
    assert queried["matched_count"]==1
    assert queried["rows"]==[{"method":"act","result":{"reached":True}}]
    inspected=surface.inspect_execution_event(asset_ref,1)
    assert inspected["event"]["result"]["reached"] is True


def test_robot_failure_summary_keeps_full_evidence_on_disk_but_bounds_model_context(tmp_path):
    import json
    from embodied_codex.legacy.engineering import _agent_robot_summary

    candidates=[{"rank":index,"score":index/1000,"matrix":[float(index)]*16}
                for index in range(1000)]
    snapshot=tmp_path/"iterations"/"iteration_001"/"controller.py"
    snapshot.parent.mkdir(parents=True);snapshot.write_text("def run(robot): pass\n")
    result={"controller_path":"controller.py","controller_snapshot":str(snapshot),
        "execution":{"completed":True,"error":None,
            "result":{"sensor_failure":"no_contact","candidates":candidates},
            "rpc_events":[
                {"method":"use","arguments":{"tool_id":"grasp:v1"},
                 "result":{"result":{"candidates":candidates}}},
                {"method":"record","arguments":{"event":{
                    "phase":"ranked","candidates":candidates}}}]},
        "sensor_report":{"sensor_verification_passed":False,
                         "diagnostics":candidates},
        "sensor_success_candidate":False}

    full_path=snapshot.parent/"robot_execution.json"
    full_path.write_text(json.dumps(result))
    assert full_path.stat().st_size>500_000

    tool_summary=_agent_robot_summary(result,full_path)
    persistent_summary=EvolutionEngine._brief(result)
    assert len(json.dumps(tool_summary))<30_000
    assert len(json.dumps(persistent_summary))<30_000
    assert tool_summary["full_execution_artifact"]==str(full_path.resolve())
    assert persistent_summary["execution_artifact_ref"]=="previous_robot_execution"
    compacted=persistent_summary["controller_result"]["candidates"]
    assert compacted["count"]==1000 and len(compacted["head"])==6

    # A long physical recovery must also remain bounded when the payload size
    # comes from many legitimate motion receipts rather than one candidate tree.
    motions=[]
    for index in range(80):
        motions.append({"method":"act","arguments":{"action":{
            "type":"move_to_pose","pose_ref":"pose-1","offset":[0,0,index/1000],
            "rotation_matrix":[[1,0,0],[0,1,0],[0,0,1]],"gripper":-1}},
            "result":{"type":"move_to_pose","reached":index<79,"step":index,
                "eef_before":[0,0,0],"eef_after":[0,0,index/1000],
                "target_xyz":[0,0,index/1000],
                "target_quaternion_xyzw":[0,0,0,1],
                "final_position_error_m":0.001}})
    long_result={**result,"execution":{**result["execution"],"rpc_events":motions}}
    bounded=_agent_robot_summary(long_result,full_path)
    assert bounded["rpc_event_count"]==80
    assert len(bounded["rpc_evidence"])==24
    assert bounded["rpc_evidence_omitted_range"]=={"start":6,"end":61,"count":56}
    assert len(json.dumps(bounded))<30_000


def test_conformance_requires_real_later_rollout_to_recover_controller_error(tmp_path):
    import json
    root=tmp_path/"recovered_controller"
    for index,completed in ((1,False),(2,True)):
        iteration=root/"iterations"/f"iteration_{index:03d}"
        episode=root/"episodes"/f"episode_{index:03d}"
        iteration.mkdir(parents=True);episode.mkdir(parents=True)
        controller=iteration/"controller.py"
        controller.write_text("def run(robot):\n    return {'status':'sensor_failure'}\n")
        trace=episode/"adapter_trace.json";trace.write_text("[]\n")
        rollout=episode/"rollout.mp4";rollout.write_bytes(b"video")
        report={"controller_path":"controller.py","controller_snapshot":str(controller),
            "execution":{"completed":completed,
                         "error":None if completed else "TypeError: bad rotation",
                         "result":{"status":"sensor_failure"},"rpc_events":[]},
            "sensor_report":{"benchmark_signal_exposed":False,
                             "trace_path":str(trace),"rollout_path":str(rollout)},
            "sensor_success_candidate":False,
            "robot_contract_preflight":{"passed":True}}
        (iteration/"robot_execution.json").write_text(json.dumps(report))
        (iteration/"agent_trace.jsonl").write_text(
            json.dumps({"type":"task","instruction":json.dumps({
                "previous_sensor_evidence":None if index==1 else {"failure":True}})})+"\n"+
            json.dumps({"type":"model","turn":1,"tool_calls":[]})+"\n"+
            json.dumps({"type":"tool_result","turn":1,"name":"write_file","ok":True})+"\n")
    (root/"state.json").write_text(json.dumps({"task":"move object","status":"evolving",
        "iterations":[{"evidence":{}},{"evidence":{}}]}))
    audit=audit_run(root)
    assert audit["controller_errors"]==[]
    assert audit["recovered_controller_errors"]==[{
        "iteration":"iteration_001","error":"TypeError: bad rotation"}]
    assert audit["gates"]["controller_execution_completed"] is True


def test_coding_agent_delivers_sensor_image_as_vision_input(tmp_path):
    import json
    registry=FunctionRegistry()
    registry.add("view_sensor_image","view",{"type":"object"},lambda **_: {
        "_embodied_codex_image":{"path":"frame.png","mime_type":"image/png",
                                  "data_base64":"YWJj"}})
    class VisionAwareModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"view_sensor_image",{})
            assert any(isinstance(m.get("content"),list) and
                any(x.get("type")=="image_url" for x in m["content"])
                for m in messages)
            return {"content":"image inspected","tool_calls":[]}
    trace=tmp_path/"trace.jsonl"
    result=CodingAgent(model=VisionAwareModel(),registry=registry,
        system_prompt="test",trace_path=trace).run("inspect")
    assert result["completed"] is True
    assert "data_base64" not in trace.read_text()
    requests=[json.loads(line) for line in trace.read_text().splitlines()
              if '"type": "model_request"' in line]
    assert len(requests)==2 and requests[1]["image_payload_sha256"]
    assert len(requests[0]["tool_schema_sha256"])==64


def test_coding_agent_defers_images_beyond_per_turn_batch_budget(tmp_path):
    registry=FunctionRegistry()
    registry.add("view_sensor_image","view",{"type":"object"},lambda path: {
        "_embodied_codex_image":{"path":path,"mime_type":"image/png",
                                  "data_base64":"YWJj"}},
        evidence_policy="image_twice")
    class BatchedVisionModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:
                return {"content":"","tool_calls":[{
                    "id":str(index),"name":"view_sensor_image",
                    "arguments":json.dumps({"path":f"frame-{index}.png"})}
                    for index in range(5)]}
            images=[item for message in messages if isinstance(message.get("content"),list)
                    for item in message["content"] if item.get("type")=="image_url"]
            receipts=[json.loads(message["content"]) for message in messages
                      if message.get("role")=="tool"]
            assert len(images)==3
            assert sum(item.get("delivery_deferred") is True for item in receipts)==2
            return {"content":"bounded batch inspected","tool_calls":[]}
    trace=tmp_path/"trace.jsonl"
    result=CodingAgent(model=BatchedVisionModel(),registry=registry,
        system_prompt="test",trace_path=trace,per_turn_images=3).run("inspect")
    assert result["completed"] is True
    events=[json.loads(line) for line in trace.read_text().splitlines()]
    delivered=[event["result"]["vision_delivered"] for event in events
               if event.get("type")=="tool_result" and event.get("name")=="view_sensor_image"]
    assert delivered==[True,True,True,False,False]


def test_coding_agent_suppresses_duplicate_read_by_normalized_arguments(tmp_path):
    import json
    registry=FunctionRegistry();invocations=[]
    registry.add("inspect_evidence","inspect",{"type":"object"},
                 lambda **arguments:invocations.append(arguments) or {
                     "path":"evidence.json","content":"immutable evidence"},
                 evidence_policy="read_once")
    class RepeatingModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"inspect_evidence",{"b":2,"a":1})
            if self.calls==2:return {"content":"","tool_calls":[{
                "id":"2","name":"inspect_evidence","arguments":'{"a":1,"b":2}'}]}
            assert "duplicate_read_suppressed" in json.dumps(messages)
            return {"content":"act on delivered evidence","tool_calls":[]}
    result=CodingAgent(model=RepeatingModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl").run("inspect")
    assert result["completed"] is True
    assert invocations==[{"b":2,"a":1}]
    assert result["tool_results"][-1]["duplicate_read_suppressed"] is True


def test_coding_agent_ends_pass_when_duplicate_reads_do_not_lead_to_action(tmp_path):
    registry=FunctionRegistry();invocations=[]
    registry.add("inspect_evidence","inspect",{"type":"object"},
                 lambda **arguments:invocations.append(arguments) or {
                     "path":"evidence.json","content":"immutable evidence"},
                 evidence_policy="read_once")
    class StalledModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            return _call(self.calls,"inspect_evidence",{"page":1})
    model=StalledModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",post_duplicate_read_max_turns=2).run("inspect")
    assert result["completed"] is False
    assert result["error"]=="engineering action deadline after repeated evidence request"
    assert model.calls==4
    assert invocations==[{"page":1}]


def test_coding_agent_new_evidence_clears_duplicate_read_deadline(tmp_path):
    registry=FunctionRegistry();invocations=[]
    registry.add("inspect_evidence","inspect",{"type":"object"},
                 lambda **arguments:invocations.append(arguments) or arguments,
                 evidence_policy="read_once")
    class RecoveringModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls in {1,2}:return _call(self.calls,"inspect_evidence",{"page":1})
            if self.calls==3:return _call(3,"inspect_evidence",{"page":2})
            if self.calls in {4,5}:return _call(self.calls,"inspect_evidence",{"page":2})
            return {"content":"acted on evidence","tool_calls":[]}
    result=CodingAgent(model=RecoveringModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",post_duplicate_read_max_turns=2).run("inspect")
    assert result["completed"] is True
    assert invocations==[{"page":1},{"page":2}]


def test_working_memory_reads_are_deduplicated_without_consuming_evidence_budget(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("read_workspace","read source",{"type":"object","properties":{
                     "page":{"type":"integer"}},"required":["page"]},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="working_memory")
    registry.add("read_sensor","read evidence",{"type":"object","properties":{
                     "page":{"type":"integer"}},"required":["page"]},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="read_once")
    class SourceThenEvidenceModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls<=8:return _call(self.calls,"read_workspace",{"page":self.calls})
            if self.calls<=14:return _call(self.calls,"read_sensor",{"page":self.calls})
            if self.calls==15:return _call(15,"read_sensor",{"page":15})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=SourceThenEvidenceModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl",
        max_evidence_deliveries=6).run("inspect")
    assert result["completed"] is True
    sensor_results=[row for row in result["tool_results"] if row["name"]=="read_sensor"]
    assert sensor_results[-1]["evidence_acquisition_paused"] is True
    assert len(reads)==14


def test_working_memory_pages_have_separate_action_budget(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("read_workspace","read source",{"type":"object","properties":{
                     "page":{"type":"integer"}},"required":["page"]},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="working_memory")
    class WorkspaceLoopModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            return _call(self.calls,"read_workspace",{"page":self.calls})
    model=WorkspaceLoopModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",max_working_memory_deliveries=6,
        post_evidence_pause_max_turns=2).run("inspect")
    assert result["completed"] is False
    assert result["error"]=="engineering action deadline after evidence pause"
    assert len(reads)==6
    paused=[row for row in result["tool_results"]
            if row.get("working_memory_acquisition_paused")]
    assert paused and paused[0]["delivery_limit"]==6


def test_command_workspace_mutation_reopens_reads_and_requires_controller_run(tmp_path):
    workspace=TaskWorkspace(tmp_path/"workspace")
    workspace.write_file("controller.py","def run(robot):\n    return 1\n")
    registry=FunctionRegistry()
    registry.add("read_source","read",{"type":"object"},
                 lambda **_:workspace.read_file("controller.py"),
                 evidence_policy="working_memory",evidence_group="workspace")
    registry.add("command","command",{"type":"object","properties":{
                     "argv":{"type":"array","items":{"type":"string"}}},
                     "required":["argv"]},workspace.run_command,
                 evidence_policy="budgeted_output",evidence_group="workspace")
    class CommandEditingModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"read_source",{})
            if self.calls==2:return _call(2,"command",{"argv":[sys.executable,"-c",
                "from pathlib import Path; Path('controller.py').write_text('def run(robot):\\n    return 2\\n')"]})
            if self.calls==3:return _call(3,"read_source",{})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=CommandEditingModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl").run("edit")
    assert result["completed"] is True
    reads=[row for row in result["tool_results"] if row["name"]=="read_source"]
    assert len(reads)==2
    assert all(not row.get("duplicate_read_suppressed") for row in reads)
    command=next(row for row in result["tool_results"] if row["name"]=="command")
    assert command["result"]["_embodied_codex_controller_mutated"] is True


def test_compile_command_does_not_count_cache_files_as_workspace_mutation(tmp_path):
    workspace=TaskWorkspace(tmp_path/"workspace")
    workspace.write_file("controller.py","def run(robot):\n    return 1\n")
    result=workspace.run_command([sys.executable,"-m","py_compile","controller.py"])
    assert result["exit_code"]==0
    assert result.get("_embodied_codex_engineering_progress") is not True


def test_semantically_empty_controller_edit_does_not_count_as_agent_progress(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("edit_controller","edit",{"type":"object"},
                 lambda **_:{"_embodied_codex_semantic_progress":False},
                 evidence_policy="invalidates_reads",evidence_progress=True,
                 execution_progress=True)
    registry.add("think","think",{"type":"object"},
                 lambda **_:reads.append(True) or {"ok":True})
    class NoOpThenFinishModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"edit_controller",{})
            if self.calls<=7:return _call(self.calls,"think",{})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=NoOpThenFinishModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl",
        post_mutation_max_turns=4).run("edit")
    assert result["completed"] is True
    assert len(reads)==6


def test_engineering_surface_marks_log_only_controller_edit_semantically_empty(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.record({"strategy": "v1"})
    return {"sensor_failure": True}
''')
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root),
        runtime=object(),deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001")
    result=surface.registry().invoke("replace_in_file",{
        "path":"controller.py","old":"v1","new":"v2"})
    assert result["controller_semantic_progress"] is False


def test_coding_agent_bounds_reads_after_unexecuted_controller_mutation(tmp_path):
    registry=FunctionRegistry();reads=[];runs=[]
    registry.add("read_evidence","read",{"type":"object"},
                 lambda **_:reads.append(True) or {"content":"evidence"},
                 evidence_policy="read_once")
    registry.add("run_robot_controller","run",{"type":"object"},
                 lambda **_:runs.append(True) or {"completed":True},
                 execution_progress=True)
    class ReadThenRunModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls<=4:return _call(self.calls,"read_evidence",{"page":self.calls})
            if self.calls==5:return _call(self.calls,"run_robot_controller",{})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=ReadThenRunModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl",
        executable_pending=True,post_mutation_read_deliveries=2).run("run pending")
    assert result["completed"] is True
    assert len(reads)==2
    assert runs==[True]


def test_failed_compile_reopens_one_local_controller_repair_read(tmp_path):
    registry=FunctionRegistry();reads=[];runs=[]
    registry.add("read_file","read",{"type":"object","properties":{
        "path":{"type":"string"},"start_line":{"type":"integer"},
        "end_line":{"type":"integer"}},"required":["path"]},
        lambda **arguments:reads.append(arguments) or {"content":"broken source"},
        evidence_policy="working_memory",evidence_group="workspace")
    registry.add("edit_controller","edit",{"type":"object"},
        lambda **_:{"written":True},evidence_policy="invalidates_reads",
        evidence_progress=True,execution_progress=True,
        invalidates_evidence_groups=("workspace",))
    compile_results=iter([
        {"argv":["python","-m","py_compile","controller.py"],"exit_code":1,
         "output":"Sorry: IndentationError: expected an indented block "
                  "(controller.py, line 178)"},
        {"argv":["python","-m","py_compile","controller.py"],"exit_code":0,
         "output":""},
    ])
    registry.add("run_command","command",{"type":"object"},
        lambda **_:next(compile_results),evidence_policy="budgeted_output",
        evidence_group="workspace")
    registry.add("run_robot_controller","robot",{"type":"object"},
        lambda **_:runs.append(True) or {"completed":True,"sensor_success_candidate":True},
        execution_progress=True)
    class CompileRepairModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls<=2:
                return _call(self.calls,"read_file",{
                    "path":"controller.py","start_line":self.calls,
                    "end_line":self.calls})
            if self.calls==3:return _call(3,"edit_controller",{})
            if self.calls==4:return _call(4,"run_command",{})
            if self.calls==5:return _call(5,"read_file",{
                "path":"controller.py","start_line":170,"end_line":190})
            if self.calls==6:return _call(6,"edit_controller",{})
            if self.calls==7:return _call(7,"run_command",{})
            return _call(8,"run_robot_controller",{})
    result=CodingAgent(model=CompileRepairModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl",
        executable_pending=True,post_mutation_read_deliveries=2,
        post_mutation_max_turns=10).run("repair and run")
    assert result["completed"] is True
    assert len(reads)==3
    assert runs==[True]
    repair=result["tool_results"][4]
    assert repair["name"]=="read_file"
    assert repair.get("evidence_acquisition_paused") is not True


def test_failed_compile_repair_read_is_scoped_to_reported_file_and_line(tmp_path):
    scope=CodingAgent._failed_command_repair_scope({
        "exit_code":1,"output":'  File "controller.py", line 42\nSyntaxError: invalid syntax'})
    assert scope=={"path":"controller.py","line":42}
    assert CodingAgent._is_local_repair_read("read_file",{
        "path":"controller.py","start_line":30,"end_line":60},scope) is True
    assert CodingAgent._is_local_repair_read("read_file",{
        "path":"other.py","start_line":30,"end_line":60},scope) is False
    assert CodingAgent._is_local_repair_read("read_file",{
        "path":"controller.py","start_line":1,"end_line":200},scope) is False


def test_restore_previous_executed_controller_is_ledger_gated(tmp_path):
    root=tmp_path/"run";workspace=TaskWorkspace(root/"workspace")
    workspace.write_file("controller.py","def run(robot):\n    broken =\n")
    ignored=root/"iterations"/"iteration_002";ignored.mkdir(parents=True)
    (ignored/"controller.py").write_text("def run(robot):\n    return 'not executed'\n")
    executed=root/"iterations"/"iteration_001";executed.mkdir(parents=True)
    source="def run(robot):\n    return {'sensor_failure': True}\n"
    (executed/"controller.py").write_text(source)
    (executed/"robot_execution.json").write_text("{}")
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(root/"tools",workspace.root),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=root/"iterations"/"iteration_003")
    receipt=surface.registry().invoke("restore_previous_executed_controller",{})
    assert workspace.read_file("controller.py")["content"]==source.rstrip("\n")
    assert receipt["source_artifact_ref"]=="iterations/iteration_001/controller.py"
    assert receipt["controller_semantic_progress"] is True


def test_coding_agent_does_not_cache_deferred_read_delivery(tmp_path):
    registry=FunctionRegistry();read_calls=[]
    registry.add("bulk_first","bulk",{"type":"object"},
                 lambda **_: {"content":"f"*8000})
    registry.add("read_evidence","read",{"type":"object"},
                 lambda **_:read_calls.append(True) or {
                     "path":"evidence.json","content":"r"*7000},
                 evidence_policy="read_once")
    class RetryDeferredModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return {"content":"","tool_calls":[
                {"id":"bulk","name":"bulk_first","arguments":"{}"},
                {"id":"read1","name":"read_evidence","arguments":"{}"}]}
            if self.calls==2:return _call(2,"read_evidence",{})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=RetryDeferredModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",per_turn_tool_characters=10000).run("inspect")
    assert result["completed"] is True
    assert len(read_calls)==2
    reads=[row for row in result["tool_results"] if row["name"]=="read_evidence"]
    assert reads[0]["delivery_deferred"] is True
    assert reads[1].get("duplicate_read_suppressed") is not True


def test_coding_agent_allows_two_image_deliveries_and_resets_after_new_evidence(tmp_path):
    registry=FunctionRegistry();image_calls=[];mutations=[]
    registry.add("view_image","view",{"type":"object"},
                 lambda **_:image_calls.append(True) or {"_embodied_codex_image":{
                     "path":"frame.png","mime_type":"image/png","data_base64":"YWJj"}},
                 evidence_policy="image_twice")
    registry.add("new_evidence","mutate",{"type":"object"},
                 lambda **_:mutations.append(True) or {"created":True},
                 evidence_policy="invalidates_reads")
    class ImageLoopModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls<=3:return _call(self.calls,"view_image",{})
            if self.calls==4:return _call(4,"new_evidence",{})
            if self.calls==5:return _call(5,"view_image",{})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=ImageLoopModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl").run("inspect")
    assert result["completed"] is True
    assert len(image_calls)==3 and len(mutations)==1
    image_results=[row for row in result["tool_results"] if row["name"]=="view_image"]
    assert image_results[2]["duplicate_read_suppressed"] is True
    assert image_results[3].get("duplicate_read_suppressed") is not True


def test_coding_agent_pauses_unbounded_evidence_until_engineering_progress(tmp_path):
    registry=FunctionRegistry();reads=[];writes=[]
    registry.add("read_evidence","read",{"type":"object"},
                 lambda **arguments:reads.append(arguments) or {"value":arguments["page"]},
                 evidence_policy="read_once")
    registry.add("write_diagnosis","write",{"type":"object"},
                 lambda **arguments:writes.append(arguments) or {"written":True},
                 evidence_policy="invalidates_reads",evidence_progress=True)
    class BudgetedModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls<=7:return _call(self.calls,"read_evidence",{"page":self.calls})
            if self.calls==8:return _call(8,"write_diagnosis",{"hypothesis":"contact"})
            if self.calls==9:return _call(9,"read_evidence",{"page":7})
            return {"content":"implemented","tool_calls":[]}
    result=CodingAgent(model=BudgetedModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",max_evidence_deliveries=6).run("diagnose")
    assert result["completed"] is True
    assert [row["page"] for row in reads]==[1,2,3,4,5,6,7]
    assert writes==[{"hypothesis":"contact"}]
    read_results=[row for row in result["tool_results"] if row["name"]=="read_evidence"]
    assert read_results[6]["evidence_acquisition_paused"] is True
    assert read_results[7].get("evidence_acquisition_paused") is not True


def test_coding_agent_ends_pass_when_evidence_pause_is_ignored(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("read_evidence","read",{"type":"object"},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="read_once")
    class IgnoringModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            return _call(self.calls,"read_evidence",{"page":self.calls})
    model=IgnoringModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",max_evidence_deliveries=6,
        post_evidence_pause_max_turns=4).run("diagnose")
    assert result["completed"] is False
    assert result["error"]=="engineering action deadline after evidence pause"
    assert model.calls==11
    assert len(reads)==6


def test_coding_agent_applies_evidence_budget_to_large_terminal_output_and_notes(tmp_path):
    registry=FunctionRegistry();reads=[];commands=[];writes=[]
    registry.add("read_evidence","read",{"type":"object"},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="read_once")
    registry.add("command","command",{"type":"object"},
                 lambda **arguments:commands.append(arguments) or {
                     "stdout":"source"*1000 if arguments["kind"]=="cat" else "",
                     "exit_code":0},evidence_policy="budgeted_output")
    registry.add("write","write",{"type":"object"},
                 lambda **arguments:writes.append(arguments) or {"written":arguments["path"]},
                 evidence_policy="invalidates_reads",
                 evidence_progress=lambda arguments:arguments["path"].endswith(".py"))
    class TerminalModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls<=6:return _call(self.calls,"read_evidence",{"page":self.calls})
            if self.calls==7:return _call(7,"command",{"kind":"cat"})
            if self.calls==8:return _call(8,"command",{"kind":"compile"})
            if self.calls==9:return _call(9,"write",{"path":"notes.md"})
            if self.calls==10:return _call(10,"read_evidence",{"page":7})
            if self.calls==11:return _call(11,"write",{"path":"controller.py"})
            if self.calls==12:return _call(12,"read_evidence",{"page":7})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=TerminalModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",max_evidence_deliveries=6).run("diagnose")
    assert result["completed"] is True
    assert commands==[{"kind":"cat"},{"kind":"compile"}]
    command_results=[row for row in result["tool_results"] if row["name"]=="command"]
    assert command_results[0]["evidence_acquisition_paused"] is True
    assert command_results[1]["result"]["exit_code"]==0
    assert writes==[{"path":"notes.md"},{"path":"controller.py"}]
    assert [row["page"] for row in reads]==[1,2,3,4,5,6,7]
    reads_result=[row for row in result["tool_results"] if row["name"]=="read_evidence"]
    assert reads_result[6]["evidence_acquisition_paused"] is True
    assert reads_result[7].get("evidence_acquisition_paused") is not True


def test_coding_agent_uses_tighter_evidence_budget_after_robot_episode(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("run_robot_controller","robot",{"type":"object"},
                 lambda **_: {"sensor_success_candidate":False},
                 evidence_policy="invalidates_reads",invalidates_evidence_groups=("run",))
    registry.add("read_evidence","read",{"type":"object"},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="read_once",evidence_group="run")
    class PostRobotModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"run_robot_controller",{})
            if self.calls<=5:return _call(self.calls,"read_evidence",{"page":self.calls})
            return {"content":"diagnosed","tool_calls":[]}
    result=CodingAgent(model=PostRobotModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",max_evidence_deliveries=18,
        post_robot_evidence_deliveries=3).run("diagnose")
    assert result["completed"] is True
    assert reads==[{"page":2},{"page":3},{"page":4}]
    read_results=[row for row in result["tool_results"] if row["name"]=="read_evidence"]
    assert read_results[-1]["evidence_acquisition_paused"] is True
    assert read_results[-1]["delivery_limit"]==3


def test_acquisition_reads_remain_available_after_controller_mutation(tmp_path):
    registry=FunctionRegistry();calls=[]
    registry.add("edit_controller","edit",{"type":"object"},
                 lambda **_:calls.append("edit") or {"changed":True},
                 evidence_policy="invalidates_reads",execution_progress=True)
    registry.add("search_public_capability","search",{"type":"object"},
                 lambda **_:calls.append("search") or {"results":["planner"]},
                 evidence_policy="read_once",evidence_group="web",
                 post_mutation_read_allowed=True)
    class MutationThenSearchModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"edit_controller",{})
            if self.calls==2:return _call(2,"search_public_capability",{})
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=MutationThenSearchModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl",
        post_mutation_read_deliveries=0).run("acquire")
    assert result["completed"] is True
    assert calls==["edit","search"]
    search_result=next(row for row in result["tool_results"]
                       if row["name"]=="search_public_capability")
    assert search_result.get("evidence_acquisition_paused") is not True


def test_coding_agent_suppresses_paraphrased_asset_search_with_same_results(tmp_path):
    registry=FunctionRegistry();queries=[]
    def search_assets(**arguments):
        queries.append(arguments["query"])
        return {"query":arguments["query"],
            "tools":[{"tool_id":"selector:v001","retrieval_score":1.0}],
            "gaps":[{"gap_id":"source_grounding:v002","retrieval_score":0.5}]}
    registry.add("search_assets","search",{"type":"object"},search_assets,
                 evidence_policy="budgeted_output")
    class SearchModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"search_assets",{"query":"bowl on support"})
            if self.calls==2:return _call(2,"search_assets",{"query":"object above container"})
            return {"content":"act on the delivered assets","tool_calls":[]}
    result=CodingAgent(model=SearchModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl").run("acquire capability")
    searches=[row for row in result["tool_results"] if row["name"]=="search_assets"]
    assert queries==["bowl on support","object above container"]
    assert searches[0].get("duplicate_read_suppressed") is not True
    assert searches[1]["duplicate_read_suppressed"] is True
    assert searches[1]["semantic_duplicate_suppressed"] is True
    assert searches[1]["returned_asset_ids"]==[
        ("gaps","source_grounding:v002"),("tools","selector:v001")]


def test_coding_agent_bounds_delay_after_executable_mutation(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("write_controller","write executable",{"type":"object"},
                 lambda **_: {"written":True},evidence_policy="invalidates_reads",
                 evidence_progress=True,execution_progress=True)
    registry.add("read_more","read",{"type":"object","properties":{
                     "page":{"type":"integer"}},"required":["page"]},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="read_once")
    class DelayingModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"write_controller",{})
            return _call(self.calls,"read_more",{"page":self.calls})
    model=DelayingModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",post_mutation_max_turns=4).run("execute")
    assert result["completed"] is False
    assert result["error"]=="controller execution deadline after executable mutation"
    # The turn deadline remains the final guard, while the tighter evidence
    # gate prevents more than two additional reads after executable mutation.
    assert model.calls==5 and len(reads)==2


def test_coding_agent_bounds_delay_after_unchanged_replay_rejection(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("run_robot_controller","robot",{"type":"object"},
        lambda **_:(_ for _ in ()).throw(RuntimeError(
            "unchanged_controller_after_failed_episode: modify executable behavior")))
    registry.add("read_more","read",{"type":"object"},
        lambda **_:reads.append(True) or {"page":len(reads)},
        evidence_policy="read_once")
    class DelayingModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"run_robot_controller",{})
            return _call(self.calls,"read_more",{"page":self.calls})
    model=DelayingModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",post_rejection_max_turns=4).run("execute")
    assert result["completed"] is False
    assert result["error"]=="controller mutation deadline after unchanged replay rejection"
    assert model.calls==5 and len(reads)==4


def test_executable_mutation_clears_unchanged_replay_deadline(tmp_path):
    registry=FunctionRegistry();runs=[]
    def run_controller(**unused):
        runs.append(True)
        if len(runs)==1:raise RuntimeError(
            "unchanged_controller_after_failed_episode: modify executable behavior")
        return {"sensor_success_candidate":True}
    registry.add("run_robot_controller","robot",{"type":"object"},run_controller)
    registry.add("write_controller","write",{"type":"object"},lambda **_:{"written":True},
        evidence_policy="invalidates_reads",evidence_progress=True,execution_progress=True)
    class RepairingModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return _call(1,"run_robot_controller",{})
            if self.calls==2:return _call(2,"write_controller",{})
            return _call(3,"run_robot_controller",{})
    result=CodingAgent(model=RepairingModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",post_rejection_max_turns=4).run("execute")
    assert result["completed"] is True and len(runs)==2


def test_coding_agent_bounds_delay_for_pending_executable_after_resume(tmp_path):
    registry=FunctionRegistry();reads=[]
    registry.add("read_more","read",{"type":"object","properties":{
                     "page":{"type":"integer"}},"required":["page"]},
                 lambda **arguments:reads.append(arguments) or arguments,
                 evidence_policy="read_once")
    class DelayingModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1;return _call(self.calls,"read_more",{"page":self.calls})
    model=DelayingModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",post_mutation_max_turns=4,
        executable_pending=True).run("resume and execute")
    assert result["completed"] is False
    assert result["error"]=="controller execution deadline after executable mutation"
    assert model.calls==4 and len(reads)==2


def test_coding_agent_stops_immediately_after_sensor_success(tmp_path):
    registry=FunctionRegistry()
    registry.add("run_robot_controller","robot",{"type":"object"},
                 lambda **_: {"sensor_success_candidate":True},
                 evidence_policy="invalidates_reads")
    class OneShotModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls>1:raise AssertionError("successful robot run must end the coding pass")
            return _call(1,"run_robot_controller",{})
    model=OneShotModel()
    result=CodingAgent(model=model,registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl").run("execute")
    assert result["completed"] is True and model.calls==1
    assert result["final_text"].startswith("sensor success")


def test_coding_agent_compacts_consumed_images_and_large_tool_pages(tmp_path):
    import json
    registry=FunctionRegistry()
    registry.add("bulk","bulk",{"type":"object"},lambda **_: {
        "path":"/run/log.json","content":"x"*10000,"next_start_line":401})
    registry.add("image","image",{"type":"object"},lambda **_: {
        "_embodied_codex_image":{"path":"frame.png","mime_type":"image/png",
                                  "data_base64":"YWJj"}})
    class MemoryModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return {"content":"", "tool_calls":[
                {"id":"a","name":"bulk","arguments":"{}"},
                {"id":"b","name":"image","arguments":"{}"}]}
            if self.calls==2:
                assert any(isinstance(item.get("content"),list) for item in messages)
                return _call(2,"bulk",{})
            serialized=json.dumps(messages)
            assert "data:image/png;base64" not in serialized
            assert "working_memory_compacted" in serialized
            assert len(serialized)<20000
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=MemoryModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl").run("inspect")
    assert result["completed"] is True


def test_consumed_rollout_compaction_keeps_authoritative_sensor_outcome_capsule():
    payload={"ok":True,"result":{
        "completed":True,"error":None,
        "controller_result":{"sensor_failure":True,"reason":"rim contact failed",
            "detail":{"reached":False,"final_position_error_m":0.023}},
        "rpc_evidence":[{"blob":"x"*1000} for _ in range(20)],
        "rpc_event_count":29,"sensor_success_candidate":False,
        "sensor_report":{"independent_task_outcome":{"verified":False,
            "reason":"bowl remains in drawer"},"outcome_observations":{
                "before":{"rgb_path":"before.png"},
                "after":{"rgb_path":"after.png"}}},
        "full_execution_artifact":"robot_execution.json",
        "execution_artifact_ref":"latest_robot_execution"}}
    messages=[{"role":"tool","tool_call_id":"run",
               "content":json.dumps(payload)}]

    CodingAgent._compact_consumed_messages(messages,tool_character_limit=1000)

    receipt=json.loads(messages[0]["content"])
    assert receipt["working_memory_compacted"] is True
    assert receipt["authoritative_execution_capsule"] is True
    assert receipt["controller_result"]["reason"]=="rim contact failed"
    assert receipt["independent_task_outcome"]["verified"] is False
    assert receipt["outcome_observations"]["after"]["rgb_path"]=="after.png"
    assert receipt["execution_artifact_ref"]=="latest_robot_execution"
    assert "rpc_evidence" not in receipt


def test_coding_agent_keeps_bounded_python_source_as_working_memory(tmp_path):
    import json
    registry=FunctionRegistry()
    source="def run(robot):\n"+("    robot.record({'phase':'inspect'})\n"*260)
    registry.add("source","source",{"type":"object"},lambda **_: {
        "path":"/run/workspace/controller.py","start_line":1,"end_line":261,
        "content":source})
    registry.add("bulk","bulk",{"type":"object"},lambda **_: {
        "path":"/run/robot_execution.json","start_line":1,"end_line":20,
        "content":"x"*10_000})
    class SourceMemoryModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return {"content":"","tool_calls":[
                {"id":"source","name":"source","arguments":"{}"},
                {"id":"bulk","name":"bulk","arguments":"{}"}]}
            if self.calls==2:return _call(2,"bulk",{})
            serialized=json.dumps(messages)
            assert "def run(robot)" in serialized
            assert "working_memory_compacted" in serialized
            # The newest page is intentionally delivered once; the consumed
            # copy from the prior turn has already become a receipt.
            assert serialized.count("x"*100)<=100
            assert len(serialized)<45_000
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=SourceMemoryModel(),registry=registry,
        system_prompt="test",trace_path=tmp_path/"trace.jsonl").run("inspect")
    assert result["completed"] is True


def test_coding_agent_bounds_parallel_bulk_evidence_in_one_turn(tmp_path):
    import json
    registry=FunctionRegistry()
    registry.add("bulk","bulk",{"type":"object"},lambda **_: {
        "path":"/run/evidence.json","content":"x"*50000,"next_start_line":201})
    class BudgetModel:
        calls=0
        def decide(self,*,messages,tools):
            self.calls+=1
            if self.calls==1:return {"content":"","tool_calls":[
                {"id":str(i),"name":"bulk","arguments":"{}"} for i in range(3)]}
            serialized=json.dumps(messages)
            assert len(serialized)<70000
            assert serialized.count("delivery_deferred")>=2
            return {"content":"done","tool_calls":[]}
    result=CodingAgent(model=BudgetModel(),registry=registry,system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",per_turn_tool_characters=60000).run("inspect")
    assert result["completed"] is True


def test_openai_model_has_true_wall_clock_stream_deadline():
    import time
    from types import SimpleNamespace
    from embodied_codex.model import ModelResponseTimeout,OpenAIModel

    class NeverEndingStream:
        def __iter__(self):
            while True:
                time.sleep(1)
                yield SimpleNamespace(choices=[])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **unused:NeverEndingStream())))
    model=OpenAIModel.__new__(OpenAIModel)
    model.client=client;model.model="test";model.reasoning_effort="high"
    model.max_tokens=10;model.total_response_timeout=0.05
    started=time.monotonic()
    with pytest.raises(ModelResponseTimeout,match="exceeded"):
        model.decide(messages=[],tools=[])
    assert time.monotonic()-started<0.5


def test_coding_agent_limits_true_deadline_failures_to_one_retry(tmp_path):
    class DeadlineModel:
        calls=0
        def decide(self,**unused):
            self.calls+=1
            raise TimeoutError("model response exceeded 120 seconds")
    model=DeadlineModel()
    result=CodingAgent(model=model,registry=FunctionRegistry(),system_prompt="test",
        trace_path=tmp_path/"trace.jsonl",model_attempts=3).run("task")
    assert result["completed"] is False
    assert model.calls==2


def test_bing_search_result_parser():
    html='''<li class="b_algo"><h2><a href="https://github.com/example/grasp">A <strong>Grasp</strong> Tool</a></h2><div><p>Point cloud grasp generation.</p></div></li>'''
    assert _bing_results(html,5)==[{"title":"A Grasp Tool",
        "url":"https://github.com/example/grasp",
        "snippet":"Point cloud grasp generation."}]


def test_web_search_degrades_to_structured_provider_errors(monkeypatch):
    from urllib.error import URLError
    import embodied_codex.web as web
    def unavailable(*unused_args,**unused_kwargs):
        raise URLError("temporary provider outage")
    monkeypatch.setattr(web,"urlopen",unavailable)
    result=web.search_web("collision aware robot planning",limit=3)
    assert result["results"]==[]
    assert result["provider"]=="unavailable"
    assert {row["provider"] for row in result["provider_errors"]}=={
        "duckduckgo","bing"}


def test_web_search_bounds_untrusted_provider_snippets(monkeypatch):
    from urllib.error import URLError
    import embodied_codex.web as web
    monkeypatch.setattr(web,"_github_results",lambda query,limit:[{
        "title":"public/grasp-planner","url":"https://github.com/public/grasp-planner",
        "snippet":"untrusted "+("payload "*2000),"source":"github"}])
    monkeypatch.setattr(web,"urlopen",lambda *args,**kwargs:(_ for _ in ()).throw(
        URLError("general search unavailable")))
    result=web.search_web("robot grasp planner",limit=3)
    candidate=result["results"][0]
    assert len(candidate["snippet"])<=603
    assert candidate["snippet_truncated"] is True
    assert candidate["url"]=="https://github.com/public/grasp-planner"


def test_iteration_budget_counts_robot_episodes_not_coding_only_passes(tmp_path):
    class DelayedRunner:
        step=0
        def decide(self,*,messages,tools):
            import json
            instruction=json.loads(messages[1]["content"])
            if instruction["coding_pass"]==1:return {"content":"edited only","tool_calls":[]}
            assert instruction["prior_coding_pass_handoff"]=={
                "completed":True,"error":None,"recent_tool_results":[]}
            program='''def run(robot):
    robot.act({"target_x": 1})
    proof = robot.verify("goal", {})
    return {"status": "sensor_success", "proof": proof}
'''
            sequence=[("write_file",{"path":"controller.py","content":program}),
                      ("run_robot_controller",{"path":"controller.py"})]
            if self.step>=len(sequence):return {"content":"done","tool_calls":[]}
            name,args=sequence[self.step];self.step+=1
            return _call(100+self.step,name,args)
    engine=EvolutionEngine(root=tmp_path/"run",model=DelayedRunner(),
        deployment_factory=ClosedLoopDeployment,python=PYTHON)
    state=engine.run(task="move the cube",skill_name="delayed_runner_skill",max_iterations=1)
    assert state["status"]=="sensor_success"
    assert state["iterations"][0]["robot_episode"]==1
    assert state["iterations"][0]["coding_passes"]==2


def test_persisted_plan_is_handed_to_correction_pass_and_executed(tmp_path):
    class PlannedCorrectionModel:
        steps={1:0,2:0}
        def decide(self,*,messages,tools):
            instruction=json.loads(messages[1]["content"])
            coding_pass=instruction["coding_pass"]
            step=self.steps[coding_pass];self.steps[coding_pass]+=1
            if coding_pass==1:
                if step==0:
                    return _call(1,"write_file",{
                        "path":"next_experiment.md",
                        "content":"Write the closed-loop controller and run it."})
                return {"content":"plan persisted for correction","tool_calls":[]}
            handoff=instruction["prior_coding_pass_handoff"]
            assert handoff["persisted_action_artifacts"]==["next_experiment.md"]
            assert "engineering TODO" in handoff["continuation_rule"]
            program='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
            sequence=[("read_file",{"path":"next_experiment.md"}),
                      ("write_file",{"path":"controller.py","content":program}),
                      ("run_robot_controller",{"path":"controller.py"})]
            if step>=len(sequence):return {"content":"done","tool_calls":[]}
            name,args=sequence[step];return _call(10+step,name,args)
    state=EvolutionEngine(root=tmp_path/"run",model=PlannedCorrectionModel(),
        deployment_factory=ClosedLoopDeployment,python=PYTHON).run(
            task="move the cube",skill_name="planned_correction",max_iterations=1)
    assert state["status"]=="sensor_success"
    assert state["iterations"][0]["coding_passes"]==2


def test_complex_controller_can_continue_across_more_than_three_bounded_passes(tmp_path):
    class MultiPassRepairModel:
        step=0
        def decide(self,*,messages,tools):
            instruction=json.loads(messages[1]["content"])
            coding_pass=instruction["coding_pass"]
            if coding_pass<5:
                return {"content":f"bounded repair pass {coding_pass}","tool_calls":[]}
            program='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
            sequence=[("write_file",{"path":"controller.py","content":program}),
                      ("run_robot_controller",{"path":"controller.py"})]
            if self.step>=len(sequence):return {"content":"done","tool_calls":[]}
            name,args=sequence[self.step];self.step+=1
            return _call(50+self.step,name,args)
    state=EvolutionEngine(root=tmp_path/"run",model=MultiPassRepairModel(),
        deployment_factory=ClosedLoopDeployment,python=PYTHON,
        max_coding_passes=6).run(
            task="move the cube",skill_name="multi_pass_repair",max_iterations=1)
    assert state["status"]=="sensor_success"
    assert state["iterations"][0]["coding_passes"]==5


def test_zero_action_transient_tool_outage_does_not_consume_episode_budget(tmp_path):
    class FlakyDeployment(FakeDeployment):
        def __init__(self,outage):super().__init__();self.outage=outage
        def dispatch(self,method,arguments):
            if method=="use" and self.outage:
                return {"tool_id":arguments["tool_id"],"result":{"tool_error":{
                    "type":"APIConnectionError","message":"temporary connection error"}}}
            return super().dispatch(method,arguments)
        def sensor_report(self,execution):
            return {"sensor_verification_passed":self.x==1,
                    "benchmark_signal_exposed":False}
    deployments=[]
    def factory():
        value=FlakyDeployment(outage=not deployments);deployments.append(value);return value
    class RetryAfterOutageModel:
        steps={1:0,2:0}
        def decide(self,*,messages,tools):
            instruction=json.loads(messages[1]["content"]);iteration=instruction["iteration"]
            step=self.steps[iteration];self.steps[iteration]+=1
            outage='''def run(robot):
    failure = robot.use("flaky:v1", {})
    return {"sensor_failure": True, "detail": failure}
'''
            success='''def run(robot):
    robot.act({"target_x": 1})
    return robot.verify("goal", {})
'''
            sequence=[("write_file",{"path":"controller.py",
                                      "content":outage if iteration==1 else success}),
                      ("run_robot_controller",{"path":"controller.py"})]
            if step>=len(sequence):return {"content":"done","tool_calls":[]}
            name,args=sequence[step];return _call(iteration*10+step,name,args)
    state=EvolutionEngine(root=tmp_path/"run",model=RetryAfterOutageModel(),
        deployment_factory=factory,python=PYTHON).run(
            task="move cube",skill_name="transient_retry",max_iterations=1)
    assert state["status"]=="sensor_success"
    assert len(state["iterations"])==2
    assert state["iterations"][0]["transient_infrastructure_failure"]["robot_actions"]==0
    assert state["iterations"][1]["evidence"]["sensor_success_candidate"] is True


def test_libero_pose_reference_and_six_dof_action_contract():
    """A Tool pose remains provenance-bound and drives both OSC components."""
    from types import SimpleNamespace
    import numpy as np
    from robosuite.utils.transform_utils import mat2quat
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.references={};deployment.trace=[];deployment.step=0
    deployment.episode=SimpleNamespace(horizon=200)
    target_rotation=np.diag([1.0,-1.0,-1.0])
    wrapped=deployment._references("grasp_tool:v001",{
        "world_xyz":[0.10,-0.05,1.0],
        "world_bounds_10_90":[[0.08,-0.07,.98],[.12,-0.03,1.02]],
        "eef_rotation_world":target_rotation.tolist(),
    })
    assert wrapped["pose_ref"]==wrapped["point_ref"]
    reference=deployment.references[wrapped["pose_ref"]]
    assert reference["tool_id"]=="grasp_tool:v001"
    assert reference["world_bounds_10_90"]==[[.08,-.07,.98],[.12,-.03,1.02]]

    deployment.obs={
        "robot0_eef_pos":np.array([0.0,0.0,1.0]),
        "robot0_eef_quat":np.array([0.0,0.0,0.0,1.0]),
        "robot0_gripper_qpos":np.array([0.02,-0.02]),
    }
    commands=[]
    target_quaternion=mat2quat(target_rotation)
    def simulated_step(command):
        commands.append(np.asarray(command,float).copy())
        deployment.obs["robot0_eef_pos"] += np.asarray(command[:3])*0.02
        # This fake checks that an orientation command was emitted; LIBERO's
        # real OSC controller owns the dynamics that follow it.
        if np.linalg.norm(command[3:6])>0:
            deployment.obs["robot0_eef_quat"]=target_quaternion.copy()
        deployment.step+=1
    deployment._sim_step=simulated_step
    result=deployment._act({"type":"move_to_pose","pose_ref":wrapped["pose_ref"],
        "offset":[0.0,0.0,0.02],"gripper":-1,"max_steps":40})
    assert result["reached"] is True
    assert result["final_position_error_m"]<=0.012
    assert result["final_orientation_error_rad"]<=0.08
    assert any(np.linalg.norm(command[3:6])>0 for command in commands)
    assert all(command[6]==-1 for command in commands)


def test_libero_pose_contract_rejects_invalid_rotation():
    from embodied_codex.deployments.libero import LiberoDeployment,LiberoDeploymentError
    deployment=LiberoDeployment.__new__(LiberoDeployment);deployment.references={}
    with pytest.raises(LiberoDeploymentError,match="orthonormal"):
        deployment._references("bad_tool:v001",{
            "world_xyz":[0.0,0.0,1.0],
            "eef_rotation_world":[[1,0,0],[0,1,0],[0,0,2]],
        })


def test_libero_sensor_report_exposes_canonical_before_after_observations(tmp_path):
    from embodied_codex.deployments.libero import LiberoDeployment
    deployment=LiberoDeployment.__new__(LiberoDeployment)
    before={"rgb_path":str(tmp_path/"before.png"),"rgb_sha256":"a"*64}
    after={"rgb_path":str(tmp_path/"after.png"),"rgb_sha256":"b"*64}
    deployment.outcome_verifier=lambda payload:{"verified":False,
        "reason":"task relation not visible","sensor_only":True}
    deployment._instruction="move the named object"
    deployment._outcome_before=before;deployment._outcome_after=None
    deployment._outcome_report=None;deployment.last_verify=True
    deployment.trace=[];deployment.step=7;deployment.artifact_dir=tmp_path
    deployment.episode=type("Episode",(),{"case_handle":"opaque-case"})()
    deployment._capture_outcome_rgb=lambda name:after
    deployment._proprio=lambda:{"eef":[0,0,0]}

    report=deployment.sensor_report({})

    assert report["outcome_observations"]=={"before":before,"after":after}
    assert report["independent_task_outcome"]["verified"] is False
    assert report["sensor_verification_passed"] is False


def test_libero_capability_outage_is_structured_not_controller_crash():
    from embodied_codex.deployments.libero import LiberoDeployment,LiberoDeploymentError
    class APIConnectionError(Exception):pass
    def unavailable(_payload):raise APIConnectionError("temporary connection loss")
    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.capabilities={"vlm:v001":unavailable};deployment.references={}
    deployment.capability_contracts={"vlm:v001":{
        "input_schema":{"type":"object"},"output_schema":{"type":"object"}}}
    deployment.trace=[];deployment.step=12
    receipt=deployment._use("vlm:v001",{"frame":"sensor"})
    assert receipt["result"]=={"ok":False,"tool_error":{
        "type":"APIConnectionError","message":"temporary connection loss"}}
    assert deployment.trace[-1]["tool_error"]["type"]=="APIConnectionError"
    with pytest.raises(LiberoDeploymentError,match="unregistered Tool"):
        deployment._use("missing:v001",{})


def test_zero_action_transient_tool_outage_is_infrastructure_not_task_failure():
    execution={"result":{"sensor_failure":True,"detail":{"tool_error":{
                   "type":"VLMRelationGroundingError",
                   "message":"VLM relation consensus exceeded 90 seconds"}}},
               "rpc_events":[{"method":"observe"},{"method":"use",
                   "arguments":{"tool_id":"vlm:v1"},"result":{"result":{
                       "tool_error":{"type":"VLMRelationGroundingError",
                       "message":"VLM relation consensus exceeded 90 seconds"}}}}]}
    result=transient_infrastructure_failure(execution)
    assert result["kind"]=="transient_tool_outage_before_action"
    assert result["tool_id"]=="vlm:v1" and result["robot_actions"]==0
    execution["rpc_events"].append({"method":"act","result":{"reached":True}})
    assert transient_infrastructure_failure(execution) is None


def test_post_action_task_verifier_timeout_is_transient_infrastructure_failure():
    execution={"rpc_events":[
        {"method":"act","result":{"reached":True}},
        {"method":"verify","result":{"verified":True}},
    ]}
    report={"controller_visual_verification_passed":True,
            "independent_task_outcome":{"verified":False,
                "error":"VLMTaskOutcomeError: VLM task-outcome consensus exceeded 90 seconds",
                "sensor_only":True}}
    result=transient_infrastructure_failure(execution,report)
    assert result["kind"]=="transient_post_action_sensor_verifier_outage"
    assert result["robot_actions"]==1
    assert result["retry"]=="same_controller_same_case"
    report["independent_task_outcome"]={"verified":False,
        "error":"VLMTaskOutcomeError: consensus did not reach decision quorum"}
    assert transient_infrastructure_failure(execution,report)["kind"]==(
        "transient_post_action_sensor_verifier_outage")
    report["independent_task_outcome"]={"verified":False,
        "source_relation_satisfied":True,"target_relation_satisfied":True,
        "contradiction":"none","consensus":{"rounds":3,"completed_rounds":2,
            "required":2,"true_votes":1,"false_votes":1}}
    inconclusive=transient_infrastructure_failure(execution,report)
    assert inconclusive["error_type"]=="VLMConsensusInconclusive"
    assert inconclusive["retry"]=="same_controller_same_case"
    report["independent_task_outcome"]={"verified":False,
        "reason":"the object remains at the source"}
    assert transient_infrastructure_failure(execution,report) is None


def test_historical_verifier_outage_chain_replays_first_uncontaminated_program():
    outage={"kind":"transient_post_action_sensor_verifier_outage"}
    records=[
        {"iteration":1},
        {"iteration":2,"transient_infrastructure_failure":outage},
        {"iteration":3},
        {"iteration":4,"transient_infrastructure_failure":outage},
        {"iteration":5,"transient_infrastructure_failure":outage},
    ]
    assert _post_action_transient_replay_source(records)["iteration"]==4
    records[-2].pop("transient_infrastructure_failure")
    assert _post_action_transient_replay_source(records)["iteration"]==5
    records[-1].pop("transient_infrastructure_failure")
    assert _post_action_transient_replay_source(records) is None


def test_post_action_verifier_outage_replays_same_controller_without_model(tmp_path):
    class VerifierDeployment(ClosedLoopDeployment):
        attempts=0
        def sensor_report(self,execution):
            type(self).attempts+=1
            if type(self).attempts==1:
                return {"sensor_verification_passed":False,
                    "controller_visual_verification_passed":True,
                    "independent_task_outcome":{"verified":False,
                        "error":"VLMTaskOutcomeError: consensus exceeded 90 seconds"},
                    "benchmark_signal_exposed":False}
            return {"sensor_verification_passed":True,
                    "controller_visual_verification_passed":True,
                    "independent_task_outcome":{"verified":True},
                    "benchmark_signal_exposed":False}

    class OneShotModel:
        calls=0
        iterations=[]
        def decide(self,*,messages,tools):
            self.calls+=1
            instruction=json.loads(messages[1]["content"])
            self.iterations.append(instruction["iteration"])
            if self.calls==1:
                return _call(1,"write_file",{"path":"controller.py","content":
                    "def run(robot):\n"
                    "    robot.act({'target_x': 1})\n"
                    "    return robot.verify('goal', {})\n"})
            if self.calls==2:
                return _call(2,"run_robot_controller",{"path":"controller.py"})
            if self.calls==3:
                return {"content":"rollout committed","tool_calls":[]}
            raise AssertionError("model must not run during infrastructure replay")

    model=OneShotModel()
    state=EvolutionEngine(root=tmp_path/"post-action-retry",model=model,
        deployment_factory=VerifierDeployment,python=PYTHON).run(
            task="move cube",skill_name="post_action_retry",max_iterations=1)
    assert state["status"]=="sensor_success"
    assert len(state["iterations"])==2
    first,replay=state["iterations"]
    assert first["transient_infrastructure_failure"]["robot_actions"]==1
    assert first["robot_episode"]==replay["robot_episode"]==1
    assert replay["coding_passes"]==0
    assert replay["transient_infrastructure_replay_without_model"] is True
    assert replay["infrastructure_replay_source_iteration"]==1
    assert model.calls==3
    assert model.iterations==[1,1,1]
    first_controller=(tmp_path/"post-action-retry"/"iterations"/
                      "iteration_001"/"controller.py").read_bytes()
    replay_controller=(tmp_path/"post-action-retry"/"iterations"/
                       "iteration_002"/"controller.py").read_bytes()
    assert first_controller==replay_controller


def test_libero_dynamically_binds_generated_tool_with_its_contract():
    from embodied_codex.deployments.libero import LiberoDeployment
    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.capabilities={};deployment.capability_contracts={}
    deployment.references={};deployment.trace=[];deployment.step=7
    schema={"type":"object","properties":{"value":{"type":"number"}},
            "required":["value"],"additionalProperties":False}
    deployment.register_capability("generated:v001",
        lambda payload:{"value":payload["value"]+1},
        {"input_schema":schema,"output_schema":schema})
    receipt=deployment._use("generated:v001",{"value":2})
    assert receipt=={"tool_id":"generated:v001","step":7,"result":{"value":3}}
    assert deployment.capability_contracts["generated:v001"]["input_schema"]==schema


def test_conformance_audits_deployment_owned_tool_validation_record(tmp_path):
    import json
    from embodied_codex.legacy.conformance import _tool_assets
    source=tmp_path/"adapter_tool.py";source.write_text("def deployed(payload): return payload\n")
    workspace=tmp_path/"workspace";workspace.mkdir()
    tools=tmp_path/"tools";library=CapabilityLibrary(tools,workspace,python=PYTHON)
    schema={"type":"object","properties":{},"additionalProperties":False}
    registered=library.register_deployment_tool(name="adapter_tool",implementation_path=str(source),
        description="adapter tool",input_schema=schema,output_schema=schema,
        provenance={"source_urls":["https://example.org/adapter-tool"],
            "trained_on_current_task":False,"privileged_state_used":False,
            "training_data_declaration":"No learned parameters.",
            "contamination_check":{"evaluated_benchmark":"test",
                "method":"source inspection","result":"not_applicable_source_code"}})
    surface=EngineeringSurface(workspace=TaskWorkspace(workspace),capabilities=library,
        runtime=object(),deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001")
    summary=surface.inspect_tool(registered["tool_id"])["manifest"]["test_summary"]
    assert summary=={"batches":0,"cases":0,"all_passed":None,
                     "status_authority":"deployment_adapter_binding"}
    run=tmp_path/"run";run.mkdir(exist_ok=True)
    (run/"harness_configuration.json").write_text(json.dumps({
        "capability_root":str(tools.resolve())})+"\n")
    ok,errors,counts,manuals,deps,contracts=_tool_assets(run)
    assert ok and not errors and manuals and deps and contracts
    assert counts["tested"]==1


def test_deployment_tool_sibling_dependency_is_frozen_loaded_and_hash_verified(tmp_path):
    import json
    from embodied_codex.examples.evaluate_libero_skill import (
        _controller_capability_view, _load_class, _relative_module_paths,
    )

    source=tmp_path/"deployed.py"
    dependency=tmp_path/"_support.py"
    source.write_text(
        "from ._support import offset\n"
        "class Deployed:\n"
        "    def apply(self, value): return offset(value)\n")
    dependency.write_text("def offset(value): return value + 3\n")
    workspace=tmp_path/"workspace";workspace.mkdir()
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    schema={"type":"object","properties":{},"additionalProperties":False}
    provenance={"source_urls":["https://example.org/deployed"],
        "trained_on_current_task":False,"privileged_state_used":False,
        "training_data_declaration":"No learned parameters.",
        "contamination_check":{"evaluated_benchmark":"test",
            "method":"source inspection","result":"not_applicable_source_code"}}
    with pytest.raises(AssetError,match="dependency_paths must exactly match"):
        library.register_deployment_tool(name="missing_dependency",
            implementation_path=str(source),description="deployed",input_schema=schema,
            output_schema=schema,provenance=provenance)
    tool_id=library.register_deployment_tool(name="complete_dependency",
        implementation_path=str(source),description="deployed",input_schema=schema,
        output_schema=schema,provenance=provenance,
        dependency_paths={"_support":dependency})["tool_id"]
    tool_manifest=library.inspect(tool_id)["manifest"]
    assert tool_manifest["relative_modules"]["_support"]["path"]=="_support.py"
    controller=workspace/"controller.py"
    controller.write_text("def run(robot): return {'complete': True}\n")
    skills=SkillLibrary(tmp_path/"skills")
    frozen=skills.freeze(name="dependency_skill",task="test",
        controller=controller,evidence={"sensor_only":True},tool_ids=[tool_id],tools=library)
    _controller,_manifest,frozen_tools=inspect_skill(frozen["path"])
    item=frozen_tools[tool_id]
    loaded=_load_class(item["folder"]/"tool.py","Deployed",
                       relative_modules=_relative_module_paths(item))
    assert loaded().apply(4)==7
    assert not (item["folder"]/"__pycache__").exists()
    dependency.write_text("def offset(value): return value + 4\n")
    replacement_id=library.register_deployment_tool(name="complete_dependency",
        implementation_path=str(source),description="deployed",input_schema=schema,
        output_schema=schema,provenance=provenance,
        dependency_paths={"_support":dependency})["tool_id"]
    assert replacement_id!=tool_id
    migrated=skills.repackage(skill_dir=frozen["path"],tools=library,
        tool_replacements={tool_id:replacement_id},reason="Exercise audited packaging migration")
    migrated_manifest=json.loads((Path(migrated["path"])/"manifest.json").read_text())
    assert migrated_manifest["supersedes"]=="dependency_skill:v001"
    assert migrated_manifest["packaging_migration"]["controller_sha256_unchanged"] is True
    assert migrated_manifest["controller_tool_bindings"]=={tool_id:replacement_id}
    _migrated_controller,_migrated_manifest,migrated_tools=inspect_skill(migrated["path"])
    physical={replacement_id:object()}
    logical,contracts=_controller_capability_view(_migrated_manifest,migrated_tools,physical)
    assert set(logical)==set(contracts)=={tool_id}
    assert logical[tool_id] is physical[replacement_id]
    frozen_dependency=item["folder"]/"_support.py"
    frozen_dependency.write_text("def offset(value): return value + 30\n")
    with pytest.raises(RuntimeError,match="dependency hash mismatch"):
        inspect_skill(frozen["path"])


def test_tool_tests_use_numeric_tolerance_and_preserve_real_failures(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"identity.py").write_text("def run(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace)
    tolerant=library.register_tool(name="numeric_identity",source_path="identity.py",
        description="test",input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],
        trained_on_current_task=False)["tool_id"]
    result=library.test_tool(tolerant,[{"input":{"x":0.020000000000000018},
                                        "expected":{"x":0.02}}])
    assert result["status"]=="tested"


def test_tool_versions_deduplicate_and_retrieval_prefers_latest_tested(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir();source=workspace/"versioned.py"
    source.write_text("def run(payload): return {'value': payload['value']}\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    schema={"type":"object","properties":{"value":{"type":"number"}},
            "required":["value"],"additionalProperties":False}
    first=library.register_tool(name="versioned_filter",source_path="versioned.py",
        description="versioned filter",input_schema=schema,output_schema=schema,
        source_urls=["https://example.org/filter"],trained_on_current_task=False)
    library.test_tool(first["tool_id"],[{"input":{"value":1},"expected":{"value":1}}])
    duplicate=library.register_tool(name="versioned_filter",source_path="versioned.py",
        description="versioned filter",input_schema=schema,output_schema=schema,
        source_urls=["https://example.org/filter"],trained_on_current_task=False)
    assert duplicate["tool_id"]==first["tool_id"] and duplicate["duplicate_of"]==first["tool_id"]
    source.write_text("def run(payload): return {'value': payload['value'] + 1}\n")
    second=library.register_tool(name="versioned_filter",source_path="versioned.py",
        description="versioned filter",input_schema=schema,output_schema=schema,
        source_urls=["https://example.org/filter"],trained_on_current_task=False)
    assert library.inspect(second["tool_id"])["manifest"]["supersedes"]==first["tool_id"]
    before={row["tool_id"] for row in library.search("versioned filter",8)}
    assert before=={first["tool_id"],second["tool_id"]}
    library.test_tool(second["tool_id"],[{"input":{"value":1},"expected":{"value":2}}])
    after={row["tool_id"] for row in library.search("versioned filter",8)}
    assert after=={second["tool_id"]}


def test_generated_tool_runtime_isolated_and_only_sees_explicit_sensor_files(tmp_path):
    import os
    os.environ["APEX_API_KEY"]="must-not-enter-tool"
    workspace=tmp_path/"workspace";workspace.mkdir()
    evidence=tmp_path/"run"/"episodes"/"episode_001"/"sensors"/"frame.txt"
    evidence.parent.mkdir(parents=True);evidence.write_text("visible sensor")
    secret=tmp_path/"sealed_benchmark_secret.txt";secret.write_text("privileged")
    (workspace/"reader.py").write_text(f'''from pathlib import Path
import os
def run(payload):
    return {{"sensor":Path(payload["path"]).read_text(),
            "host_secret_visible":Path({str(secret)!r}).exists(),
            "host_api_key":os.environ.get("APEX_API_KEY")}}
''')
    library=CapabilityLibrary(tmp_path/"tools",workspace,
        python=PYTHON,allowed_input_roots=[tmp_path/"run"/"episodes"])
    tool_id=library.register_tool(name="isolated_reader",source_path="reader.py",
        description="read explicit evidence",input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],
        trained_on_current_task=False)["tool_id"]
    result=library.run(tool_id,{"path":str(evidence)})
    assert result=={"sensor":"visible sensor","host_secret_visible":False,
                    "host_api_key":None}
    with pytest.raises(Exception,match="outside the sensor-evidence roots"):
        library.run(tool_id,{"path":str(secret)})


def test_generated_tool_dependencies_are_locked_vendored_and_hash_verified(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"tool.py").write_text(
        "import acquired_math\ndef run(payload): return {'value': acquired_math.twice(payload['x'])}\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    with pytest.raises(Exception,match="vendored dependency lock"):
        library.register_tool(name="unlocked_external",source_path="tool.py",description="bad",
            input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],trained_on_current_task=False,
            dependency_spec={"mode":"stdlib"})
    vendor=workspace/"vendor"/"acquired_math";vendor.mkdir(parents=True)
    (vendor/"__init__.py").write_text("def twice(x): return 2*x\n")
    lock=workspace/"requirements.lock"
    lock.write_text("acquired-math==1.0 --hash=sha256:"+"a"*64+"\n")
    tool_id=library.register_tool(name="locked_external",source_path="tool.py",description="good",
        input_schema={},output_schema={},source_urls=["https://example.org/acquired-math"],
        trained_on_current_task=False,dependency_spec={"mode":"vendored",
            "requirements_lock_path":"requirements.lock","vendor_path":"vendor"})["tool_id"]
    assert library.run(tool_id,{"x":4})=={"value":8}
    frozen=library._path(tool_id)/"vendor"/"acquired_math"/"__init__.py"
    frozen.write_text("def twice(x): return 3*x\n")
    with pytest.raises(Exception,match="dependency bundle hash mismatch"):
        library.run(tool_id,{"x":4})


def test_checkpoint_backed_capability_package_is_hashed_tested_and_isolated(tmp_path):
    import hashlib
    workspace=tmp_path/"workspace";bundle=workspace/"visual_servo_model";bundle.mkdir(parents=True)
    (bundle/"driver.py").write_text(
        "def run(payload): return {'correction': 2 * payload['error']}\n")
    checkpoint=bundle/"weights.bin";checkpoint.write_bytes(b"public-task-disjoint-weights")
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    schema_in={"type":"object","properties":{"error":{"type":"number"}},
               "required":["error"],"additionalProperties":False}
    schema_out={"type":"object","properties":{"correction":{"type":"number"}},
                "required":["correction"],"additionalProperties":False}
    manual={"purpose":"Compute a visual-servo correction","when_to_use":["After image error"],
        "inputs":{"error":"sensor-derived scalar error"},
        "outputs":{"correction":"bounded correction"},"examples":[],
        "failure_modes":["Malformed sensor input"],"limitations":["Test fixture model"]}
    provenance={"models":["public visual servo model"],
        "model_card_urls":["https://example.org/model-card"],
        "checkpoint_sha256":{"weights.bin":digest},
        "training_data_declaration":"Public synthetic images; no LIBERO episodes.",
        "contamination_check":{"evaluated_benchmark":"LIBERO",
            "method":"model-card and dataset manifest inspection","result":"no_declared_overlap"}}
    registered=library.register_package(name="visual_servo_package",
        bundle_path="visual_servo_model",description="checkpoint-backed visual servo",
        input_schema=schema_in,output_schema=schema_out,
        source_urls=["https://example.org/visual-servo"],trained_on_current_task=False,
        manual=manual,provenance=provenance,package_spec={"kind":"model",
            "entrypoint":"driver.py","accelerator":"cpu","network":False,
            "timeout_seconds":30})
    tool_id=registered["tool_id"]
    assert library.test_tool(tool_id,[{"input":{"error":1.5},
        "expected":{"correction":3.0}}])["status"]=="tested"
    manifest=library.inspect(tool_id)["manifest"]
    assert manifest["asset_kind"]=="model"
    assert manifest["runtime_spec"]["protocol"]=="isolated-json-worker-v2"
    assert manifest["runtime_spec"]["transport"]=="json-stdio"
    assert manifest["runtime_spec"]["lifecycle"]=="per-invocation"
    assert library.run(tool_id,{"error":2})=={"correction":4}
    controller=workspace/"controller.py"
    controller.write_text("def run(robot): return robot.use('"+tool_id+"', {'error': 1})\n")
    frozen=SkillLibrary(tmp_path/"skills").freeze(name="visual_servo_skill",task="align",
        controller=controller,evidence={"sensor_only":True},tool_ids=[tool_id],tools=library)
    _controller,skill_manifest,frozen_tools=inspect_skill(frozen["path"])
    assert skill_manifest["interface"]["tool_dependencies"]==[tool_id]
    assert frozen_tools[tool_id]["manifest"]["asset_kind"]=="model"
    frozen_manual=Path(frozen["path"])/"tools"/tool_id.replace(":","_")/"manual.json"
    frozen_manual.write_text("{}\n")
    with pytest.raises(RuntimeError,match="bundle hash mismatch"):
        inspect_skill(frozen["path"])
    (library._path(tool_id)/"bundle"/"weights.bin").write_bytes(b"tampered")
    with pytest.raises(AssetError,match="bundle hash mismatch"):
        library.run(tool_id,{"error":2})


def test_capability_package_rejects_false_checkpoint_provenance(tmp_path):
    workspace=tmp_path/"workspace";bundle=workspace/"model";bundle.mkdir(parents=True)
    (bundle/"driver.py").write_text("def run(payload): return {}\n")
    (bundle/"weights.bin").write_bytes(b"real")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    with pytest.raises(AssetError,match="checkpoint sha256 mismatch"):
        library.register_package(name="false_checkpoint_model",bundle_path="model",
            description="bad",input_schema={"type":"object"},output_schema={"type":"object"},
            source_urls=["https://example.org/model"],trained_on_current_task=False,
            manual={"purpose":"bad","when_to_use":[],"inputs":{},"outputs":{},
                "examples":[],"failure_modes":[],"limitations":[]},
            provenance={"models":["bad"],"model_card_urls":["https://example.org/card"],
                "checkpoint_sha256":{"weights.bin":"0"*64},
                "training_data_declaration":"public data",
                "contamination_check":{"evaluated_benchmark":"LIBERO",
                    "method":"manifest","result":"no_declared_overlap"}},
            package_spec={"kind":"model","entrypoint":"driver.py","accelerator":"cpu",
                "network":False,"timeout_seconds":30})


def test_capability_package_rejects_missing_run_entrypoint(tmp_path):
    workspace=tmp_path/"workspace";bundle=workspace/"planner";bundle.mkdir(parents=True)
    (bundle/"driver.py").write_text("def plan(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    with pytest.raises(AssetError,match=r"def run\(payload\)"):
        library.register_package(name="missing_run_planner",bundle_path="planner",
            description="planner",input_schema={"type":"object"},
            output_schema={"type":"object"},source_urls=["https://example.org/planner"],
            trained_on_current_task=False,
            manual={"purpose":"planner","when_to_use":[],"inputs":{},"outputs":{},
                "examples":[],"failure_modes":[],"limitations":[]},
            provenance={"training_data_declaration":"No learned parameters.",
                "contamination_check":{"evaluated_benchmark":"LIBERO",
                    "method":"source inspection","result":"not_applicable_source_code"}},
            package_spec={"kind":"planner","entrypoint":"driver.py","accelerator":"cpu",
                "network":False,"timeout_seconds":30})


def test_capability_package_requires_reproducible_external_dependencies(tmp_path):
    workspace=tmp_path/"workspace";bundle=workspace/"planner";bundle.mkdir(parents=True)
    (bundle/"driver.py").write_text("import jsonschema\ndef run(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    with pytest.raises(AssetError,match="pinned runtime_requirements"):
        library.register_package(name="unlocked_planner",bundle_path="planner",
            description="planner",input_schema={"type":"object"},output_schema={"type":"object"},
            source_urls=["https://example.org/planner"],trained_on_current_task=False,
            manual={"purpose":"planner","when_to_use":[],"inputs":{},"outputs":{},
                "examples":[],"failure_modes":[],"limitations":[]},
            provenance={"training_data_declaration":"No learned parameters.",
                "contamination_check":{"evaluated_benchmark":"LIBERO",
                    "method":"source inspection","result":"not_applicable_source_code"}},
            package_spec={"kind":"planner","entrypoint":"driver.py","accelerator":"cpu",
                "network":False,"timeout_seconds":30})


def test_capability_package_rejects_fake_ros_bridge_runtime(tmp_path):
    workspace=tmp_path/"workspace";bundle=workspace/"bridge";bundle.mkdir(parents=True)
    (bundle/"driver.py").write_text("def run(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    with pytest.raises(AssetError,match="package_spec.kind"):
        library.register_package(name="fake_ros_bridge",bundle_path="bridge",
            description="not really connected to ROS",input_schema={"type":"object"},
            output_schema={"type":"object"},source_urls=["https://www.ros.org/"],
            trained_on_current_task=False,
            manual={"purpose":"bridge","when_to_use":[],"inputs":{},"outputs":{},
                "examples":[],"failure_modes":[],"limitations":[]},
            provenance={"training_data_declaration":"No learned parameters.",
                "contamination_check":{"evaluated_benchmark":"LIBERO",
                    "method":"source inspection","result":"not_applicable_source_code"}},
            package_spec={"kind":"ros_bridge","entrypoint":"driver.py",
                "accelerator":"cpu","network":False,"timeout_seconds":30})


def test_tool_json_schemas_are_enforced_at_registration_test_and_runtime(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"tool.py").write_text("def run(payload): return {'value': payload['value']}\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    (workspace/"missing_entrypoint.py").write_text(
        "def normalize(payload): return payload\n")
    with pytest.raises(AssetError,match=r"def run\(payload\)"):
        library.register_tool(name="missing_entrypoint",
            source_path="missing_entrypoint.py",description="bad entrypoint",
            input_schema={"type":"object"},output_schema={"type":"object"},
            source_urls=["https://example.org/test-algorithm"],
            trained_on_current_task=False)
    with pytest.raises(Exception,match="descriptive mapping"):
        library.register_tool(name="descriptive_contract",source_path="tool.py",description="bad",
            input_schema={"value":"number"},output_schema={},source_urls=["https://example.org/test-algorithm"],
            trained_on_current_task=False)
    schema={"type":"object","properties":{"value":{"type":"number"}},
            "required":["value"],"additionalProperties":False}
    with pytest.raises(Exception,match="manual output fields"):
        library.register_tool(name="inconsistent_manual",source_path="tool.py",description="bad manual",
            input_schema=schema,output_schema=schema,source_urls=["https://example.org/test-algorithm"],trained_on_current_task=False,
            manual={"purpose":"bad","when_to_use":["test"],"inputs":{"value":"number"},
                "outputs":{"other":"number"},"examples":[],"failure_modes":[],"limitations":[]})
    tool_id=library.register_tool(name="validated_contract",source_path="tool.py",description="good",
        input_schema=schema,output_schema=schema,source_urls=["https://example.org/test-algorithm"],trained_on_current_task=False)["tool_id"]
    with pytest.raises(Exception,match="test input violates"):
        library.test_tool(tool_id,[{"input":{"value":"wrong"},"expected":{"value":"wrong"}}])
    failed=library.inspect(tool_id)["manifest"]
    assert failed["status"]=="test_failed"
    from embodied_codex.legacy.conformance import _tool_assets
    run=tmp_path/"run";run.mkdir()
    (run/"harness_configuration.json").write_text(__import__("json").dumps({
        "capability_root":str((tmp_path/"tools").resolve())}))
    asset_ok,errors,*_rest=_tool_assets(run)
    assert asset_ok is True and errors==[]
    assert failed["tests"][-1][-1]["passed"] is False
    assert "test input violates" in failed["tests"][-1][-1]["error"]
    with pytest.raises(Exception,match="input violates"):
        library.run(tool_id,{"value":"wrong"})


def test_tool_listing_is_compact_and_inspection_keeps_full_evidence(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"identity.py").write_text("def run(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace)
    tool_id=library.register_tool(name="identity",source_path="identity.py",
        description="identity",input_schema={"type":"object","properties":{"value":{"type":"number"}}},
        output_schema={"type":"object","properties":{"value":{"type":"number"}}},
        source_urls=["https://example.org/tool"],
        trained_on_current_task=False)["tool_id"]
    library.test_tool(tool_id,[{"input":{"value":1},"expected":{"value":1}}])
    listed=library.list_summaries()
    assert listed==[{"protocol":"embodied-codex-tool-v1","tool_id":tool_id,
        "name":"identity","version":1,"description":"identity",
        "input_schema":{"type":"object","properties":{"value":{"type":"number"}}},
        "output_schema":{"type":"object","properties":{"value":{"type":"number"}}},
        "status":"tested","trained_on_current_task":False,
        "privileged_state_used":False,
        "manual":{"purpose":"identity","when_to_use":["identity"],
                  "inputs":{"value":{"type":"number"}},
                  "outputs":{"value":{"type":"number"}},
                  "examples":[],"failure_modes":[
                      "May reject malformed input or return a documented structured error."],
                  "limitations":[]}}]
    assert "tests" not in listed[0] and "source_urls" not in listed[0]
    inspected=library.inspect(tool_id)
    assert inspected["manifest"]["tests"] and inspected["manifest"]["source_urls"]

    strict=library.register_tool(name="history_identity",source_path="identity.py",
        description="test",input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],
        trained_on_current_task=False)["tool_id"]
    assert library.test_tool(strict,[{"input":{"x":1},"expected":{"x":2}}])["status"]=="test_failed"
    # A later convenient subset cannot erase a genuine immutable-version failure.
    assert library.test_tool(strict,[{"input":{"x":1},"expected":{"x":1}}])["status"]=="test_failed"


def test_engineering_surface_hides_superseded_deployment_tool_versions(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    library=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root)
    family=library.root/"vision_adapter"
    for version in (1,2):
        path=family/f"v{version:03d}";path.mkdir(parents=True)
        source="def run(payload): return payload\n";(path/"tool.py").write_text(source)
        import hashlib,json
        (path/"manifest.json").write_text(json.dumps({
            "protocol":"embodied-codex-deployment-tool-v1",
            "tool_id":f"vision_adapter:v{version:03d}","name":"vision_adapter",
            "version":version,"description":"vision","input_schema":{},"output_schema":{},
            "source_sha256":hashlib.sha256(source.encode()).hexdigest(),"status":"tested",
            "trained_on_current_task":False,"privileged_state_used":False,
            "execution_owned_by_deployment":True}))
    surface=EngineeringSurface(workspace=workspace,capabilities=library,runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        active_deployment_tool_ids=["vision_adapter:v002"])
    assert [row["tool_id"] for row in surface.list_available_tools()]==[
        "vision_adapter:v002"]


def test_visual_attachment_requires_source_vacancy_eef_proximity_and_retained_width():
    import numpy as np
    from embodied_codex.capabilities import OpenVocabularyRGBD
    capability=OpenVocabularyRGBD.__new__(OpenVocabularyRGBD)
    detections=[{"world_xyz":[0.10,0.0,1.0]}]
    capability.detect=lambda payload:{"detections":{"bowl":list(detections)}}
    frame={"proprioception":{"robot0_eef_pos":[0.11,0.0,1.05],
                              "robot0_gripper_qpos":[0.01,-0.01]}}
    payload={"frame":frame,"object_query":"bowl","source_world_xyz":[0.0,0.0,1.0]}
    result=capability.verify_attachment(payload)
    assert result["verified"] is True and result["source_vacated"] is True
    frame["proprioception"]["robot0_gripper_qpos"]=[0.0005,-0.0005]
    result=capability.verify_attachment(payload)
    assert result["verified"] is False and result["retained_width"] is False
    frame["proprioception"]["robot0_gripper_qpos"]=[0.01,-0.01]
    detections[:]=[{"world_xyz":[0.001,0.0,1.0]}]
    result=capability.verify_attachment(payload)
    assert result["verified"] is False and result["source_vacated"] is False


def test_visual_support_overlap_handles_support_smaller_than_object():
    from embodied_codex.capabilities import OpenVocabularyRGBD

    capability = OpenVocabularyRGBD.__new__(OpenVocabularyRGBD)
    object_detection = {
        "world_xyz": [0.20, 0.10, 1.04],
        "world_bounds_10_90": [[0.15, 0.05, 1.02], [0.25, 0.15, 1.06]],
    }
    target_detection = {
        "world_xyz": [0.20, 0.10, 1.00],
        "world_bounds_10_90": [[0.165, 0.065, 0.995], [0.235, 0.135, 1.005]],
    }
    capability.detect = lambda payload: {
        "detections": {"bowl": [object_detection], "plate": [target_detection]}
    }
    result = capability.verify_support_relation({
        "frame": {}, "object_query": "bowl", "target_query": "plate",
        "source_world_xyz": [0.0, 0.0, 1.04],
        "target_world_xyz": [0.20, 0.10, 1.00],
        "target_world_bounds_10_90": target_detection["world_bounds_10_90"],
    })
    assert result["verified"] is True
    assert result["object_coverage_fraction"] == pytest.approx(0.49)
    assert result["target_coverage_fraction"] == pytest.approx(1.0)
    assert result["support_overlap_fraction"] == pytest.approx(1.0)
    assert result["support_overlap_normalization"] == "smaller_metric_footprint"


def test_visual_support_gap_uses_pre_action_plane_when_fresh_target_is_occluded():
    """A placed bowl may contaminate the fresh plate mask's upper Z bound."""
    from embodied_codex.capabilities import OpenVocabularyRGBD

    capability = OpenVocabularyRGBD.__new__(OpenVocabularyRGBD)
    bowl = {
        "world_xyz": [0.055, 0.200, 0.945],
        "world_bounds_10_90": [[0.015, 0.153, 0.910], [0.102, 0.245, 0.956]],
    }
    # This is still a credible plate detection by centroid, but its upper
    # depth quantile contains the occluding bowl and must not define contact.
    occluded_plate = {
        "world_xyz": [0.087, 0.208, 0.918],
        "world_bounds_10_90": [[0.053, 0.151, 0.909], [0.108, 0.260, 0.947]],
    }
    capability.detect = lambda payload: {
        "detections": {"bowl": [bowl], "plate": [occluded_plate]}
    }
    result = capability.verify_support_relation({
        "frame": {}, "object_query": "bowl", "target_query": "plate",
        "source_world_xyz": [0.021, -0.272, 1.171],
        "target_world_xyz": [0.053, 0.207, 0.914],
        "target_world_bounds_10_90": [[0.010, 0.150, 0.909],
                                        [0.110, 0.260, 0.920]],
    })
    assert result["verified"] is True
    assert result["support_gap_m"] == pytest.approx(-0.004)
    assert result["support_height_source"] == "pre_action_sensor_anchor"


def test_visual_support_overlap_uses_unoccluded_pre_action_footprint():
    """A credible exposed plate crescent must not replace its full anchor."""
    from embodied_codex.capabilities import OpenVocabularyRGBD

    capability = OpenVocabularyRGBD.__new__(OpenVocabularyRGBD)
    bowl = {
        "world_xyz": [0.0756, 0.2143, 0.9542],
        "world_bounds_10_90": [[0.0505, 0.1741, 0.9198],
                                [0.1124, 0.2478, 0.9606]],
    }
    exposed_plate_crescent = {
        "world_xyz": [0.1377, 0.2021, 0.9144],
        "world_bounds_10_90": [[0.1058, 0.1547, 0.9085],
                                [0.1693, 0.2571, 0.9173]],
    }
    capability.detect = lambda payload: {
        "detections": {"bowl": [bowl], "plate": [exposed_plate_crescent]}
    }
    result = capability.verify_support_relation({
        "frame": {}, "object_query": "bowl", "target_query": "plate",
        "source_world_xyz": [-0.0811, 0.1957, 0.9428],
        "target_world_xyz": [0.0753, 0.2055, 0.9144],
        "target_world_bounds_10_90": [[0.0286, 0.1581, 0.9087],
                                        [0.1203, 0.2510, 0.9173]],
    })
    assert result["target_surface_height_error_m"] < \
        result["maximum_target_surface_height_error_m"]
    assert result["target_geometry_source"] == "pre_action_sensor_anchor"
    assert result["support_overlap_fraction"] == pytest.approx(1.0)
    assert result["center_inside_target_bounds"] is True
    assert result["verified"] is True


def test_vlm_relation_grounder_returns_only_a_live_candidate(tmp_path):
    import hashlib
    from embodied_codex.capabilities import VLMVisualRelationGrounder

    image=tmp_path/"frame.png"
    image.write_bytes(b"sensor-image-bytes")
    frame={"cameras":{"agentview":{
        "rgb_path":str(image),
        "rgb_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
    }}}
    grounder=VLMVisualRelationGrounder(
        api_key="",base_url="https://unused.invalid",client=object())
    captured={}
    def complete(prompt,image_url):
        captured.update({"prompt":prompt,"image_url":image_url})
        return ('```json\n{"selected_id":1,"reference_description":"colorful box",'
                '"reason":"visually supported","confidence":0.91}\n```')
    grounder._complete=complete
    candidates=[
        {"label":"bowl","box_xyxy":[0,0,10,10],"world_xyz":[0,0,1]},
        {"label":"bowl","box_xyxy":[20,20,30,30],"world_xyz":[1,0,1],
         "point_ref":"point-live"},
    ]
    result=grounder.select({
        "frame":frame,"instruction":"pick the bowl on the box",
        "relation":"the bowl on the box","candidates":candidates})
    assert result["selected_index"]==1
    assert "selected" not in result
    assert result["confidence"]==pytest.approx(.91)
    assert "point-live" not in captured["prompt"]
    assert captured["image_url"].startswith("data:image/png;base64,")


def test_vlm_relation_grounder_requires_consensus_on_joint_live_pair(tmp_path):
    import hashlib
    from embodied_codex.capabilities import VLMVisualRelationGrounder

    image=tmp_path/"frame.png"
    image.write_bytes(b"sensor-image-bytes")
    frame={"cameras":{"agentview":{
        "rgb_path":str(image),
        "rgb_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
    }}}
    grounder=VLMVisualRelationGrounder(
        api_key="",base_url="https://unused.invalid",client=object())
    answers=iter([
        '{"selected_id":0,"selected_reference_id":0,"confidence":0.9}',
        '{"selected_id":1,"selected_reference_id":1,"confidence":0.8}',
        '{"selected_id":1,"selected_reference_id":1,"confidence":0.95}',
    ])
    grounder._complete=lambda prompt,image_url:next(answers)
    objects=[
        {"label":"bowl","box_xyxy":[0,0,10,10],"point_ref":"wrong"},
        {"label":"bowl","box_xyxy":[20,20,30,30],"point_ref":"object-live"},
    ]
    references=[
        {"label":"platform","box_xyxy":[0,10,10,20],"point_ref":"wrong-ref"},
        {"label":"cookie box","box_xyxy":[20,30,30,40],"point_ref":"ref-live"},
    ]
    result=grounder.select({
        "frame":frame,"instruction":"pick the bowl on the cookie box",
        "relation":"the bowl on the cookie box","candidates":objects,
        "reference_candidates":references,"consensus_rounds":3})
    assert result["selected_index"]==1
    assert result["selected_reference_index"]==1
    assert "selected" not in result and "selected_reference" not in result
    assert result["consensus"]=={
        "rounds":3,"completed_rounds":3,"required":2,
        "winning_votes":2,"agreed":True}


def test_vlm_capabilities_retry_transient_transport_errors_without_new_robot_logic():
    from types import SimpleNamespace
    from embodied_codex.capabilities.vlm_relation_grounder import VLMVisualRelationGrounder
    from embodied_codex.capabilities.vlm_task_outcome import VLMVisualTaskOutcomeVerifier

    class APIConnectionError(Exception):pass
    class Completions:
        def __init__(self):self.calls=0
        def create(self,**unused):
            self.calls+=1
            if self.calls==1:raise APIConnectionError("temporary")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"verified": false}'))])
    def client():
        completions=Completions()
        return SimpleNamespace(chat=SimpleNamespace(completions=completions)),completions

    relation_client,relation_calls=client()
    relation=VLMVisualRelationGrounder(api_key="",base_url="",client=relation_client,
                                       retry_delays=(0.0,))
    assert relation._complete("prompt","data:image/png;base64,eA==")
    assert relation_calls.calls==2
    outcome_client,outcome_calls=client()
    outcome=VLMVisualTaskOutcomeVerifier(api_key="",base_url="",client=outcome_client,
                                         retry_delays=(0.0,))
    assert outcome._complete("prompt","before","after")
    assert outcome_calls.calls==2


def test_vlm_consensus_rounds_run_concurrently_without_weakening_votes(tmp_path):
    import hashlib
    import threading
    from embodied_codex.capabilities import (VLMVisualRelationGrounder,
                                              VLMVisualTaskOutcomeVerifier)

    image=tmp_path/"frame.png";image.write_bytes(b"sensor-image-bytes")
    item={"rgb_path":str(image),
          "rgb_sha256":hashlib.sha256(image.read_bytes()).hexdigest()}
    frame={"cameras":{"agentview":item}}
    relation=VLMVisualRelationGrounder(api_key="",base_url="",client=object())
    relation_barrier=threading.Barrier(3)
    def relation_complete(*unused):
        relation_barrier.wait(timeout=2)
        return '{"selected_id":0,"selected_reference_id":0,"confidence":0.9}'
    relation._complete=relation_complete
    selected=relation.select({"frame":frame,"instruction":"bowl on box",
        "relation":"bowl on box","candidates":[{"label":"bowl","box_xyxy":[0,0,1,1]}],
        "reference_candidates":[{"label":"box","box_xyxy":[1,1,2,2]}],
        "consensus_rounds":3})
    assert selected["consensus"]["winning_votes"]==3

    outcome=VLMVisualTaskOutcomeVerifier(api_key="",base_url="",client=object(),
                                          consensus_rounds=3)
    outcome_barrier=threading.Barrier(3)
    def outcome_complete(*unused):
        outcome_barrier.wait(timeout=2)
        return ('{"verified":true,"source_relation_satisfied":true,'
                '"target_relation_satisfied":true,"contradiction":"","confidence":0.9}')
    outcome._complete=outcome_complete
    verified=outcome.verify({"instruction":"move bowl","before":item,"after":item})
    assert verified["verified"] is True
    assert verified["consensus"]["true_votes"]==3


def test_vlm_consensus_has_one_wall_clock_deadline(tmp_path):
    import hashlib
    import time
    from embodied_codex.capabilities import (VLMRelationGroundingError,
                                              VLMVisualRelationGrounder)

    image=tmp_path/"frame.png";image.write_bytes(b"sensor-image-bytes")
    item={"rgb_path":str(image),
          "rgb_sha256":hashlib.sha256(image.read_bytes()).hexdigest()}
    grounder=VLMVisualRelationGrounder(
        api_key="",base_url="",client=object(),total_timeout=.05)
    grounder._complete=lambda *unused:time.sleep(.5)
    started=time.monotonic()
    with pytest.raises(VLMRelationGroundingError,match="exceeded"):
        grounder.select({"frame":{"cameras":{"agentview":item}},
            "instruction":"bowl on plate","relation":"bowl on plate",
            "candidates":[{"label":"bowl","box_xyxy":[0,0,1,1]}],
            "reference_candidates":[{"label":"plate","box_xyxy":[1,1,2,2]}],
            "consensus_rounds":3})
    assert time.monotonic()-started < .3


def test_vlm_consensus_uses_quorum_when_one_request_hangs():
    import time
    from embodied_codex.capabilities._vlm_support import bounded_consensus

    def call(index, unused_deadline):
        if index==2:time.sleep(.5)
        return f"vote-{index}"

    started=time.monotonic()
    votes=bounded_consensus(3,.4,call,error_type=RuntimeError,
        operation="test consensus",minimum_results=2,completion_grace=.02)
    assert votes==["vote-0","vote-1"]
    assert time.monotonic()-started < .2


def test_vlm_consensus_waits_for_deciding_vote_after_split_pair():
    import time
    from embodied_codex.capabilities._vlm_support import bounded_consensus

    def call(index, unused_deadline):
        if index==2:time.sleep(.04)
        return [True,False,True][index]

    votes=bounded_consensus(3,.3,call,error_type=RuntimeError,
        operation="split vote",minimum_results=2,completion_grace=.01,
        decision_quorum=lambda values:
            max((values.count(True),values.count(False)),default=0)>=2)
    assert votes==[True,False,True]


def test_vlm_task_outcome_normalizes_none_and_waits_for_majority(tmp_path):
    import hashlib
    from queue import Queue
    from embodied_codex.capabilities import VLMVisualTaskOutcomeVerifier

    image=tmp_path/"frame.png";image.write_bytes(b"sensor-image")
    item={"rgb_path":str(image),
          "rgb_sha256":hashlib.sha256(image.read_bytes()).hexdigest()}
    answers=Queue()
    answers.put('{"verified":true,"source_relation_satisfied":true,'
        '"target_relation_satisfied":true,"contradiction":"none",'
        '"reason":"complete","confidence":0.9}')
    answers.put('{"verified":false,"source_relation_satisfied":true,'
        '"target_relation_satisfied":false,"contradiction":"off target",'
        '"reason":"failed","confidence":0.8}')
    answers.put('{"verified":true,"source_relation_satisfied":true,'
        '"target_relation_satisfied":true,"contradiction":"",'
        '"reason":"complete","confidence":0.95}')
    verifier=VLMVisualTaskOutcomeVerifier(api_key="",base_url="",client=object(),
        consensus_rounds=3,total_timeout=1)
    verifier._complete=lambda *unused:answers.get_nowait()
    result=verifier.verify({"instruction":"move bowl","before":item,"after":item})
    assert result["verified"] is True
    assert result["consensus"]["completed_rounds"]==3
    assert result["consensus"]["true_votes"]==2


def test_vlm_transport_preview_preserves_source_and_reduces_large_png(tmp_path):
    import hashlib
    import random
    from PIL import Image
    from embodied_codex.capabilities import VLMVisualTaskOutcomeVerifier

    image=tmp_path/"frame.png"
    pixels=random.Random(7).randbytes(256*256*3)
    Image.frombytes("RGB",(256,256),pixels).save(image,format="PNG")
    source=image.read_bytes();digest=hashlib.sha256(source).hexdigest()
    verifier=VLMVisualTaskOutcomeVerifier(
        api_key="",base_url="",client=object())
    url=verifier._image({"rgb_path":str(image),"rgb_sha256":digest})
    assert hashlib.sha256(image.read_bytes()).hexdigest()==digest
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) < len(source)*4/3


def test_vlm_task_outcome_rejects_same_class_substitution(tmp_path):
    import hashlib
    from embodied_codex.capabilities import VLMVisualTaskOutcomeVerifier

    frames=[]
    for name in ("before","after"):
        path=tmp_path/f"{name}.png";path.write_bytes(name.encode())
        frames.append({"rgb_path":str(path),
                       "rgb_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    verifier=VLMVisualTaskOutcomeVerifier(
        api_key="",base_url="https://unused.invalid",client=object(),
        consensus_rounds=3)
    answers=iter([
        '{"verified":false,"source_relation_satisfied":false,'
        '"target_relation_satisfied":true,"contradiction":"specified object remains",'
        '"reason":"wrong instance moved","confidence":0.96}',
        '{"verified":false,"source_relation_satisfied":false,'
        '"target_relation_satisfied":true,"contradiction":"source unchanged",'
        '"reason":"wrong instance moved","confidence":0.94}',
        '{"verified":true,"source_relation_satisfied":true,'
        '"target_relation_satisfied":true,"contradiction":"",'
        '"reason":"looks complete","confidence":0.7}',
    ])
    verifier._complete=lambda prompt,before_url,after_url:next(answers)
    result=verifier.verify({"instruction":"move the bowl on the box to the plate",
                            "before":frames[0],"after":frames[1]})
    assert result["verified"] is False
    assert result["consensus"]["false_votes"]==2


def test_frozen_class_loader_resolves_only_declared_relative_dependency(tmp_path):
    from embodied_codex.examples.evaluate_libero_skill import _load_class

    dependency = tmp_path / "dependency.py"
    dependency.write_text("class FrozenError(ValueError): pass\n")
    tool = tmp_path / "tool.py"
    tool.write_text(
        "from .dependency import FrozenError\n"
        "class FrozenTool:\n"
        "    error_type = FrozenError\n"
    )
    loaded = _load_class(
        tool, "FrozenTool", relative_modules={"dependency": dependency})
    assert loaded.error_type.__name__ == "FrozenError"
    assert loaded.__module__.startswith("embodied_codex_frozen_package_")


def test_bootstrap_tool_ids_are_rebound_by_python_ast_only():
    import ast
    from embodied_codex.legacy.evolution import remap_controller_tool_ids

    source = (
        'PERCEPT = "perception:v002"\n'
        'UNCHANGED = "prefix perception:v002 suffix"\n'
        'def run(robot):\n'
        '    return robot.use(PERCEPT, {})\n'
    )
    rebound, changed = remap_controller_tool_ids(
        source, {"perception:v002": "perception:v001"})
    values = [node.value for node in ast.walk(ast.parse(rebound))
              if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert changed == 1
    assert "perception:v001" in values
    assert "prefix perception:v002 suffix" in values
    assert "perception:v002" not in values


def test_invalid_persistent_controller_rebinds_string_tokens_before_agent_repair():
    from embodied_codex.legacy.evolution import remap_controller_tool_ids
    source=(
        'TOOL = "perception:v002"\n'
        '# perception:v002 must not be rewritten in comments\n'
        'def run(robot):\n'
        '    return robot.use(TOOL, {})}\n')
    rebound,changed=remap_controller_tool_ids(
        source,{"perception:v002":"perception:v003"})
    assert changed==1
    assert 'TOOL = \'perception:v003\'' in rebound
    assert '# perception:v002 must not be rewritten in comments' in rebound
    with pytest.raises(SyntaxError):compile(rebound,"controller.py","exec")

    indentation_broken=(
        'TOOL = "perception:v002"\n'
        'def run(robot):\n'
        '    value = 1\n'
        '  return robot.use(TOOL, {})\n')
    rebound,changed=remap_controller_tool_ids(
        indentation_broken,{"perception:v002":"perception:v003"})
    assert changed==1 and "perception:v003" in rebound
    with pytest.raises(IndentationError):compile(rebound,"controller.py","exec")


def test_resumed_bootstrap_preserves_evolved_controller(tmp_path):
    import hashlib
    import json
    from embodied_codex.legacy.evolution import EvolutionEngine

    skill = tmp_path / "skill"
    skill.mkdir()
    original = "def run(robot):\n    return {'status': 'original'}\n"
    (skill / "controller.py").write_text(original)
    (skill / "manifest.json").write_text(json.dumps({
        "protocol": "embodied-codex-skill-v1",
        "skill_id": "test_skill:v001",
        "task": "test",
        "controller_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "tool_ids": [],
    }))
    engine = EvolutionEngine(
        root=tmp_path / "run", model=object(), deployment_factory=lambda: None)
    first = engine.bootstrap_skill(skill)
    evolved = "def run(robot):\n    return {'status': 'evolved'}\n"
    engine.workspace.write_file("controller.py", evolved)
    engine.state_path.write_text(json.dumps({"status": "evolving"}))
    second = engine.bootstrap_skill(skill)
    assert first["skill_id"] == second["skill_id"] == "test_skill:v001"
    assert engine.workspace._path("controller.py").read_text() == evolved


def test_kernel_prompt_delegates_domain_reasoning_to_the_engineering_agent():
    from embodied_codex.legacy.evolution import SYSTEM_PROMPT

    assert "no task template" in SYSTEM_PROMPT
    assert "externally supplied failure" in SYSTEM_PROMPT
    assert "search_web" in SYSTEM_PROMPT
    assert "register it as a Tool" in SYSTEM_PROMPT
    assert "not a whitelist for Agent-authored Tools" in SYSTEM_PROMPT
    assert "task-specific rule" in SYSTEM_PROMPT
    assert "inspect_robot_sdk_contract" in SYSTEM_PROMPT
    assert "Never\ncall read_file for deployment guidance" in SYSTEM_PROMPT
    for leaked_policy in ("black bowl", "cookie box", "ramekin",
                          "globally highest-scoring same-class object",
                          "grasp pose", "support relation"):
        assert leaked_policy not in SYSTEM_PROMPT


def test_asset_registration_exposes_exact_provenance_contract(tmp_path):
    workspace = TaskWorkspace(tmp_path / "workspace")
    surface = EngineeringSurface(
        workspace=workspace,
        capabilities=CapabilityLibrary(tmp_path / "capabilities", workspace.root,
                                       python=PYTHON),
        runtime=object(), deployment_factory=lambda: None,
        artifact_dir=tmp_path / "iterations" / "iteration_001",
        skills=SkillLibrary(tmp_path / "skills"))
    schemas = {item["function"]["name"]: item["function"]["parameters"]
               for item in surface.registry().schemas}
    for operation in ("register_tool", "register_capability_package"):
        source_urls = schemas[operation]["properties"]["source_urls"]
        assert source_urls["minItems"] == 1
        assert source_urls["items"]["pattern"] == "^https://"
    tool_schema=schemas["register_tool"]
    assert set(tool_schema["required"])=={
        "name","source_path","description","input_schema","output_schema",
        "source_urls","implementation_origin","trained_on_current_task"}
    assert set(tool_schema["properties"]["implementation_origin"]["properties"][
        "kind"]["enum"])=={"original_synthesis","adapted_source","adopted_source"}
    assert "provenance" not in tool_schema["properties"]
    assert "manual" not in tool_schema["required"]
    assert schemas["propose_skill_interface"]["required"]==[]
    proposed=surface.propose_skill_interface()
    assert proposed["accepted"] is True
    assert proposed["interface"]["required_robot_operations"]==[]
    provenance=schemas["register_capability_package"]["properties"]["provenance"]
    assert set(provenance["required"])=={
        "training_data_declaration","contamination_check"}
    contamination=provenance["properties"]["contamination_check"]
    assert set(contamination["required"])=={
        "evaluated_benchmark","method","result"}
    assert "models" in provenance["properties"]


def test_experience_revision_supersedes_and_retrieval_hides_stale_version(tmp_path):
    evidence=tmp_path/"evidence.json";evidence.write_text("{}\n")
    library=ExperienceLibrary(tmp_path/"experiences")
    first=library.register(name="dynamic_binding_lesson",summary="Old binding failed.",
        applicability="Old adapter version only.",keywords=["binding"],
        evidence_paths=[evidence])
    second=library.register(name="dynamic_binding_lesson",summary="Binding now succeeds.",
        applicability="Current adapter version.",keywords=["binding"],
        evidence_paths=[evidence])
    assert library.inspect(second["experience_id"])["supersedes"]==first["experience_id"]
    summaries=library.list_summaries()
    assert [row["experience_id"] for row in summaries]==[second["experience_id"]]
    assert library.search("binding",limit=8)[0]["summary"]=="Binding now succeeds."


@pytest.mark.parametrize("case_name", ["cursor", "valves", "thermal"])
def test_benchmark_neutral_kernel_adapters_have_distinct_machine_contracts(tmp_path,
                                                                          case_name):
    from embodied_codex.examples.run_kernel_conformance import (
        CASES, ConformanceDeployment, sdk_contract)

    case=CASES[case_name];contract=sdk_contract(case)
    deployment=ConformanceDeployment(case_name,tmp_path/case_name)
    observation=deployment.dispatch("observe",{"channel":"state","request":{}})
    assert observation==case["initial"]
    assert set(contract["actions"])==set(case["actions"])
    assert set(contract["verifiers"])=={case["verifier"]}
    deployment.close()


def _causal_task_model():
    from embodied_codex.legacy.task_model import canonical_sha256
    value={
        "protocol":"embodied-codex-task-model-v1",
        "instruction":"retrieve the object from a container and place it on a support",
        "entities":[{"id":"object","description":"the object","role":"manipulated"}],
        "requirements":[
            {"id":"accessible","kind":"accessibility","description":"object is reachable"},
            {"id":"placed","kind":"goal","description":"object is stably on support"},
        ],
        "phases":[
            {"id":"make_accessible","purpose":"inspect and conditionally expose object",
             "depends_on":[],"satisfies_requirements":["accessible"],
             "required_robot_operations":["observe","act","verify"],
             "observations":["container geometry"],"actions":["conditional articulation"],
             "success_evidence":["accessibility verification"],
             "capability_requirements":["articulation"]},
            {"id":"place","purpose":"place and verify object","depends_on":["make_accessible"],
             "satisfies_requirements":["placed"],
             "required_robot_operations":["act","verify"],
             "observations":["support"],"actions":["place"],
             "success_evidence":["stable support relation"],
             "capability_requirements":["placement"]},
        ],"capability_gaps":[]}
    value["task_model_sha256"]=canonical_sha256(value)
    return value


def test_task_model_rejects_uncovered_language_requirement():
    from embodied_codex.legacy.task_model import TaskModelError,validate_task_model
    value=_causal_task_model();value.pop("task_model_sha256")
    value["requirements"].append(
        {"id":"identity","kind":"source_relation","description":"specific source instance"})
    with pytest.raises(TaskModelError,match="requirements without a phase"):
        validate_task_model(value,value["instruction"])


def test_task_model_preflight_binds_reachable_robot_operations_and_seals_hash(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def expose(robot):
    robot.observe("rgbd", {})
    robot.act({"kind": "conditional articulation"})
    return robot.verify("accessible", {})

def place(robot):
    robot.act({"kind": "place"})
    return robot.verify("support", {})

def run(robot):
    expose(robot)
    proof = place(robot)
    return {"status": "sensor_success" if proof["verified"] else "sensor_failure"}
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        task_model=_causal_task_model(),semantic_reviewer=lambda **unused:{
            "approved":True,"covered_phase_ids":["make_accessible","place"],"issues":[]})
    binding=[
        {"phase_id":"make_accessible","functions":["expose"],
         "robot_operations":["observe","act","verify"]},
        {"phase_id":"place","functions":["place"],"robot_operations":["act","verify"]},
    ]
    receipt=surface.preflight_controller("controller.py",binding)
    assert receipt["eligible"] is True
    assert surface._require_current_preflight(workspace._path("controller.py"))
    workspace.replace_in_file("controller.py",'"kind": "place"','"kind": "changed"')
    with pytest.raises(RuntimeError,match="changed after task-model preflight"):
        surface._require_current_preflight(workspace._path("controller.py"))


def test_task_model_preflight_rejects_declared_but_absent_action(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def expose(robot):
    robot.observe("rgbd", {})
    return robot.verify("accessible", {})
def place(robot):
    robot.act({"kind":"place"})
    return robot.verify("support", {})
def run(robot):
    expose(robot)
    return place(robot)
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        task_model=_causal_task_model())
    with pytest.raises(RuntimeError,match="claims absent operations"):
        surface.preflight_controller("controller.py",[
            {"phase_id":"make_accessible","functions":["expose"],
             "robot_operations":["observe","act","verify"]},
            {"phase_id":"place","functions":["place"],
             "robot_operations":["act","verify"]},
        ])


def test_task_model_preflight_rejects_semantically_wrong_recovered_controller(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def expose(robot):
    robot.observe("rgbd", {})
    robot.act({"kind": "select unrelated tabletop object"})
    return robot.verify("accessible", {})
def place(robot):
    robot.act({"kind": "place"})
    return robot.verify("support", {})
def run(robot):
    expose(robot)
    return place(robot)
''')
    reviewed=[]
    def reviewer(**payload):
        reviewed.append(payload)
        return {"approved":False,"covered_phase_ids":["place"],
                "issues":["source qualifier and accessibility phase are not implemented"]}
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        task_model=_causal_task_model(),semantic_reviewer=reviewer)
    with pytest.raises(RuntimeError,match="source qualifier"):
        surface.preflight_controller("controller.py",[
            {"phase_id":"make_accessible","functions":["expose"],
             "robot_operations":["observe","act","verify"]},
            {"phase_id":"place","functions":["place"],
             "robot_operations":["act","verify"]},
        ])
    assert reviewed and "unrelated tabletop object" in reviewed[0]["source"]
    assert not (workspace.root/"task_plan_binding.json").exists()


def test_libero_campaign_uses_free_coding_without_semantic_critics():
    from embodied_codex.examples.run_libero import (LIBERO_REQUIRE_TASK_FIDELITY_REVIEW,
                                                     LIBERO_REQUIRE_TASK_MODEL)
    assert LIBERO_REQUIRE_TASK_FIDELITY_REVIEW is False
    assert LIBERO_REQUIRE_TASK_MODEL is False


def test_task_fidelity_review_rejects_task_drift_and_caches_verdict(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.use("selector", {"relation":"next_to", "reference":"ramekin"})
    return {"sensor_failure": "not run"}
''')
    reviews=[]
    def reviewer(**payload):
        reviews.append(payload)
        return {"approved":False,
                "issues":["controller selects a bowl next to a ramekin, not the bowl in the drawer"]}
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        task_instruction="pick up the black bowl in the top drawer",
        task_fidelity_reviewer=reviewer)
    controller=workspace._path("controller.py")
    with pytest.raises(RuntimeError,match="not the bowl in the drawer"):
        surface._require_task_fidelity(controller)
    with pytest.raises(RuntimeError,match="not the bowl in the drawer"):
        surface._require_task_fidelity(controller)
    assert len(reviews)==1
    binding=json.loads((workspace.root/"task_fidelity_binding.json").read_text())
    assert binding["review"]["approved"] is False


def test_fidelity_rejected_source_is_not_scheduled_as_pending_executable(tmp_path):
    root=tmp_path/"fidelity-handoff"
    engine=EvolutionEngine(root=root,model=object(),deployment_factory=lambda:None,
        python=PYTHON)
    controller=engine.workspace.root/"controller.py"
    controller.write_text('def run(robot):\n    return {"sensor_failure":"wrong relation"}\n')
    digest=hashlib.sha256(controller.read_bytes()).hexdigest()
    (engine.workspace.root/"task_fidelity_binding.json").write_text(json.dumps({
        "protocol":"embodied-codex-task-fidelity-binding-v1",
        "controller_sha256":digest,"instruction_sha256":"instruction",
        "review":{"approved":False,"issues":["wrong source relation"]}})+"\n")
    rejection=engine._current_task_fidelity_rejection(controller)
    assert rejection and rejection["controller_sha256"]==digest
    assert rejection["issues"]==["wrong source relation"]
    controller.write_text(controller.read_text()+"# changed\n")
    assert engine._current_task_fidelity_rejection(controller) is None


def test_exhausted_frontier_does_not_build_task_model_on_resume(tmp_path):
    root=tmp_path/"exhausted"
    engine=EvolutionEngine(root=root,model=object(),deployment_factory=lambda:None,
        python=PYTHON,require_task_model=True)
    state={"task":"move the cube","skill_name":"frontier_skill","status":"evolving",
           "iterations":[{"iteration":1,"agent_completed":True,
                          "evidence":{"sensor_success_candidate":False}}]}
    engine._save(state)
    engine._task_model=lambda task: pytest.fail("exhausted task must not invoke task modeling")
    resumed=engine.run(task="move the cube",skill_name="frontier_skill",max_iterations=1)
    assert resumed==state


def test_robot_contract_preflight_rejects_action_typo_and_null_ref(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.act({"type":"move_pose", "pose_ref":"p"})
    return robot.verify("support", {"source_ref":"s", "target_ref":None})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match="unsupported literal action type.*move_pose"):
        surface._lint_robot_contract(workspace._path("controller.py"))
    workspace.replace_in_file("controller.py",'"move_pose"','"move_to_pose"')
    with pytest.raises(RuntimeError,match="target_ref cannot be literal None"):
        surface._lint_robot_contract(workspace._path("controller.py"))


def test_robot_contract_preflight_rejects_missing_or_invalid_run_before_deployment(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    deployments=[]
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:deployments.append(True),
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)

    workspace.write_file("controller.py","def run_identity_gated(robot):\n    return {}\n")
    with pytest.raises(RuntimeError,match=r"top-level run\(robot\)"):
        surface.run_robot_controller("controller.py")
    assert deployments==[]
    assert surface.robot_runs==0

    workspace.write_file("controller.py","def run(robot, optional=None):\n    return {}\n")
    with pytest.raises(RuntimeError,match=r"exact signature run\(robot\)"):
        surface.run_robot_controller("controller.py")
    assert deployments==[]
    assert surface.robot_runs==0

    workspace.write_file("controller.py","async def run(robot):\n    return {}\n")
    with pytest.raises(RuntimeError,match=r"run\(robot\) must be synchronous"):
        surface.run_robot_controller("controller.py")
    assert deployments==[]
    assert surface.robot_runs==0


def test_robot_sdk_contract_is_available_as_sectioned_on_demand_manual(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    surface=EngineeringSurface(workspace=TaskWorkspace(tmp_path/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=tmp_path/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    section=surface.inspect_robot_sdk_contract("actions.move_to_pose")
    assert section["protocol"]==LIBERO_ROBOT_SDK_CONTRACT["protocol"]
    assert section["section"]=="actions.move_to_pose"
    assert section["contract"]==LIBERO_ROBOT_SDK_CONTRACT["actions"]["move_to_pose"]
    with pytest.raises(RuntimeError,match="unknown Robot SDK contract section"):
        surface.inspect_robot_sdk_contract("actions.move_pose")


def test_libero_osc_contract_distinguishes_commands_from_metric_displacement():
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT

    contract=LIBERO_ROBOT_SDK_CONTRACT["actions"]["osc_delta"]
    assert "NOT metres" in contract["field_semantics"]["translation"]
    assert "move_to_point/move_to_pose" in contract["rule"]


def test_engineering_agent_can_extract_and_view_rollout_keyframes(tmp_path):
    import cv2
    import json
    import numpy as np

    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    episode=run/"episodes"/"episode_001";episode.mkdir(parents=True)
    video=episode/"rollout.mp4"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(32,24))
    assert writer.isOpened()
    for value in (20,80,140,220):
        writer.write(np.full((24,32,3),value,dtype=np.uint8))
    writer.release()
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=run/"iterations"/"iteration_001")
    execution=run/"iterations"/"iteration_001"/"robot_execution.json"
    execution.parent.mkdir(parents=True,exist_ok=True)
    execution.write_text(json.dumps({"sensor_report":{"rollout_path":str(video)}}))
    result=surface.extract_rollout_frames(str(video),[1,3],max_frames=4)
    assert result["total_frames"]==4
    assert [row["index"] for row in result["frames"]]==[1,3]
    for row in result["frames"]:
        assert row["artifact_ref"].startswith("image-")
        delivered=surface.view_sensor_image(artifact_ref=row["artifact_ref"])
        assert delivered["_embodied_codex_image"]["mime_type"]=="image/png"
    sampled=surface.extract_rollout_frames(str(video),[],max_frames=3)
    assert [row["index"] for row in sampled["frames"]]==[0,2,3]
    latest=surface.extract_rollout_frames("latest_rollout",[0],max_frames=1)
    assert latest["rollout_path"]==str(video.resolve())
    compatible=surface.extract_rollout_frames("latest_robot_execution",[3],max_frames=1)
    assert [row["index"] for row in compatible["frames"]]==[3]
    outside=tmp_path/"outside.mp4";outside.write_bytes(video.read_bytes())
    with pytest.raises(RuntimeError,match="inside the current run"):
        surface.extract_rollout_frames(str(outside),[],max_frames=3)
    execution.write_text(json.dumps({"sensor_report":{"rollout_path":str(outside)}}))
    with pytest.raises(RuntimeError,match="inside the current run"):
        surface.extract_rollout_frames("latest_rollout",[],max_frames=3)


def test_tool_manual_is_default_and_source_access_is_explicitly_paginated(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("long_tool.py","\n".join(
        ["def run(payload):", "    return payload"]+[f"# line {i}" for i in range(500)])+"\n")
    library=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root)
    tool=library.register_tool(name="long_inspection_tool",source_path="long_tool.py",
        description="long",input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],
        trained_on_current_task=False)
    surface=EngineeringSurface(workspace=workspace,capabilities=library,runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001")
    described=surface.inspect_tool(tool["tool_id"])
    assert "source" not in described
    assert "tests" not in described["manifest"]
    assert described["manifest"]["test_summary"]=={
        "batches":0,"cases":0,"all_passed":False,
        "status_authority":"capability_library_tests"}
    assert described["manual"]["manual"]["purpose"]=="long"
    page=surface.read_tool_source(tool["tool_id"],start_line=101,end_line=150)
    assert page["source"]["start_line"]==101
    assert page["source"]["end_line"]==150
    assert page["source"]["total_lines"]==502
    assert page["source"]["next_start_line"]==151
    assert "# line 98" in page["source"]["content"]
    assert "# line 148" not in page["source"]["content"]


def test_tool_manual_revisions_are_immutable_and_require_run_local_evidence(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True);evidence=artifact/"robot_execution.json"
    evidence.write_text('{"observed_contract_gap": true}\n')
    workspace=TaskWorkspace(run/"workspace")
    workspace.write_file("tool.py","def run(payload): return payload\n")
    library=CapabilityLibrary(run/"capabilities",workspace.root,python=PYTHON)
    tool_id=library.register_tool(name="manual_revision_tool",source_path="tool.py",
        description="identity",input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],
        trained_on_current_task=False)["tool_id"]
    surface=EngineeringSurface(workspace=workspace,capabilities=library,runtime=object(),
        deployment_factory=lambda:None,artifact_dir=artifact)
    original=library.manual(tool_id);assert original["manual_revision"]==1
    manual=dict(original["manual"]);manual["limitations"]=["Observed limitation"]
    revised=surface.revise_tool_manual(tool_id,manual,["latest_robot_execution"])
    assert revised["manual_revision"]==2
    assert revised["evidence"][0]["sha256"]
    revision_dir=library._manual_dir(tool_id)
    assert (revision_dir/"r001.json").is_file() and (revision_dir/"r002.json").is_file()
    assert __import__("json").loads((revision_dir/"r001.json").read_text())==original
    outside=tmp_path/"outside.json";outside.write_text("{}")
    with pytest.raises(RuntimeError,match="current run"):
        surface.revise_tool_manual(tool_id,manual,[str(outside)])


def test_model_can_publish_and_reuse_evidence_backed_experience(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True);evidence=artifact/"robot_execution.json"
    evidence.write_text('{"sensor_only": true}\n')
    experiences=ExperienceLibrary(tmp_path/"shared_experiences")
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact,experiences=experiences)
    registered=surface.register_experience(
        name="bounded_control_lesson",summary="Use feedback when commands saturate.",
        applicability="Adapters returning measured state after bounded commands.",
        keywords=["feedback","saturation"],evidence_refs=["latest_robot_execution"])
    assert registered["status"]=="evidence_backed"
    summaries=experiences.list_summaries()
    assert summaries[0]["summary"]=="Use feedback when commands saturate."
    evidence.write_text('{"original_run_changed": true}\n')
    inspected=experiences.inspect(registered["experience_id"])
    assert inspected["evidence"][0]["path"]=="evidence/001_robot_execution.json"
    assert inspected["evidence"][0]["original_path"]==str(evidence.resolve())
    public=surface.inspect_experience(registered["experience_id"])
    public_evidence=public["evidence"][0]
    assert "original_path" not in public_evidence
    assert public_evidence["asset_ref"].startswith(registered["experience_id"]+"#")
    copied=surface.read_run_artifact(public_evidence["asset_ref"])
    assert copied["exists"] is True and '"sensor_only": true' in copied["content"]
    copied_by_file=surface.read_file(public_evidence["asset_ref"])
    assert copied_by_file["exists"] is True and '"sensor_only": true' in copied_by_file["content"]
    with pytest.raises(RuntimeError,match="inside the current run"):
        surface.register_experience(name="escaped_lesson",summary="bad",applicability="bad",
            keywords=[],evidence_refs=[str(tmp_path/"outside.json")])


def test_pre_execution_asset_default_binds_previous_not_missing_latest(tmp_path):
    run=tmp_path/"run"
    previous=run/"iterations"/"iteration_001"/"robot_execution.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({"sensor_success_candidate":False,
        "sensor_report":{"independent_task_outcome":{"verified":False}}}))
    current=run/"iterations"/"iteration_002"
    experiences=ExperienceLibrary(tmp_path/"experiences")
    gaps=CapabilityGapLibrary(tmp_path/"gaps")
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=current,experiences=experiences,gaps=gaps)

    lesson=surface.register_experience(name="prior_failure",summary="Prior evidence failed.",
        applicability="The next correction pass.",keywords=["failure"])
    gap=surface.record_capability_gap(name="prior_gap",task="move object",
        failure_summary="Prior execution failed.",status="observed")

    assert experiences.inspect(lesson["experience_id"])["evidence"][0]["original_path"]==str(previous)
    assert gaps.inspect(gap["gap_id"])["evidence"][0]["original_path"]==str(previous)


def test_execution_evidence_gate_prevents_failed_rollout_success_promotion(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True);evidence=artifact/"robot_execution.json"
    evidence.write_text(json.dumps({
        "sensor_success_candidate":False,
        "sensor_report":{"sensor_verification_passed":False,
            "independent_task_outcome":{"verified":False}}}))
    experiences=ExperienceLibrary(tmp_path/"experiences")
    gaps=CapabilityGapLibrary(tmp_path/"gaps")
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact,experiences=experiences,gaps=gaps)

    registered=surface.register_experience(name="false_success_claim",
        summary="This was validated and succeeded.",applicability="All tasks.",
        keywords=["success"],evidence_refs=["latest_robot_execution"])
    experience=experiences.inspect(registered["experience_id"])
    assert registered["status"]=="failure_evidence"
    assert experience["evidence_assessment"]["outcome"]=="failure"

    gap=surface.record_capability_gap(name="failed_validation_claim",task="move object",
        failure_summary="Independent evidence failed.",hypotheses=["identity mismatch"],
        selected_diagnosis="wrong instance",required_capability={"kind":"grounding"},
        task_validation={"status":"success","independent_sensor_report":"verified=true"},
        status="diagnosed",evidence_refs=["latest_robot_execution"])
    validation=gaps.inspect(gap["gap_id"])["task_validation"]
    assert validation["authoritative_outcome"]=="failure"
    assert validation["model_claim_conflicts_with_evidence"] is True


def test_bootstrap_asset_retrieval_is_a_bounded_index():
    from embodied_codex.legacy.evolution import EvolutionEngine
    long="x"*5000
    experiences,skills,gaps=EvolutionEngine._retrieved_asset_index(
        experiences=[{"experience_id":"lesson:v001","name":"lesson","summary":long,
            "applicability":long,"keywords":[str(i) for i in range(30)],
            "status":"failure_evidence","evidence":[long]}],
        skills=[{"skill_id":"skill:v001","task":long,"status":"sensor_success",
            "interface":{"effects":[long]*10,"required_sensors":[str(i) for i in range(30)]},
            "tool_ids":[str(i) for i in range(30)],"evidence_files":[long]}],
        gaps=[{"gap_id":"gap:v001","name":"gap","task":long,"status":"diagnosed",
            "failure_summary":long,"selected_diagnosis":long,
            "required_capability":{"kind":long,"inputs":[long]},
            "authoritative_outcome":"failure",
            "model_claim_conflicts_with_evidence":True,"evidence":[long]}])
    encoded=json.dumps({"experiences":experiences,"skills":skills,"gaps":gaps})
    assert len(encoded)<6000
    assert "evidence" not in experiences[0] and "evidence_files" not in skills[0]
    assert "inputs" not in gaps[0]
    assert gaps[0]["authoritative_outcome"]=="failure"


def test_bootstrap_tool_retrieval_is_manual_first_not_full_schema_injection():
    long="description "*1000
    rows=EvolutionEngine._retrieved_tool_index([{
        "tool_id":"detector:v001","description":long,"status":"tested",
        "input_schema":{"type":"object","properties":{
            "frame":{"type":"object","description":long},
            "queries":{"type":"array","items":{"type":"string"}}},
            "required":["frame","queries"]},
        "output_schema":{"type":"object","properties":{
            "detections":{"type":"object","description":long}}},
        "execution_owned_by_deployment":True}])
    assert rows==[{
        "tool_id":"detector:v001","description":rows[0]["description"],
        "input_fields":["frame","queries"],
        "required_inputs":["frame","queries"],
        "output_fields":["detections"],
        "execution_owned_by_deployment":True}]
    assert len(rows[0]["description"])==180
    assert "input_schema" not in rows[0] and "output_schema" not in rows[0]
    assert len(json.dumps(rows))<1000


def test_previous_evidence_prompt_keeps_decision_and_artifact_not_bulk_trace():
    previous={
        "controller_path":"controller.py","controller_snapshot":"snapshot.py",
        "completed":True,"error":None,"sensor_success_candidate":False,
        "controller_result":{"sensor_failure":True,"reason":"contact failed",
            "candidates":[{"blob":"x"*5000} for _ in range(50)]},
        "rpc_evidence":[{"method":"act","index":i,"blob":"x"*1000}
                        for i in range(100)],
        "controller_records":[{"phase":str(i),"blob":"x"*1000}
                              for i in range(100)],
        "sensor_report":{"sensor_verification_passed":False,
            "independent_task_outcome":{"verified":False,"reason":"unchanged"},
            "outcome_observations":{"before":{"rgb_path":"before.png"},
                                    "after":{"rgb_path":"after.png"}},
            "unneeded_bulk":"x"*100000},
        "full_execution_artifact":"robot_execution.json",
        "execution_artifact_ref":"previous_robot_execution"}
    compact=EvolutionEngine._prompt_previous_evidence(previous)
    assert compact["sensor_success_candidate"] is False
    assert compact["sensor_report"]["independent_task_outcome"]["verified"] is False
    assert compact["full_execution_artifact"]=="robot_execution.json"
    assert len(compact["rpc_evidence_tail"])==6
    assert "controller_records" not in compact
    assert "inspect_execution_event" in compact["rpc_detail_rule"]
    assert "unneeded_bulk" not in compact["sensor_report"]
    assert len(json.dumps(compact))<8000


def test_previous_evidence_prompt_does_not_turn_precompacted_lists_into_keys():
    previous={"rpc_evidence":{"type":"list","count":20,
            "head":[{"method":"act"},{"method":"verify"}],"remaining":18},
        "controller_records":{"type":"list","count":9,
            "head":[{"phase":"ground"}],"remaining":8},
        "sensor_report":{}}

    compact=EvolutionEngine._prompt_previous_evidence(previous)

    assert compact["rpc_evidence_tail"]==[{"method":"act"},{"method":"verify"}]
    assert "controller_records" not in compact
    assert "type" not in compact["rpc_evidence_tail"]


def test_deployment_guidance_prompt_uses_compact_sdk_index():
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    guidance={"adapter":"LIBERO","robot_sdk_contract":LIBERO_ROBOT_SDK_CONTRACT,
              "seed_tool_ids":["detector:v001"]}
    compact=EvolutionEngine._prompt_deployment_guidance(guidance)
    contract=compact["robot_sdk_contract"]
    assert contract["full_manual"]=="inspect_robot_sdk_contract(section=...)"
    assert contract["actions"]["move_to_pose"]["required"]==["type"]
    assert contract["actions"]["move_to_pose"]["any_of"]==[
        {"required":["pose_ref"]},
        {"required":["target_ref","quaternion_xyzw"]},
        {"required":["target_ref","rotation_matrix"]}]
    assert "example" not in contract["actions"]["move_to_pose"]
    assert len(json.dumps(compact))<4000
    assert guidance["robot_sdk_contract"] is LIBERO_ROBOT_SDK_CONTRACT


def test_query_run_json_synthesizes_stable_rpc_event_indices_after_filtering(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    artifact=tmp_path/"run"/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    execution=artifact/"robot_execution.json"
    execution.write_text(json.dumps({"execution":{"rpc_events":[
        {"method":"observe"},
        {"method":"act","result":{"reached":False}},
        {"method":"record"},
        {"method":"act","result":{"reached":True}},
    ]}}))
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=artifact)

    result=surface.query_run_json(str(execution),"/execution/rpc_events",
        filters=[{"field":"method","op":"eq","value":"act"}],
        fields=["event_index","method","/result/reached"],limit=10)
    assert [row["event_index"] for row in result["rows"]]==[1,3]
    assert [row["/result/reached"] for row in result["rows"]]==[False,True]


def test_uncommitted_iteration_trace_recovers_as_actionable_correction_handoff(tmp_path):
    trace=tmp_path/"iteration_006"/"agent_trace.jsonl";trace.parent.mkdir()
    trace.write_text("\n".join(json.dumps(row) for row in [
        {"type":"task","instruction":"{}"},
        {"type":"tool_result","name":"read_file","ok":True,
            "result":{"path":"controller.py"}},
        {"type":"tool_result","name":"write_file","ok":True,
            "result":{"path":"diagnosis_iteration_006.md"}},
        {"type":"tool_result","name":"run_command","ok":True,
            "result":{"exit_code":0,"workspace_mutated_paths":["controller.py"],
                      "_embodied_codex_engineering_progress":True}},
    ])+"\n")

    handoff=EvolutionEngine._recover_uncommitted_pass_handoff(trace)

    assert handoff["error"]=="process_restarted_before_robot_episode"
    assert handoff["persisted_action_artifacts"]==["diagnosis_iteration_006.md"]
    assert handoff["recent_tool_results"][-1]["workspace_mutated_paths"]==["controller.py"]
    assert handoff["recovered_from_trace"]==str(trace)


def test_uncommitted_read_only_trace_does_not_skip_initial_diagnosis(tmp_path):
    trace=tmp_path/"trace.jsonl"
    trace.write_text(json.dumps({"type":"tool_result","name":"read_file","ok":True,
        "result":{"path":"controller.py"}})+"\n")
    assert EvolutionEngine._recover_uncommitted_pass_handoff(trace) is None


def test_uncommitted_recovery_keeps_short_command_correction_output(tmp_path):
    trace=tmp_path/"trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in [
        {"type":"tool_result","name":"replace_in_file","ok":True,
         "result":{"path":"controller.py","workspace_mutated_paths":["controller.py"]}},
        {"type":"tool_result","name":"run_command","ok":True,
         "result":{"argv":["grep","-n","relation","controller.py"],"exit_code":0,
                   "output":'88: "relation": "next_to"})\n'}},
    ])+"\n")

    handoff=EvolutionEngine._recover_uncommitted_pass_handoff(trace)

    command=handoff["recent_tool_results"][-1]
    assert command["argv"]==["grep","-n","relation","controller.py"]
    assert '"next_to"})' in command["command_output"]


def test_previous_outcome_surfaces_task_level_conflict_ahead_of_local_verifier():
    from embodied_codex.legacy.evolution import EvolutionEngine
    summary=EvolutionEngine._authoritative_outcome({
        "sensor_success_candidate":False,
        "controller_result":{"verified":True,"sensor_failure":False},
        "sensor_report":{"sensor_verification_passed":False,
            "controller_visual_verification_passed":True,
            "independent_task_outcome":{"verified":False,
                "source_relation_satisfied":False,
                "target_relation_satisfied":False,
                "contradiction":"The specified source remains in place.",
                "reason":"A different same-class object reached the target.",
                "confidence":0.99}}})
    assert summary["kernel_decision"]=="failure"
    assert summary["evidence_conflict"] is True
    assert summary["independent_task_level_outcome"]["source_relation_satisfied"] is False
    assert "cannot override" in summary["evidence_precedence"]


def test_correction_coding_passes_use_tighter_evidence_to_action_deadline():
    from embodied_codex.legacy.evolution import EvolutionEngine
    assert EvolutionEngine._coding_pass_limits(1)=={
        "max_evidence_deliveries":18,"post_evidence_pause_max_turns":4,
        "post_mutation_max_turns":8}
    assert EvolutionEngine._coding_pass_limits(2)=={
        "max_evidence_deliveries":6,"max_working_memory_deliveries":4,
        "post_evidence_pause_max_turns":2,"post_duplicate_read_max_turns":1,
        "post_mutation_max_turns":8}
    assert EvolutionEngine._coding_pass_limits(3)=={
        "max_evidence_deliveries":6,"max_working_memory_deliveries":4,
        "post_evidence_pause_max_turns":2,"post_duplicate_read_max_turns":1,
        "post_mutation_max_turns":8}


def test_coding_pass_handoff_is_compact_and_preserves_actionable_failure():
    from embodied_codex.legacy.evolution import EvolutionEngine
    handoff=EvolutionEngine._coding_pass_handoff({
        "completed":False,"error":"engineering action deadline",
        "tool_results":[
            {"name":"read_file","ok":True,"result":{"content":"x"*10000}},
            {"name":"run_robot_controller","ok":False,
             "error":"unchanged_controller_after_failed_episode"},
        ]})
    assert handoff=={
        "completed":False,"error":"engineering action deadline",
        "recent_tool_results":[
            {"name":"read_file","ok":True},
            {"name":"run_robot_controller","ok":False,
             "error":"unchanged_controller_after_failed_episode"},
        ]}


def test_coding_pass_handoff_preserves_bounded_command_correction_output():
    handoff=EvolutionEngine._coding_pass_handoff({
        "completed":False,"error":"correction pass ended",
        "tool_results":[{"name":"run_command","ok":True,"result":{
            "argv":["grep","-n","relation","controller.py"],
            "exit_code":0,
            "output":'88:        "target_candidates": plates, "relation": "next_to"})\n'}}]})

    row=handoff["recent_tool_results"][0]
    assert row["argv"]==["grep","-n","relation","controller.py"]
    assert '"relation": "next_to"})' in row["command_output"]
    assert len(json.dumps(handoff))<3000


def test_coding_pass_handoff_preserves_persisted_plan_without_bulk_content():
    from embodied_codex.legacy.evolution import EvolutionEngine
    handoff=EvolutionEngine._coding_pass_handoff({
        "completed":False,"error":"turn budget exhausted",
        "tool_results":[
            {"name":"write_file","ok":True,"result":{
                "path":"iteration_017_pre_run_hypotheses.md","bytes":2500}},
            {"name":"read_file","ok":True,"result":{
                "path":"controller.py","content":"x"*10000}},
            {"name":"replace_in_file","ok":True,"result":{
                "path":"controller.py","controller_semantic_progress":True}},
        ]})
    assert handoff["persisted_action_artifacts"]==[
        "iteration_017_pre_run_hypotheses.md"]
    assert "engineering TODO" in handoff["continuation_rule"]
    assert handoff["recent_tool_results"]==[
        {"name":"write_file","ok":True,
         "path":"iteration_017_pre_run_hypotheses.md"},
        {"name":"read_file","ok":True,"path":"controller.py"},
        {"name":"replace_in_file","ok":True,"path":"controller.py",
         "controller_semantic_progress":True},
    ]
    assert "x"*100 not in json.dumps(handoff)


def test_legacy_experience_is_dynamically_audited_from_copied_execution(tmp_path):
    evidence=tmp_path/"robot_execution.json"
    evidence.write_text(json.dumps({
        "sensor_success_candidate":False,
        "sensor_report":{"sensor_verification_passed":False,
            "independent_task_outcome":{"verified":False}}}))
    library=ExperienceLibrary(tmp_path/"experiences")
    registered=library.register(name="legacy_false_success",summary="Validated success.",
        applicability="All scenes.",keywords=["validated"],evidence_paths=[evidence])
    manifest_path=library._path(registered["experience_id"])/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["status"]="success_evidence"
    manifest["evidence_assessment"]={"outcome":"success","derived_by":"legacy_model"}
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")

    inspected=library.inspect(registered["experience_id"])
    assert inspected["status"]=="failure_evidence"
    assert inspected["evidence_assessment"]["outcome"]=="failure"
    assert inspected["manifest_claim"]["status"]=="success_evidence"
    assert inspected["manifest_claim_conflicts_with_evidence"] is True
    assert library.search("validated",limit=1)[0]["status"]=="failure_evidence"
    assert json.loads(manifest_path.read_text())["status"]=="success_evidence"


def test_legacy_gap_validation_is_dynamically_audited_in_search_and_inspect(tmp_path):
    evidence=tmp_path/"robot_execution.json"
    evidence.write_text(json.dumps({
        "sensor_success_candidate":False,
        "sensor_report":{"sensor_verification_passed":False,
            "independent_task_outcome":{"verified":False}}}))
    library=CapabilityGapLibrary(tmp_path/"gaps")
    registered=library.publish(name="legacy false validation",task="move object",
        failure_summary="The object did not move.",hypotheses=["wrong identity"],
        selected_diagnosis="grounding failed",required_capability={"kind":"grounding"},
        searched_candidates=[],provenance_decision={},integration_result={},
        task_validation={"status":"sensor_success","report":"verified=true"},
        reuse_evidence={},status="diagnosed",evidence_paths=[evidence])

    inspected=library.inspect(registered["gap_id"])
    assert inspected["authoritative_outcome"]=="failure"
    assert inspected["model_claim_conflicts_with_evidence"] is True
    assert inspected["task_validation"]["authoritative_outcome"]=="failure"
    searched=library.search("object grounding",limit=1)[0]
    assert searched["authoritative_outcome"]=="failure"
    assert searched["model_claim_conflicts_with_evidence"] is True


def test_unified_asset_retrieval_is_top_k_and_skill_packages_tool_manual(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    for index in range(12):
        source=workspace/f"tool_{index}.py";source.write_text("def run(payload): return payload\n")
        description=("articulated drawer handle motion planning" if index==7
                     else f"unrelated numeric utility {index}")
        tool_id=library.register_tool(name=f"retrieval_tool_{index}",source_path=source.name,
            description=description,input_schema={},output_schema={},source_urls=["https://example.org/test-algorithm"],
            trained_on_current_task=False)["tool_id"]
        library.test_tool(tool_id,[{"input":{},"expected":{}}])
    retrieved=library.search("open articulated drawer with handle",limit=3)
    assert len(retrieved)==3 and retrieved[0]["tool_id"]=="retrieval_tool_7:v001"

    controller=workspace/"controller.py";controller.write_text('''TOOL = "vision_adapter:v001"
NOTE = "vision_adapter:v001-extra"
def run(robot): return {}
''')
    proof=workspace/"robot_execution.json";proof.write_text('{"sensor_success":true}\n')
    skills=SkillLibrary(tmp_path/"skills")
    frozen=skills.freeze(name="drawer_skill",task="open an articulated drawer",
        controller=controller,evidence={"sensor_only":True},
        tool_ids=["retrieval_tool_7:v001"],tools=library,evidence_paths=[proof],task_model={
            "entities":[{"name":"drawer"}],
            "requirements":[{"id":"r1","description":"drawer is open"}],
            "phases":[{"observations":["RGB-D handle pose"],
                       "capability_requirements":["collision-aware planning"],
                       "required_robot_operations":["observe","act"]}]})
    skill_path=Path(frozen["path"])
    assert (skill_path/"tools"/"retrieval_tool_7_v001"/"manual.json").is_file()
    manifest=skills.inspect(frozen["skill_id"])
    assert manifest["interface"]["effects"]==["drawer is open"]
    assert len(manifest["evidence_files"])==1
    assert skills.search("drawer collision planning",limit=1)[0]["skill_id"]==frozen["skill_id"]
    skill_surface=EngineeringSurface(workspace=TaskWorkspace(tmp_path/"skill_run"/"workspace"),
        capabilities=library,runtime=object(),deployment_factory=lambda:None,
        artifact_dir=tmp_path/"skill_run"/"iterations"/"iteration_001",skills=skills)
    public_skill=skill_surface.inspect_skill(frozen["skill_id"])
    assert "development_evidence" not in public_skill
    assert public_skill["development_evidence_summary"]["sensor_only"] is True
    assert len(__import__("json").dumps(public_skill))<20_000
    source_page=skill_surface.read_skill_source(frozen["skill_id"])
    assert source_page["exists"] is True and "def run" in source_page["source"]["content"]
    checked_out=skill_surface.checkout_skill_controller(frozen["skill_id"])
    assert checked_out["source_skill_id"]==frozen["skill_id"]
    assert checked_out["source_controller_sha256"]==manifest["controller_sha256"]
    assert (tmp_path/"skill_run"/"workspace"/"controller.py").read_text()==controller.read_text()
    assert "checkout_skill_controller" in skill_surface.registry().items
    rebound_surface=EngineeringSurface(
        workspace=TaskWorkspace(tmp_path/"rebound_skill_run"/"workspace"),
        capabilities=library,runtime=object(),deployment_factory=lambda:None,
        artifact_dir=tmp_path/"rebound_skill_run"/"iterations"/"iteration_001",
        skills=skills,controller_tool_replacements={
            "vision_adapter:v001":"vision_adapter:v002"})
    rebound=rebound_surface.checkout_skill_controller(frozen["skill_id"])
    rebound_source=(tmp_path/"rebound_skill_run"/"workspace"/"controller.py").read_text()
    assert rebound["deployment_tool_constants_rebound"]==1
    assert rebound["source_controller_sha256"]==manifest["controller_sha256"]
    assert rebound["checked_out_controller_sha256"]!=manifest["controller_sha256"]
    assert 'TOOL = "vision_adapter:v002"' in rebound_source
    assert 'NOTE = "vision_adapter:v001-extra"' in rebound_source
    missing=skill_surface.read_skill_source("none")
    assert missing["exists"] is False and "search_assets" in missing["instruction"]
    skill_evidence=public_skill["evidence_files"][0]
    assert "original_path" not in skill_evidence
    assert '"sensor_success":true' in skill_surface.read_run_artifact(
        skill_evidence["asset_ref"])["content"]

    duplicate=skills.freeze(name="drawer_skill",task="open an articulated drawer",
        controller=controller,evidence={"sensor_only":True},
        tool_ids=["retrieval_tool_7:v001"],tools=library,evidence_paths=[proof],task_model={
            "entities":[{"name":"drawer"}],
            "requirements":[{"id":"r1","description":"drawer is open"}],
            "phases":[{"observations":["RGB-D handle pose"],
                       "capability_requirements":["collision-aware planning"],
                       "required_robot_operations":["observe","act"]}]})
    assert duplicate["duplicate_of"]==frozen["skill_id"]
    frozen_proof=Path(frozen["path"])/manifest["evidence_files"][0]["path"]
    frozen_proof.write_text("tampered\n")
    with pytest.raises(AssetError,match="evidence hash mismatch"):
        skills.inspect(frozen["skill_id"])


def test_skill_freeze_rejects_untested_dependency_and_leaves_no_partial_version(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    source=workspace/"candidate.py";source.write_text("def run(payload): return payload\n")
    controller=workspace/"controller.py";controller.write_text("def run(robot): return {}\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace,python=PYTHON)
    tool_id=library.register_tool(name="untested_candidate",source_path=source.name,
        description="candidate",input_schema={},output_schema={},
        source_urls=["https://example.org/candidate"],trained_on_current_task=False)["tool_id"]
    skills=SkillLibrary(tmp_path/"skills")
    with pytest.raises(AssetError,match="not tested"):
        skills.freeze(name="invalid_skill",task="test",controller=controller,
            evidence={"sensor_only":True},tool_ids=[tool_id],tools=library)
    assert not list((tmp_path/"skills"/"invalid_skill").glob("v*/manifest.json"))
def test_pre_experience_run_configuration_migrates_to_shared_library(tmp_path):
    import json
    root=tmp_path/"legacy";root.mkdir();capabilities=tmp_path/"capabilities"
    (root/"harness_configuration.json").write_text(json.dumps({
        "protocol":"embodied-codex-run-configuration-v1",
        "capability_root":str(capabilities.resolve())})+"\n")
    shared=tmp_path/"shared_experiences"
    engine=EvolutionEngine(root=root,model=object(),deployment_factory=lambda:None,
        capability_root=capabilities,experience_root=shared)
    migrated=json.loads((root/"harness_configuration.json").read_text())
    assert migrated["experience_root"]==str(shared.resolve())
    assert engine.experience_root==shared.resolve()


def test_single_sdk_contract_matches_adapter_and_all_examples_validate():
    import ast
    import inspect
    import textwrap
    from embodied_codex.deployments.libero import LiberoDeployment
    from embodied_codex.adapters.libero_sdk import (LIBERO_ROBOT_SDK_CONTRACT,
        SDKContractError,validate_action,validate_verifier_request)

    tree=ast.parse(textwrap.dedent(inspect.getsource(LiberoDeployment._act)))
    implemented=set()
    for node in ast.walk(tree):
        if (isinstance(node,ast.Compare) and isinstance(node.left,ast.Name)
                and node.left.id=="kind" and len(node.ops)==1
                and isinstance(node.ops[0],ast.Eq) and len(node.comparators)==1
                and isinstance(node.comparators[0],ast.Constant)
                and isinstance(node.comparators[0].value,str)):
            implemented.add(node.comparators[0].value)
    assert implemented==set(LIBERO_ROBOT_SDK_CONTRACT["actions"])
    proprio=LIBERO_ROBOT_SDK_CONTRACT["methods"]["observe"][
        "returns_by_channel"]["proprioception"]
    assert proprio["shape"]=="{step, proprioception}"
    assert "['proprioception']['robot0_eef_quat']" in proprio["example"]
    for name,contract in LIBERO_ROBOT_SDK_CONTRACT["actions"].items():
        assert validate_action(contract["example"])==name
    for name,contract in LIBERO_ROBOT_SDK_CONTRACT["verifiers"].items():
        validate_verifier_request(name,contract["example"])
    assert validate_action({"type":"move_to_pose","target_ref":"point-live",
                            "rotation_matrix":[[1,0,0],[0,1,0],[0,0,1]]})=="move_to_pose"
    with pytest.raises(SDKContractError,match="requires one of field sets"):
        validate_action({"type":"move_to_pose","target_ref":"point-live"})


def test_sdk_linter_rejects_missing_required_literal_fields(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.act({"type":"move_to_pose", "offset":[0,0,0]})
    return robot.verify("visual_attachment", {"frame":{}, "source_ref":"s"})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match="requires one of literal field sets.*pose_ref.*object_query"):
        surface._lint_robot_contract(workspace._path("controller.py"))


@pytest.mark.parametrize("controller, message", [
    ('''def run(robot):
    robot.act({"type":"move_pose", "pose_ref":"pose-1"})
''', "unsupported literal action type.*move_pose"),
    ('''def run(robot):
    frame = robot.observe("rgbd", {})
    return robot.verify("visual_support_relation", {
        "frame": frame, "object_query": "bowl", "target_query": "plate",
        "source_ref": "source-1"
    })
''', "missing literal fields.*target_ref"),
])
def test_contract_failure_never_constructs_robot_deployment(tmp_path, controller, message):
    """Typed SDK mistakes are compilation failures, not robot experiments."""
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT

    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",controller)
    factory_calls=[]
    def forbidden_factory():
        factory_calls.append(True)
        raise AssertionError("deployment must not be constructed after contract failure")
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root),
        runtime=ControllerRuntime(python=PYTHON,timeout_seconds=5),
        deployment_factory=forbidden_factory,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match=message):
        surface.run_robot_controller("controller.py")
    assert factory_calls==[]
    assert surface.robot_runs==0


def test_robot_execution_tool_disappears_after_one_episode_lifecycle(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root),
        runtime=object(),deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001")
    registry=surface.registry()
    names=lambda:{row["function"]["name"] for row in registry.schemas}
    assert "run_robot_controller" in names()
    surface.robot_runs=1
    assert "run_robot_controller" not in names()
    with pytest.raises(KeyError,match="lifecycle phase"):
        registry.invoke("run_robot_controller",{"path":"controller.py"})


def test_complete_literal_libero_sdk_controller_passes_lint_and_executes(tmp_path):
    """A GPT-style controller consumes the exact public SDK without aliases."""
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT

    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    before = robot.observe("rgbd", {})
    live = robot.use("live_refs:v001", {"frame":before})
    pose_ref = live["pose_ref"]
    source_ref = live["source_ref"]
    robot.act({"type":"gripper", "command":"open", "repeat":2})
    robot.act({"type":"move_to_pose", "pose_ref":pose_ref, "gripper":-1})
    robot.act({"type":"gripper", "command":"close", "repeat":2})
    after = robot.observe("rgbd", {})
    proof = robot.verify("visual_attachment", {
        "frame":after, "object_query":"bowl", "source_ref":source_ref
    })
    return {"status":"sensor_success" if proof["verified"] else "sensor_failure"}
''')
    class ContractDeployment:
        instruction="pick up the bowl"
        def __init__(self):self.closed=False
        def dispatch(self,method,arguments):
            if method=="observe":return {"frame_id":"frame-1"}
            if method=="use":return {"tool_id":arguments["tool_id"],"result":{
                "pose_ref":"pose-1","source_ref":"source-1",
                "dense_sensor_payload":list(range(20000))}}
            if method=="act":return {"type":arguments["action"]["type"],"reached":True}
            if method=="verify":return {"verified":True}
            if method=="record":return {"recorded":True}
            raise AssertionError(method)
        def sensor_report(self,execution):
            return {"sensor_verification_passed":True,"benchmark_signal_exposed":False}
        def project_rpc_output(self,method,arguments,result):return dict(result)
        def close(self):self.closed=True
    deployment=ContractDeployment()
    surface=EngineeringSurface(workspace=workspace,
        capabilities=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root),
        runtime=ControllerRuntime(python=PYTHON,timeout_seconds=5),
        deployment_factory=lambda:deployment,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    result=surface.run_robot_controller("controller.py")
    assert result["robot_contract_preflight"]["passed"] is True
    assert result["sensor_success_candidate"] is True
    assert deployment.closed is True
    detail=Path(result["full_execution_artifact"])
    assert detail.is_file()
    import json
    assert detail.stat().st_size>10*len(json.dumps(result))
    use_event=next(event for event in result["rpc_evidence"] if event["method"]=="use")
    assert use_event["output_shape"]["dense_sensor_payload"]["length"]==20000


def test_contract_linter_rejects_fabricated_opaque_reference_literals(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.act({"type":"move_to_pose", "pose_ref":"made-up-pose"})
    return robot.verify("visual_attachment", {
        "frame":{}, "object_query":"bowl", "source_ref":"unavailable"
    })
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match="pose_ref cannot be a literal.*source_ref cannot be a literal"):
        surface._lint_robot_contract(workspace._path("controller.py"))


def test_contract_linter_rejects_unresolved_runtime_name_before_robot_episode(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    event = {"phase": "attempt"}
    history.append(event)
    return robot.record(event)
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match=r"unresolved runtime names.*history"):
        surface._lint_robot_contract(workspace._path("controller.py"))


def test_contract_linter_allows_imports_builtins_closures_and_initialized_names(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''import math

def run(robot):
    history = []
    def remember(value):
        history.append(math.floor(value))
    remember(len(history))
    return robot.record({"history": history})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    assert surface._lint_robot_contract(workspace._path("controller.py"))["passed"] is True


def test_contract_linter_rejects_branch_local_read_before_assignment(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    if robot.instruction:
        target = {"name": "plate"}
    return robot.record({"target": target})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match=r"local 'target' may be read before assignment"):
        surface._lint_robot_contract(workspace._path("controller.py"))


def test_contract_linter_accepts_locals_assigned_on_every_continuing_branch(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    if robot.instruction:
        target = {"name": "plate"}
    else:
        target = {"name": "fallback"}
    history = []
    for item in [target]:
        history.append(item)
    return robot.record({"history": history, "target": target})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    assert surface._lint_robot_contract(workspace._path("controller.py"))["passed"] is True


def test_contract_linter_understands_nonempty_loop_assignment_and_early_break(tmp_path):
    from embodied_codex.adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    workspace.write_file("controller.py",'''def run(robot):
    for attempt in range(3):
        fresh = robot.observe(channel="rgbd", request={})
    return robot.record({"fresh": fresh})
''')
    assert surface._lint_robot_contract(workspace._path("controller.py"))["passed"] is True
    workspace.write_file("controller.py",'''def run(robot):
    for attempt in range(3):
        if robot.instruction:
            break
        target = {"name": "plate"}
    return robot.record({"target": target})
''')
    with pytest.raises(RuntimeError,match=r"local 'target' may be read before assignment"):
        surface._lint_robot_contract(workspace._path("controller.py"))


def test_libero_sealed_evaluator_is_one_shot_and_closes_controller_io():
    from embodied_codex.deployments.libero import LiberoDeployment,LiberoDeploymentError

    class FakeEnv:
        def __init__(self):self.calls=0
        def check_success(self):self.calls+=1;return True

    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed=False;deployment._controller_execution_sealed=False
    deployment._evaluator_calls=0;deployment.env=FakeEnv()
    with pytest.raises(LiberoDeploymentError,match="not sealed"):
        deployment._sealed_check_once()
    deployment.seal_controller_execution()
    with pytest.raises(LiberoDeploymentError,match="already sealed"):
        deployment.dispatch("record",{"event":{}})
    assert deployment._sealed_check_once() is True
    assert deployment.env.calls==1
    with pytest.raises(LiberoDeploymentError,match="already consumed"):
        deployment._sealed_check_once()


def test_task_local_tool_requires_robot_evidence_before_cross_run_visibility(tmp_path):
    shared=tmp_path/"shared_capabilities"
    first=EvolutionEngine(root=tmp_path/"run_a",model=object(),
        deployment_factory=lambda:None,python=PYTHON,capability_root=shared)
    first.workspace.write_file("reusable.py","def run(payload): return {'x': payload['x']}\n")
    tool_id=first.capabilities.register_tool(name="reusable_identity",
        source_path="reusable.py",description="shared tested identity",
        input_schema={"type":"object","properties":{"x":{"type":"number"}},"required":["x"]},
        output_schema={"type":"object","properties":{"x":{"type":"number"}},"required":["x"]},
        source_urls=["https://example.org/public-algorithm"],trained_on_current_task=False)["tool_id"]
    assert first.capabilities.test_tool(tool_id,[{"input":{"x":3},
                                                  "expected":{"x":3}}])["status"]=="tested"

    second=EvolutionEngine(root=tmp_path/"run_b",model=object(),
        deployment_factory=lambda:None,python=PYTHON,capability_root=shared)
    assert second.capabilities.list_summaries()==[]
    controller_sha="a"*64
    evidence=tmp_path/"run_a"/"iterations"/"iteration_001"/"robot_execution.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps({
        "sensor_success_candidate":True,
        "execution":{"program_sha256":controller_sha,"rpc_events":[{
            "method":"use","arguments":{"tool_id":tool_id}}]},
        "sensor_report":{"sensor_verification_passed":True,
                         "_harness_case_id":"opaque-development-a"}},indent=2)+"\n")
    promoted=first.capabilities.promote_for_reuse(tool_id,
        evidence_paths=[evidence],controller_sha256=controller_sha,
        required_case_handles=["opaque-development-a"])
    assert promoted["visibility"]=="shared"
    assert [row["tool_id"] for row in second.capabilities.list_summaries()]==[tool_id]
    for run in (tmp_path/"run_a",tmp_path/"run_b"):
        configuration=__import__("json").loads((run/"harness_configuration.json").read_text())
        assert Path(configuration["capability_root"])==shared.resolve()
    with pytest.raises(RuntimeError,match="capability library mismatch"):
        EvolutionEngine(root=tmp_path/"run_b",model=object(),
            deployment_factory=lambda:None,python=PYTHON,
            capability_root=tmp_path/"different_capabilities")


def test_resumed_controller_structurally_rebinds_updated_deployment_tool(tmp_path):
    engine=EvolutionEngine(root=tmp_path/"run",model=object(),
        deployment_factory=lambda:None,python=PYTHON)
    original='''TOOL = "vision_adapter:v001"
NOTE = "vision_adapter:v001-extra"
# Formatting and comments are immutable across Adapter dependency binding.
def run(robot): return robot.use(TOOL, {})
'''
    engine.workspace.write_file("controller.py",original)
    first=engine.bind_current_deployment_tools({"vision_adapter":"vision_adapter:v001"})
    assert first["changed_constants"]==0
    second=engine.bind_current_deployment_tools({"vision_adapter":"vision_adapter:v002"})
    source=(engine.workspace.root/"controller.py").read_text()
    assert second["changed_constants"]==1
    assert source==original.replace('"vision_adapter:v001"','"vision_adapter:v002"')
    assert 'TOOL = "vision_adapter:v002"' in source
    assert 'NOTE = "vision_adapter:v001-extra"' in source
    ledger=__import__("json").loads((tmp_path/"run"/"deployment_bindings.json").read_text())
    assert ledger["current"]=={"vision_adapter":"vision_adapter:v002"}
    assert ledger["compatible_replacements"]=={
        "vision_adapter:v001":"vision_adapter:v002"}
    assert len(ledger["history"])==1
    # The migration remains available on later process starts, including after
    # an immutable old Controller snapshot is restored for causal replay.
    engine.workspace.write_file("controller.py",'''TOOL = "vision_adapter:v001"
def run(robot): return robot.use(TOOL, {})
''')
    resumed=engine.bind_current_deployment_tools({"vision_adapter":"vision_adapter:v002"})
    assert resumed["changed_constants"]==1
    assert resumed["replacements"]=={"vision_adapter:v001":"vision_adapter:v002"}


def test_libero_matrix_continues_after_task_failure_and_separates_metrics(tmp_path):
    import argparse
    import subprocess
    from embodied_codex.examples.run_libero_conformance_matrix import run_matrix

    args=argparse.Namespace(output_dir=str(tmp_path/"matrix"),suite="libero_spatial",
        tasks=[0,1],states=[2],max_iterations_per_task=1,capability_library=None,
        device="cpu",model="test-model",reasoning_effort="low",
        base_url="https://example.invalid/v1",config="config/standalone_libero",
        retry_locked_validation=False)
    calls=[]
    def fake_runner(command,**unused):
        calls.append(command)
        task=int(command[command.index("--task")+1])
        run=Path(command[command.index("--run-dir")+1]);run.mkdir(parents=True,exist_ok=True)
        (run/"state.json").write_text(__import__("json").dumps({
            "task":f"task {task}","status":"evolving","iterations":[]}))
        return subprocess.CompletedProcess(command,2 if task==0 else 1)
    summary=run_matrix(args,runner=fake_runner)
    assert len(calls)==2 and len(summary["runs"])==2
    assert summary["metrics"]["tasks_attempted"]==2
    # rc=2 is a bounded physical non-success; rc=1 is infrastructure failure.
    assert summary["metrics"]["process_failures"]==1
    assert summary["metrics"]["sensor_task_successes"]==0
    assert (tmp_path/"matrix"/"matrix_summary.json").is_file()
    roots={command[command.index("--capability-library")+1] for command in calls}
    assert roots=={str((tmp_path/"matrix"/"shared_capabilities").resolve())}


def test_capability_gap_lifecycle_is_immutable_searchable_and_portable(tmp_path):
    evidence=tmp_path/"robot_execution.json";evidence.write_text('{"collision": true}\n')
    library=CapabilityGapLibrary(tmp_path/"gaps")
    first=library.publish(name="drawer collision",task="retrieve the bowl",
        failure_summary="EEF stops at the drawer lip before contact",
        hypotheses=["straight-line approach intersects cabinet geometry"],
        selected_diagnosis="collision-aware approach is missing",
        required_capability={"kind":"motion_planner","inputs":["RGB-D","EEF pose"]},
        searched_candidates=[],provenance_decision={},integration_result={},
        task_validation={},reuse_evidence={},status="diagnosed",
        evidence_paths=[evidence])
    second=library.publish(name="drawer collision",task="retrieve the bowl",
        failure_summary="EEF stops at the drawer lip before contact",
        hypotheses=["straight-line approach intersects cabinet geometry"],
        selected_diagnosis="collision-aware approach is missing",
        required_capability={"kind":"motion_planner","inputs":["RGB-D","EEF pose"]},
        searched_candidates=[{"name":"OMPL","url":"https://ompl.kavrakilab.org/"}],
        provenance_decision={"accepted":"OMPL","task_trained":False},
        integration_result={"tool_id":"ompl_bridge:v001","unit_tests":True},
        task_validation={"rollout":"pending"},reuse_evidence={},status="integrating",
        evidence_paths=[evidence],previous_gap_id=first["gap_id"])
    evidence.unlink()
    assert library.inspect(first["gap_id"])["status"]=="diagnosed"
    current=library.inspect(second["gap_id"])
    assert current["previous_gap_id"]==first["gap_id"]
    assert current["lineage_root_gap_id"]==first["gap_id"]
    assert library.search("collision motion planner",limit=1)[0]["gap_id"] in {
        first["gap_id"],second["gap_id"]}
    assert library.search("collision motion planner",limit=8)[0]["gap_id"]==second["gap_id"]
    with pytest.raises(AssetError,match="extend latest revision"):
        library.publish(name="drawer collision",task="retrieve the bowl",
            failure_summary="stale branch",hypotheses=["x"],selected_diagnosis="x",
            required_capability={"kind":"planner"},searched_candidates=[{"name":"x"}],
            provenance_decision={"accepted":True},integration_result={"tool":"x"},
            task_validation={},reuse_evidence={},status="integrating",
            evidence_paths=[second and library._path(second["gap_id"])],
            previous_gap_id=first["gap_id"])
    with pytest.raises(AssetError,match="requires recorded search candidates"):
        CapabilityGapLibrary(tmp_path/"bad_gaps").publish(name="fake integration",task="t",
            failure_summary="f",hypotheses=["h"],selected_diagnosis="d",
            required_capability={"kind":"planner"},searched_candidates=[],
            provenance_decision={"accepted":True},integration_result={"tool":"x"},
            task_validation={},reuse_evidence={},status="integrating",
            evidence_paths=[library._path(second["gap_id"])])
    with pytest.raises(AssetError,match="requires evidence"):
        library.publish(name="empty evidence",task="t",failure_summary="f",
            hypotheses=[],selected_diagnosis="",required_capability={},
            searched_candidates=[],provenance_decision={},integration_result={},
            task_validation={},reuse_evidence={},status="observed",evidence_paths=[])


def test_engineering_surface_exposes_gap_as_retrieved_asset(tmp_path):
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    artifact=run/"iterations"/"iteration_001";artifact.mkdir(parents=True)
    evidence=artifact/"robot_execution.json";evidence.write_text('{"failed": true}\n')
    gaps=CapabilityGapLibrary(run/"gaps")
    tools=CapabilityLibrary(run/"tools",workspace.root,python=PYTHON)
    workspace.write_file("planner.py","def run(payload): return {'path': []}\n")
    registered=tools.register_tool(name="planner",source_path="planner.py",
        description="deterministic planner",input_schema={"type":"object"},
        output_schema={"type":"object"},source_urls=["https://example.org/planner"],
        trained_on_current_task=False)
    surface=EngineeringSurface(workspace=workspace,
        capabilities=tools,
        runtime=ControllerRuntime(python=PYTHON),deployment_factory=lambda:None,
        artifact_dir=artifact,gaps=gaps)
    result=surface.record_capability_gap(name="missing planner",task="drawer",
        failure_summary="collision",hypotheses=["no obstacle avoidance"],
        selected_diagnosis="planner missing",required_capability={"kind":"planner"},
        searched_candidates=[],provenance_decision={},integration_result={},
        task_validation={},reuse_evidence={},status="diagnosed")
    (artifact/"controller.py").write_text("def run(robot): return {'sensor_failure':'collision'}\n")
    partial=surface.revise_capability_gap(result["gap_id"],
        status="integrating",integration_result={"controller":"draft"})
    assert gaps.inspect(partial["gap_id"])["status"]=="diagnosed"
    searched=surface.revise_capability_gap(partial["gap_id"],
        status="integrating",
        searched_candidates=[{"name":"OMPL","url":"https://ompl.kavrakilab.org/"}],
    )
    assert gaps.inspect(searched["gap_id"])["status"]=="searching"
    revised=surface.revise_capability_gap(searched["gap_id"],
        status="integrating",
        provenance_decision={"accepted":"OMPL"},
        integration_result={"controller":"executed_controller"},
        evidence_refs=["latest_robot_execution","controller.py",registered["tool_id"]])
    revision=gaps.inspect(revised["gap_id"])
    assert revision["task"]=="drawer" and revision["status"]=="integrating"
    assert revision["previous_gap_id"]==searched["gap_id"]
    assert len(revision["evidence"])==3
    validated=surface.revise_capability_gap(revised["gap_id"],
        status="validated",task_validation={"passed":True})
    assert gaps.inspect(validated["gap_id"])["status"]=="validated"
    found=surface.search_assets("planner collision",["gap"],8)
    assert found["gaps"][0]["gap_id"]==validated["gap_id"]
    registry=surface.registry()
    assert {"record_capability_gap","revise_capability_gap"} <= set(registry.items)
    create_schema=registry.items["record_capability_gap"].parameters
    revise_schema=registry.items["revise_capability_gap"].parameters
    assert set(create_schema["required"])=={"name","task","failure_summary"}
    assert "evidence_refs" not in create_schema["required"]
    assert revise_schema["required"]==["previous_gap_id"]
    assert "changes" not in revise_schema["properties"]
    assert "evidence_refs" not in revise_schema["required"]


def test_retrieved_gap_can_be_revised_before_new_run_has_robot_evidence(tmp_path):
    shared=tmp_path/"shared";old_run=tmp_path/"old_run"
    old_evidence=old_run/"iterations"/"iteration_001"/"robot_execution.json"
    old_evidence.parent.mkdir(parents=True);old_evidence.write_text('{"failed": true}')
    gaps=CapabilityGapLibrary(shared/"gaps")
    first=gaps.publish(name="shared gap",task="pick",failure_summary="empty grasp",
        hypotheses=["bad contact"],selected_diagnosis="contact failed",
        required_capability={"kind":"adaptive grasp"},searched_candidates=[],
        provenance_decision={},integration_result={},task_validation={},reuse_evidence={},
        status="diagnosed",evidence_paths=[old_evidence])
    new_run=tmp_path/"new_run";workspace=TaskWorkspace(new_run/"workspace")
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,
        artifact_dir=new_run/"iterations"/"iteration_001",gaps=gaps)
    revised=surface.revise_capability_gap(first["gap_id"],status="searching",
        searched_candidates=[{"name":"public planner"}])
    manifest=gaps.inspect(revised["gap_id"])
    assert manifest["status"]=="searching" and len(manifest["evidence"])==1
    copied=gaps._path(revised["gap_id"]).parent/manifest["evidence"][0]["path"]
    assert copied.read_bytes()==old_evidence.read_bytes()


def test_persistent_diagnosed_gap_requires_audited_acquisition_before_rollout(tmp_path):
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    artifact=run/"iterations"/"iteration_003";artifact.mkdir(parents=True)
    evidence=run/"iterations"/"iteration_002"/"robot_execution.json"
    evidence.parent.mkdir(parents=True);evidence.write_text('{"failed": true}\n')
    gaps=CapabilityGapLibrary(run/"gaps")
    first=gaps.publish(name="persistent grasp gap",task="pick bowl",
        failure_summary="first empty grasp",hypotheses=["bad contact"],
        selected_diagnosis="contact family failed",
        required_capability={"kind":"adaptive grasp"},searched_candidates=[],
        provenance_decision={},integration_result={},task_validation={},reuse_evidence={},
        status="diagnosed",evidence_paths=[evidence])
    second=gaps.publish(name="persistent grasp gap",task="pick bowl",
        failure_summary="second grasp family also failed",hypotheses=["bad contact span"],
        selected_diagnosis="available grasp families failed",
        required_capability={"kind":"adaptive grasp"},searched_candidates=[],
        provenance_decision={},integration_result={},task_validation={},reuse_evidence={},
        status="diagnosed",evidence_paths=[evidence],previous_gap_id=first["gap_id"])
    workspace.write_file("controller.py","def run(robot): return {'sensor_failure':'empty'}\n")
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=artifact,gaps=gaps,
        required_acquisition_gap_id=second["gap_id"],acquisition_baseline_tool_ids=[])
    with pytest.raises(RuntimeError,match="capability_acquisition_required"):
        surface._require_capability_acquisition(workspace.root/"controller.py")

    surface._record_research("search",{"query":"public antipodal grasp planner","results":[
        {"url":"https://example.org/planner","title":"planner","source":"test"}]})
    searched=gaps.publish(name="persistent grasp gap",task="pick bowl",
        failure_summary="public candidates found",hypotheses=["antipodal contact may help"],
        selected_diagnosis="available grasp families failed",
        required_capability={"kind":"adaptive grasp"},searched_candidates=[{
            "name":"public planner","url":"https://example.org/planner",
            "query":"public antipodal grasp planner"}],provenance_decision={},
        integration_result={},task_validation={},reuse_evidence={},status="searching",
        evidence_paths=[surface.research_ledger_path],previous_gap_id=second["gap_id"])
    assert searched["status"]=="searching"
    with pytest.raises(RuntimeError,match="does not by itself authorize"):
        surface._require_capability_acquisition(workspace.root/"controller.py")


def test_persistent_gap_gate_accepts_new_tested_tool_binding(tmp_path):
    class TestedToolLibrary:
        def inspect(self,tool_id):
            if tool_id!="new_grasp_tool:v001":raise FileNotFoundError(tool_id)
            return {"manifest":{"tool_id":tool_id,"status":"tested"},
                    "source":"def run(payload): return {'grasp': True}\n"}
    class GapLibrary:
        value={"gap_id":"grasp_gap:v002","name":"grasp_gap",
               "required_capability":{"kind":"adaptive grasp"},"status":"diagnosed"}
        def inspect(self,unused):return dict(self.value)
        def latest_for_name(self,unused):return dict(self.value)
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    controller=workspace.root/"controller.py"
    workspace.write_file("controller.py",'''GRASP_TOOL = "new_grasp_tool:v001"
def run(robot):
    return robot.use(GRASP_TOOL, {})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=TestedToolLibrary(),
        runtime=object(),deployment_factory=lambda:None,
        artifact_dir=run/"iterations"/"iteration_003",gaps=GapLibrary(),
        required_acquisition_gap_id="grasp_gap:v002",
        acquisition_baseline_tool_ids=["old_grasp_tool:v001"])
    result=surface._require_capability_acquisition(controller)
    assert result["review"]["approved"] is True


def test_persistent_gap_gate_rejects_tool_that_does_not_satisfy_gap(tmp_path):
    class TestedToolLibrary:
        def inspect(self,tool_id):
            return {"manifest":{"tool_id":tool_id,"status":"tested",
                    "description":"static offsets"},
                    "source":"def run(payload): return {'offsets': [[.04, 0]]}\n"}
    class GapLibrary:
        value={"gap_id":"visual_servo_gap:v003","name":"visual_servo_gap",
               "required_capability":{"kind":"closed-loop visual servo",
                    "inputs":["fresh RGB-D after every correction"]},"status":"integrating"}
        def inspect(self,unused):return dict(self.value)
        def latest_for_name(self,unused):return dict(self.value)
    run=tmp_path/"run";workspace=TaskWorkspace(run/"workspace")
    controller=workspace.root/"controller.py"
    workspace.write_file("controller.py",'''def run(robot):
    return robot.use("static_offsets:v001", {})
''')
    reviewed=[]
    def reviewer(**payload):
        reviewed.append(payload)
        return {"approved":False,"approved_tool_ids":[],"covered_requirements":[],
                "issues":["static offsets do not consume fresh RGB-D feedback"]}
    surface=EngineeringSurface(workspace=workspace,capabilities=TestedToolLibrary(),
        runtime=object(),deployment_factory=lambda:None,
        artifact_dir=run/"iterations"/"iteration_004",gaps=GapLibrary(),
        acquisition_reviewer=reviewer,
        required_acquisition_gap_id="visual_servo_gap:v003",
        acquisition_baseline_tool_ids=[])
    with pytest.raises(RuntimeError,match="static offsets do not consume fresh RGB-D"):
        surface._require_capability_acquisition(controller)
    assert reviewed and reviewed[0]["gap"]["gap_id"]=="visual_servo_gap:v003"
    binding=json.loads((workspace.root/"capability_integration_binding.json").read_text())
    assert binding["review"]["approved"] is False


def test_record_capability_gap_upserts_latest_same_name_revision(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    (artifact/"robot_execution.json").write_text('{"failed": true}')
    gaps=CapabilityGapLibrary(run/"gaps")
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact,gaps=gaps)
    first=surface.record_capability_gap(name="grasp reliability",task="pick",
        failure_summary="empty grasp",selected_diagnosis="contact failed",
        required_capability={"kind":"adaptive grasp"},status="diagnosed")
    second=surface.record_capability_gap(name="Grasp Reliability",task="pick",
        failure_summary="second empty grasp",
        searched_candidates=[{"name":"public grasp planner"}],status="searching")
    assert second["upserted_existing_gap"] is True
    assert second["upserted_from_gap_id"]==first["gap_id"]
    manifest=gaps.inspect(second["gap_id"])
    assert manifest["previous_gap_id"]==first["gap_id"]
    assert manifest["failure_summary"]=="second empty grasp"
    assert manifest["selected_diagnosis"]=="contact failed"
    assert manifest["status"]=="searching"


def test_changed_gap_diagnosis_invalidates_inherited_downstream_claims(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    (artifact/"robot_execution.json").write_text(json.dumps({
        "sensor_success_candidate":False,
        "sensor_report":{"sensor_verification_passed":False,
            "independent_task_outcome":{"verified":False}}}))
    gaps=CapabilityGapLibrary(run/"gaps")
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact,gaps=gaps)
    first=surface.record_capability_gap(name="identity gap",task="pick",
        failure_summary="wrong object",hypotheses=["appearance ambiguity"],
        selected_diagnosis="appearance query is weak",
        required_capability={"kind":"appearance detector"},
        searched_candidates=[{"name":"detector"}],
        provenance_decision={"decision":"reuse"},
        integration_result={"controller_change":"old appearance rule"},
        task_validation={"status":"success"},status="integrating")
    revised=surface.revise_capability_gap(first["gap_id"],status="diagnosed",
        selected_diagnosis="support relation is required",
        required_capability={"kind":"relation grounder"})
    item=gaps.inspect(revised["gap_id"])
    assert item["searched_candidates"]==[]
    assert item["provenance_decision"]=={}
    assert item["integration_result"]=={}
    assert item["reuse_evidence"]=={}
    assert item["task_validation"]["authoritative_outcome"]=="failure"
    assert "status" not in item["task_validation"]


def test_gap_evidence_asset_refs_and_relative_upsert_are_hash_scoped(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    execution=artifact/"robot_execution.json";execution.write_text('{"failed": true}')
    gaps=CapabilityGapLibrary(run/"gaps")
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact,gaps=gaps)
    first=surface.record_capability_gap(name="contact gap",task="pick",
        failure_summary="empty grasp")
    inspected=surface.inspect_capability_gap(first["gap_id"])
    evidence=inspected["evidence"][0]
    assert evidence["asset_ref"]==f"{first['gap_id']}#{evidence['path']}"
    assert surface._authorized_evidence_path(evidence["asset_ref"]).read_bytes()==execution.read_bytes()
    execution.unlink()
    second=surface.record_capability_gap(name="contact gap",task="pick",
        failure_summary="still empty",evidence_refs=[evidence["path"]])
    assert second["upserted_existing_gap"] is True
    assert len(gaps.inspect(second["gap_id"])["evidence"])==1
    with pytest.raises((RuntimeError,FileNotFoundError)):
        surface._authorized_evidence_path(f"{first['gap_id']}#evidence/not-owned.json")


def test_provenance_gate_rejects_unverifiable_learned_asset(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"model.py").write_text("def run(payload): return {}\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace)
    with pytest.raises(AssetError,match="training-data declaration"):
        library.register_tool(name="unknown_model",source_path="model.py",
            description="unknown learned model",input_schema={"type":"object"},
            output_schema={"type":"object"},source_urls=["https://example.org/model"],
            trained_on_current_task=False,provenance={"models":["unknown"]})
    result=library.register_tool(name="audited_algorithm",source_path="model.py",
        description="nonlearned public algorithm",input_schema={"type":"object"},
        output_schema={"type":"object"},source_urls=["https://example.org/algorithm"],
        trained_on_current_task=False,provenance={
            "training_data_declaration":"No learned parameters.",
            "contamination_check":{"evaluated_benchmark":"LIBERO",
                "method":"source inspection","result":"not_applicable_source_code"}})
    provenance=library.inspect(result["tool_id"])["manifest"]["provenance"]
    assert provenance["audit_status"]=="complete" and len(provenance["audit_sha256"])==64


def test_agent_asset_registration_requires_hashed_research_evidence(tmp_path,monkeypatch):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001";artifact.mkdir(parents=True)
    workspace=TaskWorkspace(run/"workspace")
    workspace.write_file("normalize.py","def run(payload): return payload\n")
    library=CapabilityLibrary(run/"tools",workspace.root,python=PYTHON)
    gaps=CapabilityGapLibrary(run/"gaps")
    surface=EngineeringSurface(workspace=workspace,capabilities=library,
        runtime=object(),deployment_factory=lambda:None,artifact_dir=artifact,gaps=gaps)
    observed="https://example.org/public-algorithm"
    monkeypatch.setattr("embodied_codex.legacy.engineering.search_web",lambda query,limit:{
        "query":query,"provider":"fixture","results":[{
            "title":"Public algorithm","url":observed,"source":"fixture"}],
        "provider_errors":[]})
    result=surface.search_web("normalization algorithm",5)
    assert len(result["research_record_sha256"])==64
    sources=surface.list_research_sources()
    assert sources["count"]==1 and sources["sources"][0]["url"]==observed
    assert sources["returned_count"]==1 and sources["next_offset"] is None
    common={"name":"researched_normalizer","source_path":"normalize.py",
        "description":"public normalizer","input_schema":{"type":"object"},
        "output_schema":{"type":"object"},"trained_on_current_task":False,
        "implementation_origin":{"kind":"adapted_source",
            "summary":"Adapter around the fetched public normalization implementation.",
            "implementation_source_urls":[observed]}}
    with pytest.raises(RuntimeError,match="must be fetched or downloaded"):
        surface.register_tool(source_urls=[observed],**common)
    def fake_download(url,destination,max_bytes):
        Path(destination).write_bytes(b"public checkpoint")
        import hashlib
        return {"url":url,"path":str(destination),"bytes":17,
                "sha256":hashlib.sha256(b"public checkpoint").hexdigest()}
    monkeypatch.setattr("embodied_codex.legacy.engineering.download_public_file",fake_download)
    downloaded=surface.download_public_asset(observed,"assets/checkpoint.bin",1024,"")
    assert downloaded["bytes"]==17 and (workspace.root/"assets"/"checkpoint.bin").is_file()
    registered=surface.register_tool(source_urls=[observed],**common)
    provenance=library.inspect(registered["tool_id"])["manifest"]["provenance"]
    assert provenance["authoring_context"]=="autonomous_engineering_agent"
    assert provenance["training_data_declaration"].startswith("Deterministic source-code")
    assert provenance["implementation_origin"]["kind"]=="adapted_source"
    assert provenance["acquisition_evidence"][0]["record_sha256"]==result["research_record_sha256"]
    manual=library.manual(registered["tool_id"])
    assert manual["manual"]["inputs"]=={} and manual["manual"]["outputs"]=={}
    gap=surface.record_capability_gap(name="research_backed_gap",task="normalize",
        failure_summary="missing normalizer",hypotheses=[],selected_diagnosis="",
        required_capability={},searched_candidates=[],provenance_decision={},
        integration_result={},task_validation={},reuse_evidence={},status="observed",
        evidence_refs=["research_ledger"],previous_gap_id=None)
    assert gaps.inspect(gap["gap_id"])["evidence"][0]["original_path"].endswith(
        "research_ledger.jsonl")
    common["name"]="unresearched_normalizer"
    common["implementation_origin"]={"kind":"original_synthesis",
        "summary":"Original deterministic normalization experiment with background research."}
    with pytest.raises(RuntimeError,match="lack autonomous research evidence"):
        surface.register_tool(source_urls=["https://example.org/invented"],**common)


def test_research_source_listing_is_searchable_and_bounded(tmp_path):
    run=tmp_path/"run";artifact=run/"iterations"/"iteration_001"
    artifact.mkdir(parents=True)
    surface=EngineeringSurface(workspace=TaskWorkspace(run/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    for index in range(35):
        surface._record_research("search",{"query":f"planner {index}",
            "results":[{"url":f"https://example.org/planner-{index}",
                        "title":f"Planner {index}"}]})

    first=surface.list_research_sources()
    assert first["count"]==35 and first["matched_count"]==35
    assert first["returned_count"]==20 and first["next_offset"]==20
    assert first["sources"][0]["url"]=="https://example.org/planner-34"
    second=surface.list_research_sources(offset=20,limit=20)
    assert second["returned_count"]==15 and second["next_offset"] is None
    selected=surface.list_research_sources(query="planner 7")
    assert selected["matched_count"]==1
    assert selected["sources"][0]["url"]=="https://example.org/planner-7"


def test_libero_deployment_tool_contract_is_validated_before_and_after_call():
    from embodied_codex.deployments.libero import LiberoDeployment
    calls=[]
    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.capabilities={"tool:v001":lambda payload:calls.append(payload) or {"value":"bad"}}
    deployment.capability_contracts={"tool:v001":{
        "input_schema":{"type":"object","properties":{"x":{"type":"number"}},
                        "required":["x"],"additionalProperties":False},
        "output_schema":{"type":"object","properties":{"value":{"type":"number"}},
                         "required":["value"],"additionalProperties":False}}}
    deployment.references={};deployment.trace=[];deployment.step=0
    bad_input=deployment._use("tool:v001",{"wrong":1})["result"]
    assert bad_input["tool_error"]["type"]=="ToolContractError" and calls==[]
    bad_output=deployment._use("tool:v001",{"x":1})["result"]
    assert bad_output["tool_error"]["type"]=="ToolContractError" and len(calls)==1


def test_libero_rpc_positive_projection_rejects_new_success_alias():
    from embodied_codex.deployments.libero import LiberoDeployment,LiberoDeploymentError
    deployment=LiberoDeployment.__new__(LiberoDeployment)
    with pytest.raises(LiberoDeploymentError,match="undeclared verify output"):
        deployment.project_rpc_output("verify",{},
            {"verified":False,"goal_achieved":True})
