from pathlib import Path

import pytest

from embodied_codex.runtime import ControllerRuntime
from embodied_codex.examples.evaluate_libero_skill import inspect_skill, _sensor_success
from embodied_codex.capabilities.open_vocab_rgbd import OpenVocabularyRGBD
from embodied_codex.agent import CodingAgent
from embodied_codex.assets import CapabilityLibrary
from embodied_codex.engineering import EngineeringSurface
from embodied_codex.evolution import EvolutionEngine
from embodied_codex.registry import FunctionRegistry
from embodied_codex.workspace import TaskWorkspace, WorkspaceError
from embodied_codex.web import _bing_results
from embodied_codex.conformance import audit_run


PYTHON = "/data/zxy/envs/vla-report/bin/python"


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


def test_workspace_supports_free_code_edits_and_commands(tmp_path):
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
    with pytest.raises(WorkspaceError): workspace.write_file("../escape.py", "bad")


def _call(number, name, arguments):
    import json
    return {"content":"","tool_calls":[{"id":f"c{number}","name":name,
            "arguments":json.dumps(arguments)}]}


class ScriptedCodingModel:
    def __init__(self): self.steps={1:0,2:0}
    def decide(self, *, messages, tools):
        import json
        instruction=json.loads(messages[1]["content"])
        assert "tested_tool_contracts" in instruction
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
    assert tools == {}
    assert all(instance.closed for instance in instances)
    successor=EvolutionEngine(root=tmp_path/"successor",model=object(),
        deployment_factory=lambda:None,python=PYTHON)
    bootstrap=successor.bootstrap_skill(skill)
    assert bootstrap["skill_id"]=="move_cube_skill:v001"
    assert Path(bootstrap["experience_path"]).is_file()
    assert (tmp_path/"successor"/"workspace"/"controller.py").read_bytes()==controller.read_bytes()


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
            iteration=json.loads(messages[1]["content"])["iteration"]
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
    from embodied_codex.evolution import EvolutionEngine
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

    class CorrectedVerifierDeployment(ClosedLoopDeployment):
        def dispatch(self,method,arguments):
            if method=="verify":return {"verified":True,"corrected_infrastructure":True}
            return super().dispatch(method,arguments)
        def sensor_report(self,execution):
            return {"sensor_verification_passed":True,"benchmark_signal_exposed":False}

    class ModelMustNotRun:
        def decide(self,**unused):raise AssertionError("model must not run during replay")

    resumed=EvolutionEngine(root=run_root,model=ModelMustNotRun(),
        deployment_factory=CorrectedVerifierDeployment,python=PYTHON,
        retry_locked_validation_once=True)
    state=resumed.run(task="move the cube",skill_name="retry_skill",max_iterations=2)
    assert state["status"]=="sensor_success"
    replay=state["iterations"][1]
    assert replay["coding_passes"]==0
    assert replay["infrastructure_replay_without_model"] is True
    assert replay["locked_validation_retry_after_infrastructure_change"] is True


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


def test_engineering_agent_can_see_only_current_run_sensor_images(tmp_path):
    run_root=tmp_path/"run"; artifact=run_root/"iterations"/"iteration_001"
    surface=EngineeringSurface(workspace=TaskWorkspace(run_root/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    image_path=run_root/"episodes"/"episode_001"/"frame.png"
    image_path.parent.mkdir(parents=True)
    # Valid 1x1 PNG; the Harness transports bytes without interpreting pixels.
    import base64
    image_path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="))
    result=surface.view_sensor_image(str(image_path))["_embodied_codex_image"]
    assert result["mime_type"]=="image/png" and result["data_base64"]
    assert str(image_path) in surface.list_sensor_artifacts("episodes/**/*.png")
    log_path=run_root/"episodes"/"episode_001"/"trace.json"
    log_path.write_text('{"ok": true}\n')
    assert '"ok": true' in surface.read_run_artifact(str(log_path))["content"]
    workspace_image=run_root/"workspace"/"montage.png"
    workspace_image.write_bytes(image_path.read_bytes())
    assert surface.view_sensor_image("montage.png")["_embodied_codex_image"]["path"]==str(workspace_image)
    outside=tmp_path/"outside.png";outside.write_bytes(image_path.read_bytes())
    with pytest.raises(RuntimeError):surface.view_sensor_image(str(outside))
    with pytest.raises(RuntimeError):surface.list_sensor_artifacts("../**/*")


def test_large_run_artifact_is_readable_in_bounded_line_chunks(tmp_path):
    run_root=tmp_path/"run";artifact=run_root/"iterations"/"iteration_001"
    surface=EngineeringSurface(workspace=TaskWorkspace(run_root/"workspace"),
        capabilities=object(),runtime=object(),deployment_factory=lambda:None,
        artifact_dir=artifact)
    log=run_root/"episodes"/"episode_001"/"large.json"
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
                     "rpc_events":[]},
        "sensor_report":{"benchmark_signal_exposed":False,
            "trace_path":str(episode/"adapter_trace.json"),
            "rollout_path":str(episode/"rollout.mp4")},
        "sensor_success_candidate":False,
        "robot_contract_preflight":{"passed":True}}
    (iteration/"robot_execution.json").write_text(json.dumps(report))
    events=[
        {"type":"task","instruction":json.dumps({"previous_sensor_evidence":None})},
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
    audit=audit_run(root)
    assert audit["conformant"] is True
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


def test_latest_robot_execution_resolves_previous_iteration_before_new_rollout(tmp_path):
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    capabilities=CapabilityLibrary(tmp_path/"run"/"capabilities",workspace.root)
    previous=tmp_path/"run"/"iterations"/"iteration_001"/"robot_execution.json"
    previous.parent.mkdir(parents=True);previous.write_text('{"iteration": 1}\n')
    current=tmp_path/"run"/"iterations"/"iteration_002"
    surface=EngineeringSurface(workspace=workspace,capabilities=capabilities,
        runtime=ControllerRuntime(python=PYTHON),deployment_factory=lambda:None,
        artifact_dir=current)
    result=surface.read_run_artifact("latest_robot_execution")
    assert result["path"]==str(previous.resolve())
    assert '"iteration": 1' in result["content"]
    current_execution=current/"robot_execution.json"
    current_execution.write_text('{"iteration": 2}\n')
    result=surface.read_run_artifact("latest_robot_execution")
    assert result["path"]==str(current_execution.resolve())
    snapshot=previous.parent/"controller.py";snapshot.write_text("def run(robot): pass\n")
    source=surface.read_file(str(snapshot),1,20)
    assert source["exists"] is True and "def run" in source["content"]
    with pytest.raises(RuntimeError,match="inside the current run"):
        surface.read_file(str(tmp_path/"outside.py"),1,20)


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


def test_iteration_budget_counts_robot_episodes_not_coding_only_passes(tmp_path):
    class DelayedRunner:
        step=0
        def decide(self,*,messages,tools):
            import json
            instruction=json.loads(messages[1]["content"])
            if instruction["coding_pass"]==1:return {"content":"edited only","tool_calls":[]}
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


def test_libero_capability_outage_is_structured_not_controller_crash():
    from embodied_codex.deployments.libero import LiberoDeployment,LiberoDeploymentError
    class APIConnectionError(Exception):pass
    def unavailable(_payload):raise APIConnectionError("temporary connection loss")
    deployment=LiberoDeployment.__new__(LiberoDeployment)
    deployment.capabilities={"vlm:v001":unavailable};deployment.references={}
    deployment.trace=[];deployment.step=12
    receipt=deployment._use("vlm:v001",{"frame":"sensor"})
    assert receipt["result"]=={"ok":False,"tool_error":{
        "type":"APIConnectionError","message":"temporary connection loss"}}
    assert deployment.trace[-1]["tool_error"]["type"]=="APIConnectionError"
    with pytest.raises(LiberoDeploymentError,match="unregistered Tool"):
        deployment._use("missing:v001",{})


def test_tool_tests_use_numeric_tolerance_and_preserve_real_failures(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"identity.py").write_text("def run(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace)
    tolerant=library.register_tool(name="numeric_identity",source_path="identity.py",
        description="test",input_schema={},output_schema={},source_urls=[],
        trained_on_current_task=False)["tool_id"]
    result=library.test_tool(tolerant,[{"input":{"x":0.020000000000000018},
                                        "expected":{"x":0.02}}])
    assert result["status"]=="tested"


def test_tool_listing_is_compact_and_inspection_keeps_full_evidence(tmp_path):
    workspace=tmp_path/"workspace";workspace.mkdir()
    (workspace/"identity.py").write_text("def run(payload): return payload\n")
    library=CapabilityLibrary(tmp_path/"tools",workspace)
    tool_id=library.register_tool(name="identity",source_path="identity.py",
        description="identity",input_schema={"value":"number"},
        output_schema={"value":"number"},source_urls=["https://example.org/tool"],
        trained_on_current_task=False)["tool_id"]
    library.test_tool(tool_id,[{"input":{"value":1},"expected":{"value":1}}])
    listed=library.list_summaries()
    assert listed==[{"protocol":"embodied-codex-tool-v1","tool_id":tool_id,
        "name":"identity","version":1,"description":"identity",
        "input_schema":{"value":"number"},"output_schema":{"value":"number"},
        "status":"tested","trained_on_current_task":False,
        "privileged_state_used":False}]
    assert "tests" not in listed[0] and "source_urls" not in listed[0]
    inspected=library.inspect(tool_id)
    assert inspected["manifest"]["tests"] and inspected["manifest"]["source_urls"]

    strict=library.register_tool(name="history_identity",source_path="identity.py",
        description="test",input_schema={},output_schema={},source_urls=[],
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
        "rounds":3,"required":2,"winning_votes":2,"agreed":True}


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
    from embodied_codex.evolution import remap_controller_tool_ids

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


def test_resumed_bootstrap_preserves_evolved_controller(tmp_path):
    import hashlib
    import json
    from embodied_codex.evolution import EvolutionEngine

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


def test_engineering_prompt_requires_relation_grounded_instance_binding():
    from embodied_codex.evolution import SYSTEM_PROMPT

    assert "identifying relation" in SYSTEM_PROMPT
    assert "never fall back" in SYSTEM_PROMPT
    assert "globally highest-scoring same-class object" in SYSTEM_PROMPT


def _causal_task_model():
    from embodied_codex.task_model import canonical_sha256
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
    from embodied_codex.task_model import TaskModelError,validate_task_model
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


def test_robot_contract_preflight_rejects_action_typo_and_null_ref(tmp_path):
    from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT
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


def test_single_sdk_contract_matches_adapter_and_all_examples_validate():
    import ast
    import inspect
    import textwrap
    from embodied_codex.deployments.libero import LiberoDeployment
    from embodied_codex.sdk_contract import (LIBERO_ROBOT_SDK_CONTRACT,
        validate_action,validate_verifier_request)

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
    for name,contract in LIBERO_ROBOT_SDK_CONTRACT["actions"].items():
        assert validate_action(contract["example"])==name
    for name,contract in LIBERO_ROBOT_SDK_CONTRACT["verifiers"].items():
        validate_verifier_request(name,contract["example"])


def test_sdk_linter_rejects_missing_required_literal_fields(tmp_path):
    from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT
    workspace=TaskWorkspace(tmp_path/"run"/"workspace")
    workspace.write_file("controller.py",'''def run(robot):
    robot.act({"type":"move_to_pose", "offset":[0,0,0]})
    return robot.verify("visual_attachment", {"frame":{}, "source_ref":"s"})
''')
    surface=EngineeringSurface(workspace=workspace,capabilities=object(),runtime=object(),
        deployment_factory=lambda:None,artifact_dir=tmp_path/"run"/"iterations"/"iteration_001",
        sdk_contract=LIBERO_ROBOT_SDK_CONTRACT)
    with pytest.raises(RuntimeError,match="missing literal fields.*pose_ref.*object_query"):
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
    from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT

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
    from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT

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
    from embodied_codex.sdk_contract import LIBERO_ROBOT_SDK_CONTRACT
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


def test_shared_capability_library_is_visible_across_runs_and_resume_is_pinned(tmp_path):
    shared=tmp_path/"shared_capabilities"
    first=EvolutionEngine(root=tmp_path/"run_a",model=object(),
        deployment_factory=lambda:None,python=PYTHON,capability_root=shared)
    first.workspace.write_file("reusable.py","def run(payload): return {'x': payload['x']}\n")
    tool_id=first.capabilities.register_tool(name="reusable_identity",
        source_path="reusable.py",description="shared tested identity",
        input_schema={"x":"number"},output_schema={"x":"number"},
        source_urls=["https://example.org/public-algorithm"],trained_on_current_task=False)["tool_id"]
    assert first.capabilities.test_tool(tool_id,[{"input":{"x":3},
                                                  "expected":{"x":3}}])["status"]=="tested"

    second=EvolutionEngine(root=tmp_path/"run_b",model=object(),
        deployment_factory=lambda:None,python=PYTHON,capability_root=shared)
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
    engine.workspace.write_file("controller.py",'''TOOL = "vision_adapter:v001"
NOTE = "vision_adapter:v001-extra"
def run(robot): return robot.use(TOOL, {})
''')
    first=engine.bind_current_deployment_tools({"vision_adapter":"vision_adapter:v001"})
    assert first["changed_constants"]==0
    second=engine.bind_current_deployment_tools({"vision_adapter":"vision_adapter:v002"})
    source=engine.workspace.read_file("controller.py")["content"]
    assert second["changed_constants"]==1
    assert "TOOL = 'vision_adapter:v002'" in source
    assert "NOTE = 'vision_adapter:v001-extra'" in source
    ledger=__import__("json").loads((tmp_path/"run"/"deployment_bindings.json").read_text())
    assert ledger["current"]=={"vision_adapter":"vision_adapter:v002"}
    assert len(ledger["history"])==1


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
