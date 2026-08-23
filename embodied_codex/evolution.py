"""Persistent autonomous code-test-robot-debug loop; no controller templates."""
from __future__ import annotations

import ast
import fcntl
import json
import os
from pathlib import Path
import hashlib
import shutil
from typing import Any, Callable

from .agent import CodingAgent
from .assets import CapabilityLibrary, SkillLibrary
from .engineering import EngineeringSurface
from .runtime import ControllerRuntime
from .task_model import (build_task_model, canonical_sha256,
                         review_controller_binding, validate_task_model)
from .workspace import TaskWorkspace


def remap_controller_tool_ids(source: str, replacements: dict[str, str]) -> tuple[str, int]:
    """Structurally rebind exact Tool-id constants for a new Robot Adapter.

    Frozen Skills keep immutable provenance IDs, while deployment-owned Tools
    may receive a different local version in a newly adapted environment.  The
    binding is therefore an Adapter concern, not a coding-agent repair.  Only
    exact Python string constants are changed; identifiers, control flow,
    coordinates, and partial strings are untouched.
    """
    tree = ast.parse(source)
    changed = 0

    class Rebind(ast.NodeTransformer):
        def visit_Constant(self, node):
            nonlocal changed
            if isinstance(node.value, str) and node.value in replacements:
                changed += 1
                return ast.copy_location(
                    ast.Constant(value=str(replacements[node.value])), node)
            return node

    tree = Rebind().visit(tree)
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree) + "\n" if changed else source), changed


SYSTEM_PROMPT = '''You are the engineering agent inside Embodied Codex.
You own a persistent coding workspace. Inspect, create, test, and freely rewrite a
complete controller program defining `def run(robot)`. This is not a template or
fixed graph: use ordinary Python, helper modules, loops, recovery logic, public
algorithms, and tested registered Tools as needed.

Robot SDK inside controller.py:
- robot.instruction: task language
- robot.observe(channel="rgbd", request={}) -> sensor observation
- robot.use(tool_id, payload) -> DIRECT Tool output (already unwrapped)
- robot.act(action_dict) -> action result
- robot.verify(verifier, payload) -> sensor-only verification
- robot.record(event)

`deployment_guidance.robot_sdk_contract` is the sole authority for method
signatures, action type strings, required fields, verifier payloads, return
shapes, and opaque-reference rules. Read it before coding and use its exact
names. Do not infer aliases from prose or search the web for this local SDK.

When deployment_guidance supplies a task_model and the registry exposes
preflight_controller, treat it as a semantic contract and bind every phase to
reachable executable code before the robot episode. Ordinary deployments do
not require a separate planner: reason from the complete live instruction and
encode its conditional prerequisites directly in controller.py.

Use workspace commands for unit/integration tests before consuming the one robot
episode allowed in this iteration. Diagnose from RGB/RGB-D, language,
proprioception, action history, videos, and public resources. You may write new Tool
implementations, test them, and register them. Do not use reward, done,
check_success, evaluator feedback, BDDL, simulator object pose/identity, fixed task
coordinates, task/state branches, or a model trained on this evaluated task.

Return {"status":"sensor_success", ...} only when the LAST controller operation is
a fresh sensor verifier returning verified=true. Always run a controller before
finishing the iteration. Infrastructure plumbing is owned by the Harness; focus on
task understanding, capability acquisition, diagnosis, and controller engineering.
On a failed path, return sensor_failure directly; do not call a verifier merely to
make it the last operation. Never invent sentinel references such as "unavailable":
opaque refs must be copied from a live Adapter Tool result and checked non-null.
When deployment_guidance.bootstrap_skill contains an experience_path, read that
sensor-only history before changing the inherited controller; do not repeat a
failure mode already disproven there.
After the one robot episode, use view_sensor_image when visual evidence would help.
You may edit code for the next iteration, but do not call run_robot_controller a
second time: finish your turn so the engine can advance to a fresh iteration.
When the same sensor-confirmed failure mode survives multiple geometric or
parameter variants, stop local parameter search. Use search_web to investigate
public task-disjoint models or algorithms, implement and test the smallest useful
capability, register it as a Tool, and rewrite the controller to consume it.
Treat parameter changes, waypoint subdivision, and residual/error-feedback
variants that preserve the same grasp pose and collision geometry as ONE
capability family. After two sensor-confirmed failures from that family without
improvement in the failed physical stage, mark the family disproven in the
experience and switch capability class (for example grasp orientation/6-DoF
pose, collision-aware planning, perception, or contact policy). Do not register
another renamed Tool from the saturated family.

Generalization and runtime recovery:
- A task is learned only when one unchanged controller handles the required
  scene variations from live observations. Rewriting the whole controller to
  alternate between strategies on successive cases is diagnostic evidence, not
  a generalized solution.
- When history shows that capability families are complementary across scene
  geometries, combine them into one bounded runtime candidate pool. A failed
  physical attempt must return to a safe pose, acquire a fresh RGB-D observation,
  redetect the object and target, regenerate valid references, exclude the
  attempted candidate/family as appropriate, and try a geometrically different
  candidate. Never select a recovery from a task/state identifier.
- Candidate identity includes capability family, grasp pose/orientation,
  approach path, and contact geometry. Retrying a renamed candidate with the
  same physical geometry is not exploration. Record the attempted identity and
  sensor-only rejection reason so the loop cannot oscillate.
- Coordinate updates alone do not establish generalization: use action receipts
  and fresh visual verification to distinguish unreachable/blocked approach,
  empty closure, unstable attachment, transport loss, and bad support relation.
  Only transport after attachment is visually verified.
- Bind the manipulated instance to every identifying relation in the task
  language.  For phrases such as "the object on/in/next to the reference",
  selection must be supported by live masks and metric pair geometry for both
  the object and reference.  If the reference is absent or no pair satisfies
  the relation, fail closed and acquire another observation; never fall back to
  the globally highest-scoring same-class object.  Preserve the selected
  relation anchor so final source vacancy refers to that exact task instance,
  rather than whichever same-name object happened to move.
- A relation phrase used as an open-vocabulary detector query is not proof of
  that relation. When the deployment supplies a VLM relation-grounding Tool,
  pass both live object candidates and live reference candidates so it must
  select a joint pair from the RGB image whenever noun masks are ambiguous.
  Require the returned reference to agree with the independent live RGB-D pair,
  while retaining RGB-D point_ref provenance for every robot motion. A VLM's
  free-text reference description is a proposal, not evidence: never broaden
  detector aliases or rename an unrelated support merely to make that proposal
  self-consistent. Never treat detector confidence as relational evidence.
- A selector Tool that receives candidate arrays must return indices into those
  exact arrays. Use selected_index and selected_reference_index to retrieve the
  original detector records. Never compare or execute a copied selector record:
  only the original RGB-D record owns the valid point_ref for motion.

Artifact and Tool contracts:
- Every executed iteration stores the exact immutable controller source at
  `iterations/iteration_NNN/controller.py`; use these snapshots to recover and
  compose sensor-proven strategies. Do not try to reconstruct old source from
  action traces when its snapshot exists.
- `episodes/.../adapter_trace.json` is intentionally a compact action summary and
  omits Tool return payloads. Read the iteration's `robot_execution.json` when
  diagnosing perception, grasp-candidate, or other Tool outputs. In that file,
  each Tool RPC has its id at `execution.rpc_events[].arguments.tool_id` and its
  direct Tool payload at `execution.rpc_events[].result.result`.
- After a rollout, read the complete current execution with
  `read_run_artifact(path="latest_robot_execution", ...)`. This stable run-local
  reference is authoritative; do not reconstruct or shorten its absolute path.
- A registered Tool version is immutable. If any test case fails, that version
  remains `test_failed`; correct the implementation or oracle, register a new
  version, and test the new version. First run the workspace implementation on
  the proposed inputs so expected values are evidence-based rather than guessed.
- Only `status=tested` Tools are deployable. A newly registered capability counts
  as acquired only when its clean version is tested and controller.py actually
  calls that Tool. Do not claim or preserve unused/failed experimental versions
  as learned capability assets.
'''


class EvolutionEngine:
    def __init__(self, *, root: str|Path, model, deployment_factory: Callable[[],Any],
                 python: str|Path|None=None, deployment_guidance: dict|None=None,
                 required_success_cases: list[str]|None=None,
                 retry_locked_validation_once: bool=False,
                 success_evidence_protocol: str="sensor-verification-v1",
                 require_task_model: bool=False,
                 capability_root: str|Path|None=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.workspace=TaskWorkspace(self.root/"workspace")
        self.capability_root=Path(capability_root).resolve() if capability_root else self.root/"capabilities"
        self.capabilities=CapabilityLibrary(self.capability_root,self.workspace.root)
        self.skills=SkillLibrary(self.root/"skills")
        self.runtime=ControllerRuntime(python=python)
        self.model=model;self.deployment_factory=deployment_factory
        self.guidance=dict(deployment_guidance or {});self.state_path=self.root/"state.json"
        self.required_success_cases=sorted(set(str(x) for x in (required_success_cases or [])))
        self.retry_locked_validation_once=bool(retry_locked_validation_once)
        self.success_evidence_protocol=str(success_evidence_protocol)
        self.require_task_model=bool(require_task_model)
        configuration=self.root/"harness_configuration.json"
        if configuration.is_file():
            previous=json.loads(configuration.read_text())
            if Path(previous["capability_root"]).resolve()!=self.capability_root:
                raise RuntimeError("resumed run capability library mismatch")
        else:
            configuration.write_text(json.dumps({
                "protocol":"embodied-codex-run-configuration-v1",
                "capability_root":str(self.capability_root)},indent=2)+"\n")

    def _task_model(self,task: str):
        if not self.require_task_model:return None
        path=self.root/"task_model.json"
        if path.is_file():
            value=json.loads(path.read_text());validate_task_model(value,task)
            expected=value.get("task_model_sha256");unsigned=dict(value)
            unsigned.pop("task_model_sha256",None)
            if not expected or canonical_sha256(unsigned)!=expected:
                raise RuntimeError("persistent task model hash mismatch")
            return value
        manifests=self.capabilities.tested()
        context={"deployment_guidance":self.guidance,
                 "tested_tool_contracts":[{"tool_id":m["tool_id"],
                    "description":m.get("description"),"input_schema":m.get("input_schema"),
                    "output_schema":m.get("output_schema")} for m in manifests]}
        value=build_task_model(model=self.model,instruction=task,context=context,
                               artifact_dir=self.root/"task_modeling")
        path.write_text(json.dumps(value,indent=2,default=str)+"\n")
        return value

    def bootstrap_skill(self, skill_dir: str|Path):
        """Seed an evolution run from a hash-verified frozen Skill."""
        source=Path(skill_dir).resolve();manifest=json.loads((source/"manifest.json").read_text())
        controller=source/"controller.py"
        if manifest.get("protocol")!="embodied-codex-skill-v1":
            raise RuntimeError("unsupported bootstrap Skill protocol")
        if hashlib.sha256(controller.read_bytes()).hexdigest()!=manifest.get("controller_sha256"):
            raise RuntimeError("bootstrap controller hash mismatch")
        # A resumed run owns a persistent coding workspace. Re-copying the
        # original frozen controller here would silently erase autonomous
        # improvements made before a process restart. Validate that the same
        # bootstrap is requested, then preserve the evolved workspace exactly.
        bootstrap_path = self.root / "bootstrap.json"
        if self.state_path.is_file() and bootstrap_path.is_file():
            record = json.loads(bootstrap_path.read_text())
            if (record.get("skill_id") != manifest.get("skill_id") or
                    record.get("controller_sha256") != manifest.get("controller_sha256")):
                raise RuntimeError("resumed run bootstrap mismatch")
            if not self.workspace._path("controller.py").is_file():
                raise RuntimeError("resumed run lost its persistent controller")
            return record
        imported=self.capabilities.import_skill_tools(source)
        self.workspace.write_file("controller.py",controller.read_text())
        experience_source=source/"experience.json"
        if experience_source.is_file():
            expected=manifest.get("experience_sha256")
            if (not expected or hashlib.sha256(experience_source.read_bytes()).hexdigest()!=expected):
                raise RuntimeError("bootstrap experience hash mismatch")
        else:
            legacy=source.parents[2]/"state.json"
            experience_source=legacy if legacy.is_file() else None
        experience_path=None
        if experience_source is not None:
            destination=self.root/"bootstrap_experience.json"
            shutil.copy2(experience_source,destination);experience_path=str(destination)
        record={"skill_id":manifest.get("skill_id"),"task":manifest.get("task"),
                "controller_sha256":manifest.get("controller_sha256"),
                "experience_path":experience_path,**imported}
        bootstrap_path.write_text(json.dumps(record,indent=2)+"\n")
        return record

    def bind_bootstrap_deployment_tools(self, replacements: dict[str, str]):
        """Apply Adapter-owned dependency bindings before the first episode."""
        replacements = {str(old): str(new) for old, new in replacements.items()
                        if old and new and old != new}
        controller = self.workspace._path("controller.py")
        if not replacements or not controller.is_file():
            return {"replacements": replacements, "changed_constants": 0}
        original = controller.read_text()
        rebound, changed = remap_controller_tool_ids(original, replacements)
        controller.write_text(rebound)
        result = {
            "replacements": replacements,
            "changed_constants": changed,
            "original_controller_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "bound_controller_sha256": hashlib.sha256(rebound.encode()).hexdigest(),
            "method": "python_ast_exact_string_constant_rebind",
        }
        path = self.root / "bootstrap.json"
        if path.is_file():
            record = json.loads(path.read_text())
            record["deployment_binding"] = result
            path.write_text(json.dumps(record, indent=2) + "\n")
        return result

    def bind_current_deployment_tools(self, current: dict[str,str]):
        """Persist and structurally migrate Adapter-owned Tool dependencies.

        Deployment implementations can be corrected independently of an evolved
        task controller.  On resume, exact old Tool-id constants are rebound to
        the current Adapter version; task logic and analytic Tools remain pinned.
        """
        current={str(name):str(tool_id) for name,tool_id in current.items()}
        path=self.root/"deployment_bindings.json"
        previous={}
        if path.is_file():
            record=json.loads(path.read_text())
            previous={str(k):str(v) for k,v in (record.get("current") or {}).items()}
        else:
            # Runs created before this contract have no binding ledger.  Only
            # deployment-owned manifests with the same declared capability name
            # are eligible; arbitrary analytic Tool versions are never touched.
            for manifest in self.capabilities.list_all():
                name=str(manifest.get("name") or "")
                tool_id=str(manifest.get("tool_id") or "")
                if (name in current and manifest.get("execution_owned_by_deployment")
                        and tool_id!=current[name]):
                    previous.setdefault(name,tool_id)
        replacements={previous[name]:tool_id for name,tool_id in current.items()
                      if name in previous and previous[name]!=tool_id}
        result=self.bind_bootstrap_deployment_tools(replacements)
        history=[]
        if path.is_file():history=list(json.loads(path.read_text()).get("history") or [])
        if replacements:
            history.append({"previous":previous,"current":current,"binding":result})
        path.write_text(json.dumps({
            "protocol":"embodied-codex-deployment-bindings-v1",
            "current":current,"history":history},indent=2)+"\n")
        return result

    def _state(self,task,skill_name):
        if self.state_path.is_file():
            state=json.loads(self.state_path.read_text())
            if state["task"]!=task or state["skill_name"]!=skill_name: raise RuntimeError("run mismatch")
            return state
        return {"task":task,"skill_name":skill_name,"status":"evolving","iterations":[]}
    def _save(self,state):
        tmp=self.state_path.with_suffix(".tmp");tmp.write_text(json.dumps(state,indent=2,default=str)+"\n");tmp.replace(self.state_path)

    @staticmethod
    def _brief(execution):
        if not execution:return None
        run=execution.get("execution",{}); events=[]; records=[]
        for event in run.get("rpc_events",[]):
            if event.get("method")=="record":
                records.append(event.get("arguments",{}).get("event"))
        for event in run.get("rpc_events",[])[-24:]:
            item={"method":event.get("method")}
            if event.get("method")=="act":
                item["action"]=event.get("arguments",{}).get("action")
                result=event.get("result",{});item["result"]={k:result.get(k) for k in
                    ("reached","step","eef_after","gripper_qpos","target_xyz") if k in result}
            elif event.get("method")=="verify":
                result=event.get("result",{});item["verifier"]=event.get("arguments",{}).get("verifier")
                item["result"]={k:v for k,v in result.items() if k not in ("frame","cameras")}
            elif event.get("method")=="use": item["tool_id"]=event.get("arguments",{}).get("tool_id")
            elif event.get("method")=="observe":
                result=event.get("result",{}); cameras=result.get("cameras") or {}
                item["result"]={"frame_id":result.get("frame_id"),"step":result.get("step"),
                    "rgb_paths":{name:value.get("rgb_path") for name,value in cameras.items()
                        if isinstance(value,dict) and value.get("rgb_path")}}
            if event.get("error"):item["error"]=event["error"]
            events.append(item)
        sensor_report={k:v for k,v in (execution.get("sensor_report") or {}).items()
                       if not str(k).startswith("_harness_")}
        return {"controller_path":execution.get("controller_path"),
                "controller_snapshot":execution.get("controller_snapshot"),
                "completed":run.get("completed"),"error":run.get("error"),
                "controller_result":run.get("result"),"rpc_evidence":events,
                "controller_records":records[-32:],
                "sensor_report":sensor_report,
                "sensor_success_candidate":execution.get("sensor_success_candidate")}

    def run(self, *, task: str, skill_name: str, max_iterations: int):
        with (self.root/".run.lock").open("a+") as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError("another Embodied Codex owns this run") from exc
            lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
            return self._run(task,skill_name,max_iterations)

    def _run(self,task,skill_name,max_iterations):
        state=self._state(task,skill_name)
        if self.required_success_cases:
            gate=state.get("generalization_gate")
            if gate and gate.get("evidence_protocol")!=self.success_evidence_protocol:
                state.setdefault("invalidated_generalization_gates",[]).append({
                    "reason":"sensor success evidence protocol changed",
                    "replacement_protocol":self.success_evidence_protocol,
                    "gate":gate})
                state["generalization_gate"]={
                    "required_cases":self.required_success_cases,
                    "successes_by_program":{},
                    "evidence_protocol":self.success_evidence_protocol}
                self._save(state)
        bootstrap_path=self.root/"bootstrap.json"
        if bootstrap_path.is_file():
            bootstrap=json.loads(bootstrap_path.read_text())
            if bootstrap.get("task")!=task:raise RuntimeError("bootstrap Skill task mismatch")
            self.guidance.setdefault("bootstrap_skill",bootstrap)
        if state["status"]=="sensor_success":return state
        task_model=self._task_model(task)
        episode_count=sum(1 for record in state["iterations"] if record.get("evidence") is not None)
        while episode_count<max_iterations:
            index=len(state["iterations"])+1
            artifact=self.root/"iterations"/f"iteration_{index:03d}"
            persisted_record={}
            def persist_robot_execution(execution):
                if persisted_record:return
                record={"iteration":index,"agent_completed":False,
                        "agent_error":"post_execution_agent_pending",
                        "coding_passes":None,"robot_episode":episode_count+1,
                        "evidence":self._brief(execution),
                        "robot_execution_transaction_committed":True}
                state["iterations"].append(record);self._save(state)
                persisted_record["record"]=record
            semantic_reviewer=None
            if task_model is not None:
                semantic_reviewer=lambda **payload:review_controller_binding(
                    model=self.model,trace_path=artifact/"controller_semantic_review.jsonl",
                    **payload)
            surface=EngineeringSurface(workspace=self.workspace,capabilities=self.capabilities,
                runtime=self.runtime,deployment_factory=self.deployment_factory,
                artifact_dir=artifact,task_model=task_model,
                semantic_reviewer=semantic_reviewer,
                sdk_contract=self.guidance.get("robot_sdk_contract"),
                active_deployment_tool_ids=self.guidance.get("active_deployment_tool_ids"),
                execution_observer=persist_robot_execution)
            previous=next((record.get("evidence") for record in reversed(state["iterations"])
                           if record.get("evidence") is not None),None)
            active_deployment=set(str(x) for x in
                                  (self.guidance.get("active_deployment_tool_ids") or []))
            tested_manifests=[manifest for manifest in self.capabilities.tested()
                if not manifest.get("execution_owned_by_deployment")
                or not active_deployment or manifest.get("tool_id") in active_deployment]
            base_instruction={"task":task,"iteration":index,"robot_episode":episode_count+1,
                "previous_sensor_evidence":previous,"deployment_guidance":self.guidance,
                "tested_tools":[m["tool_id"] for m in tested_manifests],
                # Tool ids alone invite the coding model to guess whether a
                # result is a single waypoint, a ranked list, or a receipt.
                # Put the immutable contracts directly in every programming
                # instruction so controller composition is schema-driven.
                "tested_tool_contracts":[{
                    "tool_id":m["tool_id"],"description":m.get("description"),
                    "input_schema":m.get("input_schema"),
                    "output_schema":m.get("output_schema")}
                    for m in tested_manifests],
                "requirement":"engineer and run one complete controller program"}
            if task_model is not None:
                base_instruction["task_model"]=task_model
                base_instruction["preflight_requirement"]=(
                    "bind every task_model phase to reachable executable functions and call "
                    "preflight_controller on the exact source before run_robot_controller")
            if self.required_success_cases:
                gate=state.get("generalization_gate") or {"required_cases":self.required_success_cases,
                    "successes_by_program":{},
                    "evidence_protocol":self.success_evidence_protocol}
                base_instruction["generalization_gate"]={
                    "required_cases":gate["required_cases"],
                    "successes_by_program":gate["successes_by_program"],
                    "rule":"the identical controller hash must pass every case; do not add state branches"}
            passes=[];locked_validation=False;locked_retry=False
            # Once one required case accepts a controller hash, validate that
            # exact immutable program on subsequent cases without asking the
            # coding model to inspect or rewrite it.  A failed locked rollout
            # returns control to GPT on the following iteration because the
            # immediately previous evidence is then unsuccessful.
            controller_path=self.workspace.root/"controller.py"
            # After an Adapter/verifier infrastructure correction, replay the
            # current immutable program once before asking the coding model to
            # react to stale failure evidence.  This is valid even before a
            # program has accumulated generalization coverage: the whole point
            # of the replay is to determine whether the controller failed or
            # the old infrastructure did.
            if (self.retry_locked_validation_once and previous
                    and controller_path.is_file()):
                surface.run_robot_controller("controller.py")
                agent_result={"completed":True,"error":None,"tool_results":[]}
                locked_validation=True
                locked_retry=True
                self.retry_locked_validation_once=False
            elif (self.required_success_cases and previous
                    and previous.get("sensor_success_candidate") is True
                    and controller_path.is_file()):
                program_sha=hashlib.sha256(controller_path.read_bytes()).hexdigest()
                gate=state.get("generalization_gate") or {}
                coverage=(gate.get("successes_by_program") or {}).get(program_sha) or []
                required=gate.get("required_cases") or self.required_success_cases
                if coverage and not set(required).issubset(coverage):
                    surface.run_robot_controller("controller.py")
                    agent_result={"completed":True,"error":None,"tool_results":[]}
                    locked_validation=True
            if not locked_validation:
                for coding_pass in range(1,4):
                    instruction=dict(base_instruction);instruction["coding_pass"]=coding_pass
                    if coding_pass>1:
                        instruction["correction"]=("The prior coding pass ended without a robot "
                            "episode. Inspect the persistent workspace and call run_robot_controller "
                            "once before ending this pass.")
                    agent=CodingAgent(model=self.model,registry=surface.registry(),system_prompt=SYSTEM_PROMPT,
                                      trace_path=artifact/"agent_trace.jsonl")
                    agent_result=agent.run(json.dumps(instruction,default=str));passes.append(agent_result)
                    if surface.last_execution is not None:break
            if surface.last_execution is None:
                raise RuntimeError("coding agent ended three passes without a robot episode")
            evidence=self._brief(surface.last_execution)
            record=persisted_record.get("record")
            if record is None:
                # Defensive fallback for a custom EngineeringSurface that did
                # not invoke the observer; normal robot runs are committed by
                # run_robot_controller before the model sees their result.
                record={"iteration":index,"robot_episode":episode_count+1}
                state["iterations"].append(record)
            record.update({"agent_completed":agent_result["completed"],
                           "agent_error":agent_result.get("error"),
                           "coding_passes":len(passes),"evidence":evidence})
            if locked_validation:record["locked_generalization_validation"]=True
            if locked_retry:
                record["locked_validation_retry_after_infrastructure_change"]=True
                record["infrastructure_replay_without_model"]=True
            self._save(state)
            episode_count+=1
            if surface.last_execution and surface.last_execution["sensor_success_candidate"]:
                if self.required_success_cases:
                    report=surface.last_execution.get("sensor_report") or {}
                    case=str(report.get("_harness_case_id") or "")
                    if case not in self.required_success_cases:
                        raise RuntimeError("deployment omitted required Harness case id")
                    program_sha=surface.last_execution["execution"].get("program_sha256")
                    gate=state.setdefault("generalization_gate",{
                        "required_cases":self.required_success_cases,"successes_by_program":{},
                        "evidence_protocol":self.success_evidence_protocol})
                    coverage=gate["successes_by_program"].setdefault(program_sha,[])
                    if case not in coverage:coverage.append(case);coverage.sort()
                    record["generalization_case"]=case
                    record["generalization_coverage"]={"program_sha256":program_sha,
                        "passed_cases":list(coverage),"required_cases":self.required_success_cases}
                    self._save(state)
                    if not set(self.required_success_cases).issubset(coverage):
                        continue
                controller=self.workspace._path(surface.last_execution["controller_path"])
                used=sorted({e.get("arguments",{}).get("tool_id")
                    for e in surface.last_execution["execution"].get("rpc_events",[])
                    if e.get("method")=="use" and e.get("arguments",{}).get("tool_id")})
                skill=self.skills.freeze(name=skill_name,task=task,controller=controller,
                    evidence={"iteration":index,"sensor_only":True,"report":evidence},
                    tool_ids=used,tools=self.capabilities,
                    experience={"iterations":state["iterations"],
                                "generalization_gate":state.get("generalization_gate"),
                                "task_model":task_model},task_model=task_model)
                state.update({"status":"sensor_success","skill":skill});self._save(state);return state
        return state

__all__=["EvolutionEngine","SYSTEM_PROMPT"]
