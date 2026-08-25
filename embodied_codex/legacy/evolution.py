"""Persistent autonomous code-test-robot-debug loop; no controller templates."""
from __future__ import annotations

import fcntl
import ast
import json
import os
import re
from pathlib import Path
import hashlib
import shutil
from typing import Any, Callable

from .agent import CodingAgent
from .assets import (AssetError, CapabilityLibrary, SkillLibrary, ExperienceLibrary,
                     CapabilityGapLibrary)
from .engineering import (EngineeringSurface, _compact_evidence_value,
                          _controller_semantic_sha256,_controller_strategy_sha256,
                          _controller_strategy_prefix_sha256,
                          _controller_tool_ids_before_robot_event,
                          _controller_tool_ids,
                          remap_controller_tool_ids,
                          transient_infrastructure_failure)
from .runtime import ControllerRuntime
from .task_model import (build_task_model, canonical_sha256,
                         review_capability_integration, review_controller_binding,
                         review_controller_task_fidelity,
                         validate_task_model)
from .workspace import TaskWorkspace


def _post_action_transient_replay_source(iterations):
    """Return the first Controller in the latest verifier-outage chain."""
    source=None
    for record in reversed(iterations):
        transient=record.get("transient_infrastructure_failure") or {}
        if transient.get("kind")!="transient_post_action_sensor_verifier_outage":break
        source=record
    return source


SYSTEM_PROMPT = '''You are the autonomous engineering agent inside Embodied Codex.
You own the complete persistent task workspace. Understand the live task, inspect
the available interfaces and evidence, and create, test, run, and freely rewrite a
complete controller program defining `def run(robot)`. There is no task template,
fixed behavior graph, prescribed perception stack, or externally supplied failure
classifier. You—not the Harness—choose the program structure, hypotheses,
algorithms, Tools, recovery policy, and stopping decision from evidence.

Capability acquisition has two deployment paths. Use register_tool for a small
deterministic Python implementation whose source defines exactly one top-level
`def run(payload)`. Use register_capability_package for an
acquired repository, checkpoint-backed perception/policy model, planner, or a
self-contained service wrapper. A package still exposes run(payload) through the
same robot.use contract, but executes in a separate network-isolated CPU/CUDA
worker. Do not merely mention an external algorithm: install its task-disjoint
bundle, verify provenance/checkpoint hashes, test its JSON contract, and make the
Controller consume the tested version.
Host ROS or robot services require a deployment-owned Tool binding because IPC
and the safety boundary belong to the Adapter; never label an isolated package a
ROS bridge.
Asset source and model-card URLs must be exact URLs observed through search_web or
fetch_web_page in this run. The Harness binds those immutable research records to
the registered provenance; invented or unobserved URLs are rejected. After research,
call list_research_sources and copy exact eligible URLs into registration calls.
For each deterministic Tool, declare implementation_origin honestly. A search result
that inspired an original heuristic is research background, not adopted source code.
Use adapted_source/adopted_source only after fetching or downloading the exact
implementation URL and state concretely what was reused. Contract unit tests prove
I/O behavior only; they do not prove that a Tool closes a Capability Gap or works on
the robot. Persistent-Gap integrations receive an independent source/code/Controller
review before another physical episode.
For deterministic register_tool, omit manual/provenance/dependency_spec unless a
non-default value is genuinely required; the Harness creates schema-consistent defaults.
Engineering run_command is intentionally network-isolated. Use
download_public_asset for public repositories, checkpoints, or dependencies, then
use run_command offline to hash, inspect, unpack, build, and test them.
Every retrieved non-deployment Tool with status=tested is automatically bound to
each fresh Robot Adapter before controller execution. The field
active_deployment_tool_ids selects only the Adapter-owned seed Tool versions; it
is not a whitelist for Agent-authored Tools.

Robot SDK inside controller.py:
- robot.instruction: task language
- robot.observe(channel="rgbd", request={}) -> sensor observation
- robot.use(tool_id, payload) -> DIRECT Tool output (already unwrapped)
- robot.act(action_dict) -> action result
- robot.verify(verifier, payload) -> sensor-only verification
- robot.record(event)

`deployment_guidance.robot_sdk_contract` in the JSON task input is the compact
index for the sole authoritative Robot SDK contract. Use it by default. When an
unfamiliar action/verifier or a contract rejection requires exact optional
semantics, call `inspect_robot_sdk_contract` for only that dotted section. Never
call read_file for deployment guidance, infer aliases from prose, or search the
web for this local SDK.

When the task input supplies a task_model and the registry exposes
preflight_controller, treat it as a semantic contract and bind every phase to
reachable executable code before the robot episode. Otherwise reason directly
from the complete instruction and encode any necessary prerequisites and
conditional behavior in controller.py.

Use workspace commands for unit/integration tests before consuming the one robot
episode allowed in this iteration. Diagnose from RGB/RGB-D, language,
proprioception, action history, videos, and public resources. You may write new Tool
implementations, test them, and register them. Do not use reward, done,
check_success, evaluator feedback, BDDL, simulator object pose/identity, fixed task
coordinates, task/state branches, or a model trained on this evaluated task.
For an existing multi-page Controller, use `replace_file_lines` after reading the
relevant page instead of repeatedly re-reading source or copying a large exact
string into `replace_in_file`. Compile the result, then run it.

The LAST controller operation on a success path must be a fresh sensor verifier
returning verified=true. The controller may return that verifier result directly or
wrap it in diagnostic data; no magic return-status string is required. Always run a controller before
finishing the iteration. Infrastructure plumbing is owned by the Harness; focus on
task understanding, capability acquisition, diagnosis, and controller engineering.
On a failed path, return sensor_failure directly; do not call a verifier merely to
make it the last operation. Never invent sentinel references such as "unavailable":
opaque refs must be copied from a live Adapter Tool result and checked non-null.
When deployment_guidance.bootstrap_skill contains an experience_path, read that
sensor-only history before changing the inherited controller; do not repeat a
failure mode already disproven there.
After the one robot episode, inspect the complete execution artifact and use
view_sensor_image when visual evidence would help. If a decisive event happened
between explicit observations, use extract_rollout_frames and inspect the returned
images before forming a diagnosis. Distinguish an interface/program error from a
capability or physical failure using evidence; do not ask the Harness to encode a
task-specific rule for an error that the engineering agent can diagnose.
You may edit code for the next iteration, but do not call run_robot_controller a
second time: finish your turn so the engine can advance to a fresh iteration.
Maintain explicit hypotheses and compare predicted versus observed evidence.
Evidence inspection is deliberately bounded. Before requesting many alternate
pages, projections, or frames, write the current evidence-backed hypotheses and
next discriminating action to a persistent workspace note. If the Harness pauses
evidence acquisition, do not evade the limit with different paths or parameters:
externalize the diagnosis, edit/test the Controller, acquire/register a justified
capability, or run the experiment. Substantive engineering progress opens another
bounded evidence phase. Terminal output is governed by the same evidence budget;
do not use shell, Python, cat, sed, or similar commands to bypass a paused read.
When existing capabilities are insufficient, use search_web and fetch_web_page to
investigate public task-disjoint knowledge, models, algorithms, and repositories;
implement and test a useful capability, register it as a Tool, and rewrite the
controller to actually consume it. Do not create renamed assets that implement the
same disproven idea. Preserve both successful experience and unresolved failure
evidence so later tasks can reuse it.
Before acquiring a missing capability, publish an evidence-backed Capability Gap.
Update that same Gap lineage after search, integration, and task-level validation
with `revise_capability_gap`, supplying only fields changed by the new evidence;
do not leave acquisition decisions only in free-form reasoning. Search prior Gaps
to avoid repeating rejected approaches and preserve unresolved frontier evidence.
Gap fields are enforced, but evidence may move a lineage back from integrating or
unresolved to diagnosis/search when a hypothesis is falsified. A searching revision
must record actual queries/candidates; integrating must record provenance and a
concrete integration result. Always inspect and extend the latest retrieved revision,
and do not label another Controller parameter tweak as capability integration.
Review the retrieved Experience index before choosing a strategy and call
search_assets when the initial Top-K is insufficient. After a rollout, call
register_experience only for a directly observed failure fact or a mechanism or
procedure supported by a discriminating validation. A causal hypothesis inferred
from one failed rollout belongs in the Capability Gap until a later experiment
distinguishes it from competing hypotheses. Do not turn a speculative diagnosis
into reusable Experience. Supply a conditional summary, applicability boundary,
keywords, and run-local evidence; do not wait for a Skill to be frozen, and do not
claim more than the evidence.
When the rollout provides a credible success candidate, call propose_skill_interface
before ending the turn. Describe reusable preconditions, effects, sensors, Robot SDK
operations, parameters, failure modes, and composition guidance. The Harness will
replace claimed dependencies with the Tools actually used and freeze the interface
only after sensor success/generalization gates pass.

Generalization and autonomous recovery:
- A task is learned only when one unchanged controller handles the required
  scene variations from live observations. Rewriting the whole controller to
  alternate between strategies on successive cases is diagnostic evidence, not
  a generalized solution.
- Derive runtime recovery from the task, current observations, action receipts,
  and prior trials. Reobserve after actions that may invalidate earlier evidence
  or opaque references. Never select behavior from a benchmark task/state id.
- A retry is new evidence only when it changes a meaningful hypothesis,
  capability, or physical strategy. Record attempted strategies and sensor-only
  rejection reasons so the loop does not oscillate.
- In every bounded recovery or feedback-control loop, evaluate the fresh
  post-action observation before declaring the retry budget exhausted. The
  final permitted transition is evidence too; do not discard a success reached
  by the last action merely because the loop counter ended.
- Preserve every constraint and identifying qualifier in the task language, but
  decide how to establish them using the available sensors and Tools. Tool output
  is evidence with a declared contract, not unquestionable ground truth; inspect,
  cross-check, replace, or augment a Tool when observations contradict it.
- Generalization must come from live sensing and adaptive computation. Never
  embed scene coordinates, simulator identities, or per-case branches.

Artifact and Tool contracts:
- Use each Tool's dedicated manual and machine-readable schemas as the default
  calling authority. Do not inspect implementation source merely to learn how to
  call a Tool. Only use read_tool_source selectively when observed runtime evidence
  contradicts or exposes a gap in the manual, or when replacing the implementation;
  then publish an evidence-backed manual correction with revise_tool_manual.
- If a tested or live Tool result contradicts its manual/schema, do not conceal
  the asset defect indefinitely with an undocumented controller fallback. Use
  the run-local evidence, inspect only the relevant source when clarification is
  needed, and publish a corrected manual revision before declaring the task pass
  complete.
- Every executed iteration stores the exact immutable controller source at
  `iterations/iteration_NNN/controller.py`; use these snapshots to recover and
  compose sensor-proven strategies. Do not try to reconstruct old source from
  action traces when its snapshot exists.
- `run_robot_controller` returns a compact indexed RPC summary. Diagnose from
  that summary first. When one perception/grasp Tool output needs more detail,
  call `inspect_execution_event(path="latest_robot_execution", event_index=...,`
  ` max_list_items=...)`; it preserves diagnostic values while bounding large
  candidate arrays. Page the full `robot_execution.json` only if that targeted
  view is insufficient.
- `latest_robot_execution` is the stable authoritative execution reference;
  do not reconstruct or shorten its absolute path. For evidence registration,
  `controller.py` and `executed_controller` both resolve to the immutable source
  snapshot that produced the current episode.
- Before the current iteration has executed, use `previous_robot_execution` for
  the prior episode. Both aliases are stable and avoid copying host paths.
- A registered Tool version is immutable. If any test case fails, that version
  remains `test_failed`; correct the implementation or oracle, register a new
  version, and test the new version. First run the workspace implementation on
  the proposed inputs so expected values are evidence-based rather than guessed.
- Only `status=tested` Tools are deployable. A newly registered capability counts
  as acquired only when its clean version is tested and controller.py actually
  calls that Tool. Do not claim or preserve unused/failed experimental versions
  as learned capability assets.
- Tool registration provenance is fail-closed. Supply public HTTPS sources, an
  explicit training-data declaration, model-card URLs and checkpoint hashes for
  learned models, plus a contamination_check naming the evaluated benchmark,
  method, and result. Never guess provenance; reject a candidate whose task-data
  overlap cannot be ruled out and record that decision in its Capability Gap.
'''


class EvolutionEngine:
    def __init__(self, *, root: str|Path, model, deployment_factory: Callable[[],Any],
                 python: str|Path|None=None, deployment_guidance: dict|None=None,
                 required_success_cases: list[str]|None=None,
                 retry_locked_validation_once: bool=False,
                 success_evidence_protocol: str="sensor-verification-v1",
                 require_task_model: bool=False,
                 require_task_fidelity_review: bool=False,
                 max_coding_passes: int=12,
                 capability_root: str|Path|None=None,
                 experience_root: str|Path|None=None,
                 skill_root: str|Path|None=None,
                 gap_root: str|Path|None=None):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.workspace=TaskWorkspace(self.root/"workspace")
        self.capability_root=Path(capability_root).resolve() if capability_root else self.root/"capabilities"
        self.capability_scope_id=hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        self.capabilities=CapabilityLibrary(self.capability_root,self.workspace.root,
            python=python,allowed_input_roots=[self.root/"episodes",self.root/"iterations"],
            scope_id=self.capability_scope_id)
        self.experience_root=(Path(experience_root).resolve() if experience_root
                              else self.root/"experiences")
        self.experiences=ExperienceLibrary(self.experience_root)
        self.skill_root=Path(skill_root).resolve() if skill_root else self.root/"skills"
        self.skills=SkillLibrary(self.skill_root)
        self.gap_root=Path(gap_root).resolve() if gap_root else self.root/"capability_gaps"
        self.gaps=CapabilityGapLibrary(self.gap_root)
        self.runtime=ControllerRuntime(python=python)
        self.model=model;self.deployment_factory=deployment_factory
        self.guidance=dict(deployment_guidance or {});self.state_path=self.root/"state.json"
        self.required_success_cases=sorted(set(str(x) for x in (required_success_cases or [])))
        self.retry_locked_validation_once=bool(retry_locked_validation_once)
        self.success_evidence_protocol=str(success_evidence_protocol)
        self.require_task_model=bool(require_task_model)
        self.require_task_fidelity_review=bool(require_task_fidelity_review)
        self.max_coding_passes=max(3,min(int(max_coding_passes),12))
        configuration=self.root/"harness_configuration.json"
        if configuration.is_file():
            previous=json.loads(configuration.read_text())
            if Path(previous["capability_root"]).resolve()!=self.capability_root:
                raise RuntimeError("resumed run capability library mismatch")
            configured_scope=previous.get("capability_scope_id")
            if configured_scope is not None and configured_scope!=self.capability_scope_id:
                raise RuntimeError("resumed run Capability scope mismatch")
            previous["capability_scope_id"]=self.capability_scope_id
            if "experience_root" in previous:
                configured=Path(previous["experience_root"]).resolve()
                if configured!=self.experience_root:
                    raise RuntimeError("resumed run Experience Library mismatch")
            else:
                # One-time forward migration for runs created before shared
                # model-authored Experiences existed. No controller or evidence
                # is rewritten; only the new library binding is persisted.
                previous["experience_root"]=str(self.experience_root)
            configured_skill=Path(previous.get("skill_root",self.skill_root)).resolve()
            if configured_skill!=self.skill_root:raise RuntimeError("resumed run Skill Library mismatch")
            previous["skill_root"]=str(self.skill_root)
            configured_gap=Path(previous.get("gap_root",self.gap_root)).resolve()
            if configured_gap!=self.gap_root:raise RuntimeError("resumed run Capability Gap Library mismatch")
            previous["gap_root"]=str(self.gap_root)
            previous["protocol"]="embodied-codex-run-configuration-v2"
            previous["max_coding_passes"]=self.max_coding_passes
            previous["require_task_fidelity_review"]=self.require_task_fidelity_review
            previous["isolation"]={"engineering":"bubblewrap-workspace-v1",
                "controller":"bubblewrap-controller-v1","generated_tool":"bubblewrap-tool-v1"}
            configuration.write_text(json.dumps(previous,indent=2)+"\n")
        else:
            configuration.write_text(json.dumps({
                "protocol":"embodied-codex-run-configuration-v2",
                "capability_root":str(self.capability_root),
                "capability_scope_id":self.capability_scope_id,
                "experience_root":str(self.experience_root),
                "skill_root":str(self.skill_root),
                "gap_root":str(self.gap_root),
                "max_coding_passes":self.max_coding_passes,
                "require_task_fidelity_review":self.require_task_fidelity_review,
                "isolation":{"engineering":"bubblewrap-workspace-v1",
                    "controller":"bubblewrap-controller-v1",
                    "generated_tool":"bubblewrap-tool-v1"}},indent=2)+"\n")

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
        manifests=[item for item in self.capabilities.search(task,limit=8)
                   if item.get("status")=="tested"]
        context={"deployment_guidance":self.guidance,
                 "retrieved_tool_contracts":[{"tool_id":m["tool_id"],
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
            experience_source=None
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
        try:ast.parse(original);binding_method="python_ast_exact_string_constant_rebind"
        except SyntaxError:binding_method="python_tokenize_exact_string_literal_rebind"
        rebound, changed = remap_controller_tool_ids(original, replacements)
        controller.write_text(rebound)
        result = {
            "replacements": replacements,
            "changed_constants": changed,
            "original_controller_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "bound_controller_sha256": hashlib.sha256(rebound.encode()).hexdigest(),
            "method": binding_method,
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
        previous={};history=[];known_by_name={name:set() for name in current}
        if path.is_file():
            record=json.loads(path.read_text())
            previous={str(k):str(v) for k,v in (record.get("current") or {}).items()}
            history=list(record.get("history") or [])
            for transition in history:
                for snapshot_key in ("previous","current"):
                    for name,tool_id in (transition.get(snapshot_key) or {}).items():
                        if str(name) in known_by_name:
                            known_by_name[str(name)].add(str(tool_id))
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
        for name,tool_id in previous.items():
            if name in known_by_name:known_by_name[name].add(tool_id)
        # Include every contract-compatible historical deployment version, not
        # only the immediately previous ledger value. This keeps immutable
        # Controller/Skill snapshots executable after multiple Adapter
        # upgrades and makes infrastructure replays idempotent across restarts.
        manifests=self.capabilities.list_all();current_manifests={}
        for name,tool_id in current.items():
            try:current_manifests[name]=self.capabilities.inspect(tool_id)["manifest"]
            except (FileNotFoundError,RuntimeError):pass
        for manifest in manifests:
            name=str(manifest.get("name") or "")
            if (name not in known_by_name or name not in current_manifests
                    or not manifest.get("execution_owned_by_deployment")):
                continue
            current_manifest=current_manifests[name]
            if (manifest.get("input_schema")==current_manifest.get("input_schema")
                    and manifest.get("output_schema")==current_manifest.get("output_schema")):
                known_by_name[name].add(str(manifest.get("tool_id")))
        replacements={old:current[name] for name,known in known_by_name.items()
                      for old in known if old and old!=current[name]}
        result=self.bind_bootstrap_deployment_tools(replacements)
        if previous and previous!=current:
            history.append({"previous":previous,"current":current,"binding":result})
        path.write_text(json.dumps({
            "protocol":"embodied-codex-deployment-bindings-v1",
            "current":current,"compatible_replacements":replacements,
            "history":history},indent=2)+"\n")
        return result

    def _state(self,task,skill_name):
        if self.state_path.is_file():
            state=json.loads(self.state_path.read_text())
            if state["task"]!=task or state["skill_name"]!=skill_name: raise RuntimeError("run mismatch")
            return state
        return {"task":task,"skill_name":skill_name,"status":"evolving","iterations":[]}
    def _save(self,state):
        tmp=self.state_path.with_suffix(".tmp");tmp.write_text(json.dumps(state,indent=2,default=str)+"\n");tmp.replace(self.state_path)

    def _repeated_failed_strategies(self,state):
        """Return strategies with two same-mechanism sensor failures."""
        counts={};tools={}
        for record in state.get("iterations") or []:
            evidence=record.get("evidence") or {}
            if (evidence.get("sensor_success_candidate") is True
                    or record.get("transient_infrastructure_failure") is not None):
                continue
            result=evidence.get("controller_result") or {}
            mechanism=str(result.get("sensor_failure") or "").strip().casefold()
            if not mechanism:continue
            mechanism=re.sub(r"\b(?:0x)?[0-9a-f]{6,}\b","#",mechanism)
            mechanism=re.sub(r"\d+(?:\.\d+)?","#",mechanism)
            snapshot=Path(str(evidence.get("controller_snapshot") or ""))
            if not snapshot.is_file():continue
            execution_path=(self.root/"iterations"/
                f"iteration_{int(record.get('iteration') or 0):03d}"/"robot_execution.json")
            robot_event_count=None
            if execution_path.is_file():
                try:
                    execution=json.loads(execution_path.read_text()).get("execution") or {}
                    robot_event_count=sum(1 for event in (execution.get("rpc_events") or [])
                        if event.get("method") in {"observe","act","verify"})
                except (OSError,json.JSONDecodeError,TypeError,ValueError):
                    robot_event_count=None
            try:
                if robot_event_count:
                    strategy=_controller_strategy_prefix_sha256(snapshot,robot_event_count)
                    strategy_tools=_controller_tool_ids_before_robot_event(
                        snapshot,robot_event_count)
                else:
                    strategy=_controller_strategy_sha256(snapshot)
                    strategy_tools=_controller_tool_ids(snapshot)
            except (OSError,SyntaxError):continue
            key=(strategy,robot_event_count,mechanism);counts[key]=counts.get(key,0)+1
            tools.setdefault(key,set()).update(strategy_tools)
        rejected={}
        for (strategy,robot_event_count,mechanism),count in counts.items():
            if count>=2:
                entry=rejected.setdefault(strategy,{"failures":[],"prior_tool_ids":set(),
                    "strategy_prefix_sha256":strategy,
                    "robot_event_count":robot_event_count})
                entry["failures"].append(f"{mechanism} ({count}x)")
                entry["prior_tool_ids"].update(
                    tools[(strategy,robot_event_count,mechanism)])
        for entry in rejected.values():entry["prior_tool_ids"]=sorted(entry["prior_tool_ids"])
        return rejected

    def _current_task_fidelity_rejection(self,controller_path: Path):
        binding_path=self.workspace.root/"task_fidelity_binding.json"
        if not controller_path.is_file() or not binding_path.is_file():return None
        try:binding=json.loads(binding_path.read_text())
        except (OSError,json.JSONDecodeError):return None
        if not isinstance(binding,dict):return None
        current_sha=hashlib.sha256(controller_path.read_bytes()).hexdigest()
        review=binding.get("review") or {}
        if (binding.get("controller_sha256")!=current_sha
                or not isinstance(review,dict) or review.get("approved") is not False):
            return None
        return {"controller_sha256":current_sha,
                "issues":[str(item)[:900] for item in (review.get("issues") or [])[:4]],
                "rule":("Modify executable task grounding before calling "
                        "run_robot_controller again. The unchanged source is cached as "
                        "rejected and must not be retried.")}

    def _persistent_gap_acquisition_gate(self,task: str,previous_record):
        """Require acquisition only after a task Gap survives a second revision."""
        if previous_record is None:return None
        evidence=previous_record.get("evidence") or {}
        if (evidence.get("sensor_success_candidate") is True
                or evidence.get("transient_infrastructure_failure") is not None):return None
        exact=str(task).strip().casefold()
        candidates=[];names=set()
        for summary in self.gaps.list_summaries():
            if str(summary.get("task") or "").strip().casefold()==exact:
                names.add(str(summary.get("name") or ""))
        for name in names:
            try:item=self.gaps.latest_for_name(name)
            except (AssetError,FileNotFoundError):continue
            validation=(item or {}).get("task_validation") or {}
            authoritative=validation.get("authoritative_outcome")
            if (item and item.get("previous_gap_id")
                    and item.get("status") in {"diagnosed","searching","integrating","rejected"}
                    and authoritative!="success"):
                candidates.append(item)
        if not candidates:return None
        latest=max(candidates,key=lambda item:int(item.get("version") or 0))
        snapshot=Path(str(evidence.get("controller_snapshot") or ""))
        try:baseline=sorted(_controller_tool_ids(snapshot)) if snapshot.is_file() else []
        except (OSError,SyntaxError):baseline=[]
        return {"gap_id":latest["gap_id"],"gap_name":latest["name"],
            "baseline_tool_ids":baseline,
            "rule":("Before another physical episode, perform audited public research, "
                    "revise this unresolved Gap, and bind a newly tested Tool that passes "
                    "independent Gap-integration review. Search or an integrating label alone "
                    "does not satisfy acquisition; neither does a one-off Controller implementation.")}

    def _accept_generalization_success(self,state,record,execution):
        if not self.required_success_cases:return True
        report=execution.get("sensor_report") or {}
        case=str(report.get("_harness_case_id") or "")
        if case not in self.required_success_cases:
            raise RuntimeError("deployment omitted required Harness case id")
        program_sha=(execution.get("execution") or {}).get("program_sha256")
        gate=state.setdefault("generalization_gate",{
            "required_cases":self.required_success_cases,"successes_by_program":{},
            "evidence_protocol":self.success_evidence_protocol})
        coverage=gate["successes_by_program"].setdefault(program_sha,[])
        if case not in coverage:coverage.append(case);coverage.sort()
        record["generalization_case"]=case
        record["generalization_coverage"]={"program_sha256":program_sha,
            "passed_cases":list(coverage),"required_cases":self.required_success_cases}
        self._save(state)
        return set(self.required_success_cases).issubset(coverage)

    def _freeze_success(self,*,state,record,index,task,skill_name,task_model,
                        execution,evidence,artifact):
        # Always freeze the immutable program that produced the accepted
        # rollout, never a potentially edited workspace successor.
        controller=Path(execution["controller_snapshot"]).resolve()
        used=sorted({event.get("arguments",{}).get("tool_id")
            for event in (execution.get("execution") or {}).get("rpc_events",[])
            if event.get("method")=="use" and event.get("arguments",{}).get("tool_id")})
        interface_path=artifact/"skill_interface.json"
        proposed=(json.loads(interface_path.read_text()).get("interface")
                  if interface_path.is_file() else None)
        rpc_events=(execution.get("execution") or {}).get("rpc_events",[])
        actual_operations={str(event.get("method")) for event in rpc_events
            if event.get("method") in {"observe","use","act","verify","record"}}
        actual_sensors={str(event.get("arguments",{}).get("channel"))
            for event in rpc_events if event.get("method")=="observe"
            and event.get("arguments",{}).get("channel")}
        if proposed is None:
            proposed={
                "preconditions":["The declared task entities are observable and required actions are available."],
                "effects":[task],"required_sensors":sorted(actual_sensors),
                "required_robot_operations":sorted(actual_operations),
                "parameters":[{"name":"instruction","source":"robot.instruction"}],
                "failure_modes":["A sensor, Tool, motion, or verification precondition may fail."],
                "composition_notes":"Fallback interface derived from the successful execution trace."}
        else:
            if not proposed.get("preconditions"):
                proposed["preconditions"]=[
                    "The declared task entities are observable and required actions are available."]
            if not proposed.get("effects"):proposed["effects"]=[task]
            if not proposed.get("failure_modes"):
                proposed["failure_modes"]=[
                    "A sensor, Tool, motion, or verification precondition may fail."]
            proposed["required_robot_operations"]=sorted(actual_operations|{
                str(item) for item in proposed.get("required_robot_operations",[])})
            proposed["required_sensors"]=sorted(actual_sensors|{
                str(item) for item in proposed.get("required_sensors",[])})
        accepted_controller_sha=(execution.get("execution") or {}).get("program_sha256")
        development_evidence=sorted(
            self.root.glob("iterations/iteration_*/robot_execution.json"))
        promotions=[]
        for tool_id in used:
            manifest=self.capabilities.inspect(tool_id)["manifest"]
            if (not manifest.get("execution_owned_by_deployment")
                    and manifest.get("visibility")=="task_local"):
                promotions.append(self.capabilities.promote_for_reuse(
                    tool_id,evidence_paths=development_evidence,
                    controller_sha256=accepted_controller_sha,
                    required_case_handles=self.required_success_cases))
        skill=self.skills.freeze(name=skill_name,task=task,controller=controller,
            evidence={"iteration":index,"sensor_only":True,"report":evidence},
            tool_ids=used,tools=self.capabilities,
            experience={"iterations":state["iterations"],
                        "generalization_gate":state.get("generalization_gate"),
                        "tool_promotions":promotions,
                        "task_model":task_model},task_model=task_model,
            interface=proposed,evidence_paths=sorted([
                *self.root.glob("iterations/iteration_*/robot_execution.json"),
                *self.root.glob("episodes/episode_*/adapter_trace.json"),
                *self.root.glob("episodes/episode_*/rollout.mp4")]))
        state.update({"status":"sensor_success","skill":skill});self._save(state)
        return state

    @staticmethod
    def _brief(execution):
        if not execution:return None
        run=execution.get("execution",{}); events=[]; records=[]
        for event in run.get("rpc_events",[]):
            if event.get("method")=="record":
                records.append(_compact_evidence_value(
                    event.get("arguments",{}).get("event"),max_list_items=6))
        rpc_events=list(run.get("rpc_events",[]))
        for event_index,event in list(enumerate(rpc_events))[-24:]:
            item={"event_index":event_index,"method":event.get("method")}
            if event.get("method")=="act":
                item["action"]=_compact_evidence_value(
                    event.get("arguments",{}).get("action"),max_list_items=6)
                result=event.get("result",{});item["result"]={k:result.get(k) for k in
                    ("reached","step","eef_after","gripper_qpos","target_xyz") if k in result}
            elif event.get("method")=="verify":
                result=event.get("result",{});item["verifier"]=event.get("arguments",{}).get("verifier")
                item["result"]=_compact_evidence_value(
                    {k:v for k,v in result.items() if k not in ("frame","cameras")},
                    max_list_items=6)
            elif event.get("method")=="use": item["tool_id"]=event.get("arguments",{}).get("tool_id")
            elif event.get("method")=="observe":
                result=event.get("result",{}); cameras=result.get("cameras") or {}
                item["result"]={"frame_id":result.get("frame_id"),"step":result.get("step"),
                    "rgb_paths":{name:value.get("rgb_path") for name,value in cameras.items()
                        if isinstance(value,dict) and value.get("rgb_path")}}
            if event.get("error"):item["error"]=event["error"]
            events.append(item)
        sensor_report=_compact_evidence_value({
            k:v for k,v in (execution.get("sensor_report") or {}).items()
            if not str(k).startswith("_harness_")},max_list_items=6)
        return {"controller_path":execution.get("controller_path"),
                "controller_snapshot":execution.get("controller_snapshot"),
                "completed":run.get("completed"),"error":run.get("error"),
                "controller_result":_compact_evidence_value(
                    run.get("result"),max_list_items=6),"rpc_evidence":events,
                "controller_records":records[-32:],
                "sensor_report":sensor_report,
                "sensor_success_candidate":execution.get("sensor_success_candidate"),
                "transient_infrastructure_failure":execution.get(
                    "transient_infrastructure_failure"),
                "full_execution_artifact":str(Path(
                    execution.get("controller_snapshot") or "").parent/"robot_execution.json"),
                "execution_artifact_ref":"previous_robot_execution"}

    @staticmethod
    def _authoritative_outcome(previous):
        """Put task-level acceptance evidence ahead of verbose local claims."""
        if not isinstance(previous,dict):return None
        report=previous.get("sensor_report") or {}
        independent=report.get("independent_task_outcome") or {}
        controller=previous.get("controller_result") or {}
        local_claims={
            "controller_verified":controller.get("verified"),
            "controller_sensor_failure":controller.get("sensor_failure"),
            "sensor_verification_passed":report.get("sensor_verification_passed"),
            "controller_visual_verification_passed":report.get(
                "controller_visual_verification_passed")}
        local_positive=any(value is True for key,value in local_claims.items()
                           if key!="controller_sensor_failure")
        local_negative=any(value is False for key,value in local_claims.items()
                           if key!="controller_sensor_failure")
        independent_verified=independent.get("verified")
        conflict=(independent_verified is False and local_positive) or (
            independent_verified is True and local_negative)
        task_outcome={key:independent.get(key) for key in (
            "verified","source_relation_satisfied","target_relation_satisfied",
            "contradiction","reason","confidence") if key in independent}
        candidate=previous.get("sensor_success_candidate")
        return {
            "kernel_decision":"success_candidate" if candidate is True else "failure",
            "sensor_success_candidate":candidate,
            "independent_task_level_outcome":task_outcome,
            "local_controller_and_verifier_claims":local_claims,
            "evidence_conflict":bool(conflict),
            "evidence_precedence":(
                "The kernel sensor_success_candidate is the acceptance decision. "
                "The independent task-level outcome checks the exact language relation. "
                "A positive local attachment, support, or geometric verifier cannot override "
                "a negative independent task-level outcome. Any conflict remains a failure; "
                "diagnose it before publishing a causal Experience or changing the Controller.")}

    @staticmethod
    def _retrieved_asset_index(*, experiences, skills, gaps):
        """Return bounded catalog entries; full assets require an explicit inspect call."""
        def text(value,maximum):
            value=str(value or "")
            return value if len(value)<=maximum else value[:maximum-3]+"..."
        experience_rows=[{
            "experience_id":item.get("experience_id"),"name":item.get("name"),
            "summary":text(item.get("summary"),160),
            "applicability":text(item.get("applicability"),80),
            "keywords":list(item.get("keywords") or [])[:4],
            "status":item.get("status"),"retrieval_score":item.get("retrieval_score")}
            for item in experiences]
        skill_rows=[]
        for item in skills:
            interface=item.get("interface") or {}
            skill_rows.append({
                "skill_id":item.get("skill_id"),"task":text(item.get("task"),140),
                "status":item.get("status"),
                "effects":[text(value,140) for value in (interface.get("effects") or [])[:2]],
                "required_sensors":list(interface.get("required_sensors") or [])[:6],
                "tool_ids":list(item.get("tool_ids") or [])[:8],
                "retrieval_score":item.get("retrieval_score")})
        gap_rows=[]
        for item in gaps:
            capability=item.get("required_capability") or {}
            gap_rows.append({
                "gap_id":item.get("gap_id"),"name":item.get("name"),
                "task":text(item.get("task"),160),"status":item.get("status"),
                "failure_summary":text(item.get("failure_summary"),160),
                "selected_diagnosis":text(item.get("selected_diagnosis"),160),
                "required_capability_kind":text(capability.get("kind"),80),
                "authoritative_outcome":item.get("authoritative_outcome"),
                "model_claim_conflicts_with_evidence":item.get(
                    "model_claim_conflicts_with_evidence",False),
                "retrieval_score":item.get("retrieval_score")})
        return experience_rows,skill_rows,gap_rows

    @staticmethod
    def _retrieved_tool_index(manifests):
        """Manual-first catalog; schemas are loaded only for a selected Tool."""
        rows=[]
        for item in manifests:
            description=str(item.get("description") or "")
            if len(description)>180:description=description[:177]+"..."
            input_schema=item.get("input_schema") or {}
            output_schema=item.get("output_schema") or {}
            rows.append({
                "tool_id":item.get("tool_id"),"description":description,
                "input_fields":sorted((input_schema.get("properties") or {}).keys()),
                "required_inputs":list(input_schema.get("required") or []),
                "output_fields":sorted((output_schema.get("properties") or {}).keys()),
                "execution_owned_by_deployment":item.get(
                    "execution_owned_by_deployment",False)})
        return rows

    @staticmethod
    def _prompt_previous_evidence(previous):
        """Keep decisive evidence in-context and leave full detail queryable."""
        if not isinstance(previous,dict):return None
        def tail(value,limit):
            if isinstance(value,list):return value[-limit:]
            # Compatibility with runs whose prompt-facing state was already
            # bounded by _compact_evidence_value.  Never turn the summary dict
            # into a meaningless list of its keys.
            if isinstance(value,dict) and value.get("type")=="list":
                head=value.get("head")
                return list(head)[-limit:] if isinstance(head,list) else []
            return []
        report=previous.get("sensor_report") or {}
        compact_report={key:report.get(key) for key in (
            "sensor_verification_passed","controller_visual_verification_passed",
            "independent_task_outcome","outcome_observations","final_proprioception",
            "final_step","rollout_path","trace_path") if key in report}
        result={key:previous.get(key) for key in (
            "controller_path","controller_snapshot","completed","error",
            "sensor_success_candidate","transient_infrastructure_failure",
            "full_execution_artifact","execution_artifact_ref") if key in previous}
        controller=previous.get("controller_result") or {}
        controller_summary={key:controller.get(key) for key in (
            "verified","sensor_failure","reason","detail","source_vacated",
            "source_transport_verified","source_vacancy_method","target_xy_error_m",
            "vertical_offset_m","support_overlap_fraction","criterion") if key in controller}
        rpc=[]
        for item in tail(previous.get("rpc_evidence"),6):
            if not isinstance(item,dict):continue
            rpc.append({key:item.get(key) for key in (
                "event_index","method","tool_id","verifier","error","result") if key in item})
        result.update({
            "controller_result":_compact_evidence_value(
                controller_summary,max_list_items=2,max_depth=4),
            "rpc_evidence_tail":_compact_evidence_value(rpc,max_list_items=6,max_depth=4),
            "rpc_detail_rule":("Use inspect_execution_event/query_run_json on "
                "previous_robot_execution for complete selected events and records."),
            "sensor_report":_compact_evidence_value(
                compact_report,max_list_items=3,max_depth=5)})
        return result

    @staticmethod
    def _prompt_deployment_guidance(guidance):
        """Keep a callable SDK index in context; load verbose sections on demand."""
        guidance=dict(guidance or {})
        contract=guidance.get("robot_sdk_contract") or {}
        methods={name:{key:value.get(key) for key in ("signature","returns") if key in value}
                 for name,value in (contract.get("methods") or {}).items()}
        actions={}
        for name,value in (contract.get("actions") or {}).items():
            actions[name]={"required":list(value.get("required") or []),
                "optional_fields":sorted((value.get("optional") or {}).keys())}
            for key in ("any_of","enum","field_semantics","rule"):
                if key in value:actions[name][key]=value[key]
        verifiers={}
        for name,value in (contract.get("verifiers") or {}).items():
            verifiers[name]={"required":list(value.get("required") or []),
                "optional_fields":sorted((value.get("optional") or {}).keys())}
            if "rule" in value:verifiers[name]["rule"]=value["rule"]
        guidance["robot_sdk_contract"]={
            "protocol":contract.get("protocol"),"methods":methods,"actions":actions,
            "verifiers":verifiers,"reference_rules":contract.get("reference_rules") or [],
            "full_manual":"inspect_robot_sdk_contract(section=...)"}
        return guidance

    def _workspace_index(self, limit: int = 16):
        """Return a bounded, content-free index for cross-pass recovery."""
        entries=[]
        for path in self.workspace.root.rglob("*"):
            if (not path.is_file() or "__pycache__" in path.parts
                    or path.name.endswith((".pyc", ".pyo"))):
                continue
            stat=path.stat()
            entries.append({
                "path":str(path.relative_to(self.workspace.root)),
                "bytes":stat.st_size,
                "modified_unix":stat.st_mtime})
        entries.sort(key=lambda item:(-item["modified_unix"],item["path"]))
        return entries[:max(0,int(limit))]

    def _workspace_temporal_warnings(self):
        """Flag notes whose names claim evidence that did not yet exist."""
        warnings=[]
        pattern=re.compile(r"iteration_(\d+).*post[_-]?run",re.IGNORECASE)
        for path in self.workspace.root.rglob("*"):
            if not path.is_file():continue
            match=pattern.search(path.name)
            if match is None:continue
            execution=(self.root/"iterations"/
                       f"iteration_{int(match.group(1)):03d}"/"robot_execution.json")
            if (not execution.is_file()
                    or path.stat().st_mtime < execution.stat().st_mtime):
                warnings.append({
                    "path":str(path.relative_to(self.workspace.root)),
                    "warning":"named_post_run_but_created_before_execution_commit",
                    "rule":("Do not use this note as post-run evidence. Read the immutable "
                            "robot_execution.json through the correct latest/previous alias.")})
        return sorted(warnings,key=lambda item:item["path"])[:16]

    @staticmethod
    def _coding_pass_limits(coding_pass: int):
        """Diagnosis is broad once; correction passes must act on persisted evidence."""
        if int(coding_pass)<=1:
            return {"max_evidence_deliveries":18,
                    "post_evidence_pause_max_turns":4,
                    "post_mutation_max_turns":8}
        return {"max_evidence_deliveries":6,
                "max_working_memory_deliveries":4,
                "post_evidence_pause_max_turns":2,
                "post_duplicate_read_max_turns":1,
                # Four turns were insufficient in real multi-task recovery:
                # after a persisted edit, the model needed several paged reads
                # to verify a 300+ line Controller before compile/preflight/run.
                # The existing read budgets still stop open-ended inspection;
                # this deadline now permits one complete correction transaction.
                "post_mutation_max_turns":8}

    @staticmethod
    def _controller_differs_from(path: Path, semantic_sha256: str) -> bool:
        try:return _controller_semantic_sha256(path)!=semantic_sha256
        except (OSError,SyntaxError):return True

    @staticmethod
    def _coding_pass_handoff(result: dict):
        """Compact the prior Agent pass into an actionable correction receipt."""
        recent=[]
        for row in (result.get("tool_results") or [])[-8:]:
            item={"name":row.get("name"),"ok":row.get("ok")}
            if row.get("error"):item["error"]=str(row["error"])[:500]
            if row.get("duplicate_read_suppressed"):
                item["duplicate_read_suppressed"]=True
            if row.get("evidence_acquisition_paused"):
                item["evidence_acquisition_paused"]=True
            value=row.get("result")
            if isinstance(value,dict):
                # A diagnosis note is the engineering equivalent of a TODO
                # handed from one coding session to the next.  Previously the
                # handoff retained only `write_file: ok`, so the correction
                # pass did not know which persisted plan to execute and began
                # the entire investigation again.  Preserve identifiers and
                # mutation facts, never bulk file contents or sensor payloads.
                for key in ("path","workspace_mutated_paths","exit_code",
                            "controller_semantic_progress"):
                    if key in value:item[key]=value[key]
                # Short compiler/search output is often the exact correction
                # evidence needed by the next bounded pass (for example the
                # actual line ending around a failed exact replacement).  Keep
                # it bounded instead of forcing the model to repeat the same
                # command after every context reset.
                if row.get("name")=="run_command":
                    if isinstance(value.get("argv"),list):
                        item["argv"]=[str(part)[:200] for part in value["argv"][:12]]
                    output=value.get("output")
                    if isinstance(output,str) and output.strip():
                        item["command_output"]=output[:2000]
            recent.append(item)
        action_artifacts=[]
        for row in result.get("tool_results") or []:
            if row.get("name") not in {"write_file","replace_in_file",
                                       "replace_file_lines"}:
                continue
            value=row.get("result") or {}
            path=value.get("path") if isinstance(value,dict) else None
            if path and path!="controller.py" and path not in action_artifacts:
                action_artifacts.append(path)
        handoff={"completed":bool(result.get("completed")),
                 "error":result.get("error"),"recent_tool_results":recent}
        if action_artifacts:
            handoff["persisted_action_artifacts"]=action_artifacts[-4:]
            handoff["continuation_rule"]=(
                "Treat the newest persisted action artifact as the prior pass's "
                "engineering TODO. Read it at most once if its contents are not "
                "already in context, then implement/test its discriminating action; "
                "do not restart diagnosis.")
        return handoff

    @staticmethod
    def _recover_uncommitted_pass_handoff(trace_path: Path):
        """Recover actionable coding state left before a process restart.

        Robot execution is transactionally committed elsewhere.  This helper
        handles the complementary case: an iteration has no robot record yet,
        but a prior process already persisted a diagnosis or executable
        workspace mutation.  Resume as a correction pass instead of granting a
        fresh broad investigation budget.
        """
        if not trace_path.is_file():return None
        rows=[]
        for line in trace_path.read_text(errors="replace").splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get("type")=="tool_result":rows.append(item)
        recent=[];artifacts=[];progress=False
        for row in rows:
            result=row.get("result") or {}
            if row.get("name") in {"write_file","replace_in_file","replace_file_lines"}:
                path=result.get("path") if isinstance(result,dict) else None
                if path and path!="controller.py" and path not in artifacts:
                    artifacts.append(path);progress=True
            if isinstance(result,dict) and (result.get("_embodied_codex_engineering_progress")
                    or result.get("workspace_mutated_paths")):
                progress=True
        if not progress:return None
        for row in rows[-8:]:
            item={"name":row.get("name"),"ok":row.get("ok")}
            if row.get("error"):item["error"]=str(row["error"])[:500]
            result=row.get("result")
            if isinstance(result,dict):
                for key in ("path","workspace_mutated_paths","exit_code",
                            "controller_semantic_progress"):
                    if key in result:item[key]=result[key]
                if row.get("name")=="run_command":
                    if isinstance(result.get("argv"),list):
                        item["argv"]=[str(part)[:200] for part in result["argv"][:12]]
                    output=result.get("output")
                    if isinstance(output,str) and output.strip():
                        item["command_output"]=output[:2000]
            recent.append(item)
        handoff={"completed":False,"error":"process_restarted_before_robot_episode",
                 "recent_tool_results":recent,"recovered_from_trace":str(trace_path)}
        if artifacts:
            handoff["persisted_action_artifacts"]=artifacts[-4:]
            handoff["continuation_rule"]=(
                "A prior process already persisted this engineering TODO. Read the newest "
                "artifact at most once, implement/test it, and run the Controller; do not "
                "restart diagnosis.")
        return handoff

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
        # A robot transaction is committed before the post-rollout model turn.
        # If the host was interrupted in that narrow window, recover an already
        # accepted rollout without repeating physical actions or freezing an
        # unexecuted workspace edit.
        pending=next((row for row in reversed(state["iterations"])
                      if row.get("agent_error")=="post_execution_agent_pending"),None)
        task_model=None
        if pending and (pending.get("evidence") or {}).get("sensor_success_candidate") is True:
            # A committed success may still need to be frozen after a host
            # interruption, so build/load its semantic contract before the
            # ordinary iteration-budget fast path below.
            task_model=self._task_model(task)
            pending_index=int(pending["iteration"])
            artifact=self.root/"iterations"/f"iteration_{pending_index:03d}"
            execution_path=artifact/"robot_execution.json"
            execution=json.loads(execution_path.read_text())
            if execution.get("sensor_success_candidate") is not True:
                raise RuntimeError("committed success evidence disagrees with robot execution")
            snapshot=Path(execution["controller_snapshot"]).resolve()
            shutil.copy2(snapshot,self.workspace.root/"controller.py")
            pending.update({"agent_completed":False,
                "agent_error":"recovered_after_post_execution_interruption",
                "coding_passes":pending.get("coding_passes")})
            complete=self._accept_generalization_success(state,pending,execution)
            self._save(state)
            if complete:
                return self._freeze_success(state=state,record=pending,index=pending_index,
                    task=task,skill_name=skill_name,task_model=task_model,
                    execution=execution,evidence=pending["evidence"],artifact=artifact)
        # Migrate previously persisted zero-action transient Tool outages into
        # infrastructure-only trials. Their immutable evidence remains, but
        # they must not consume the task capability budget or development case.
        migrated=False
        for record in state["iterations"]:
            if record.get("evidence") is None or record.get(
                    "transient_infrastructure_failure") is not None:continue
            execution_path=(self.root/"iterations"/
                f"iteration_{int(record['iteration']):03d}"/"robot_execution.json")
            if not execution_path.is_file():continue
            try:full_execution=json.loads(execution_path.read_text())
            except (OSError,json.JSONDecodeError):continue
            transient=(full_execution.get("transient_infrastructure_failure") or
                       transient_infrastructure_failure(
                           full_execution.get("execution") or {},
                           full_execution.get("sensor_report") or {}))
            if transient is not None:
                record["transient_infrastructure_failure"]=transient
                (record.get("evidence") or {})["transient_infrastructure_failure"]=transient
                migrated=True
        if migrated:self._save(state)
        episode_count=sum(1 for record in state["iterations"]
                          if record.get("evidence") is not None
                          and not record.get("transient_infrastructure_failure"))
        # Do not spend model calls creating a new semantic task model for an
        # already exhausted historical frontier.  This matters when the
        # semantic gate is introduced into a resumed multi-task campaign: only
        # tasks that can still execute a controller need a task model.  A
        # committed success was handled above and therefore cannot be skipped.
        if episode_count>=max_iterations:
            return state
        if task_model is None:
            task_model=self._task_model(task)
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
            task_fidelity_reviewer=None
            if self.require_task_fidelity_review:
                task_fidelity_reviewer=lambda **payload:review_controller_task_fidelity(
                    model=self.model,
                    trace_path=artifact/"controller_task_fidelity_review.jsonl",
                    **payload)
            previous_record=next((record for record in reversed(state["iterations"])
                                  if record.get("evidence") is not None),None)
            previous_transient=(previous_record or {}).get(
                "transient_infrastructure_failure")
            acquisition_advisory=self._persistent_gap_acquisition_gate(
                task,previous_record)
            surface=EngineeringSurface(workspace=self.workspace,capabilities=self.capabilities,
                runtime=self.runtime,deployment_factory=self.deployment_factory,
                artifact_dir=artifact,task_model=task_model,
                semantic_reviewer=semantic_reviewer,
                task_instruction=task,
                task_fidelity_reviewer=task_fidelity_reviewer,
                acquisition_reviewer=None,
                sdk_contract=self.guidance.get("robot_sdk_contract"),
                active_deployment_tool_ids=self.guidance.get("active_deployment_tool_ids"),
                execution_observer=persist_robot_execution,experiences=self.experiences,
                skills=self.skills,gaps=self.gaps,
                # Strategy repetition and capability acquisition are model
                # decisions informed by evidence, not kernel execution gates.
                # The hard boundary here is limited to SDK/static contracts,
                # process isolation, and one physical episode per iteration.
                rejected_controller_semantic_sha256=None,
                rejected_controller_strategy_failures=None,
                controller_tool_replacements=dict((self.guidance.get(
                    "deployment_dependency_binding") or {}).get("replacements") or {}),
                required_acquisition_gap_id=None,
                acquisition_baseline_tool_ids=None)
            # Keep the authoritative in-memory record intact here.  The two
            # prompt consumers below explicitly project bounded fields.  A
            # prior blanket compaction converted long evidence lists to
            # summary dicts, after which _prompt_previous_evidence accidentally
            # iterated their keys and injected [type,count,head,remaining].
            previous=(previous_record.get("evidence")
                      if previous_record is not None else None)
            # Runs created before context-bounded evidence remain resumable.
            # Preserve the full immutable execution on disk and migrate only
            # the prompt-facing view instead of rewriting historical evidence.
            if isinstance(previous,dict) and "full_execution_artifact" not in previous:
                prior_index=int(previous_record.get("iteration"))
                previous["full_execution_artifact"]=str((self.root/"iterations"/
                    f"iteration_{prior_index:03d}"/"robot_execution.json").resolve())
                previous["execution_artifact_ref"]="previous_robot_execution"
            prompt_previous=self._prompt_previous_evidence(previous)
            active_deployment=set(str(x) for x in
                                  (self.guidance.get("active_deployment_tool_ids") or []))
            query=task+" "+json.dumps(prompt_previous or {},default=str)[:2500]
            tested_manifests=[manifest for manifest in self.capabilities.search(query,limit=8)
                if manifest.get("status")=="tested"
                if not manifest.get("execution_owned_by_deployment")
                or not active_deployment or manifest.get("tool_id") in active_deployment]
            existing_ids={item.get("tool_id") for item in tested_manifests}
            for tool_id in sorted(active_deployment):
                if tool_id not in existing_ids:
                    manifest=self.capabilities.inspect(tool_id)["manifest"]
                    if manifest.get("status")=="tested":tested_manifests.append(manifest)
            experience_index,skill_index,gap_index=self._retrieved_asset_index(
                experiences=self.experiences.search(query,limit=4),
                skills=self.skills.search(query,limit=2),
                gaps=self.gaps.search(query,limit=3))
            base_instruction={"task":task,"iteration":index,"robot_episode":episode_count+1,
                "authoritative_previous_outcome":self._authoritative_outcome(previous),
                "previous_sensor_evidence":prompt_previous,
                "deployment_guidance":self._prompt_deployment_guidance(self.guidance),
                "persistent_workspace_index":self._workspace_index(),
                "workspace_temporal_warnings":self._workspace_temporal_warnings(),
                "retrieved_tool_index":self._retrieved_tool_index(tested_manifests),
                "retrieved_experiences":experience_index,
                "retrieved_skills":skill_index,
                "retrieved_capability_gaps":gap_index,
                "capability_gap_advisory":acquisition_advisory,
                "tool_deployment_contract":(
                    "Every retrieved status=tested Agent Tool is dynamically bound to each "
                    "fresh deployment. active_deployment_tool_ids selects only Adapter-owned "
                    "seed versions and does not exclude Agent-authored Tools."),
                "asset_retrieval":"Call search_assets for broader candidates, then inspect only selected assets.",
                "requirement":"engineer and run one complete controller program",
                "failed_rollout_replay_guidance":(
                    "Use the prior sensor evidence to decide whether to rerun, modify the "
                    "Controller, or acquire a capability. The kernel does not force that "
                    "strategy decision; record the reason for a deliberate replay.")}
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
                    "required_case_count":len(gate["required_cases"]),
                    "coverage_by_program":{program:len(cases) for program,cases in
                                           gate["successes_by_program"].items()},
                    "rule":"the identical controller hash must pass every case; do not add state branches"}
            passes=[];locked_validation=False;locked_retry=False
            locked_retry_source_iteration=None
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
            automatic_post_action_replay=(
                isinstance(previous_transient,dict) and
                previous_transient.get("kind")==
                    "transient_post_action_sensor_verifier_outage")
            if (automatic_post_action_replay and previous
                    and controller_path.is_file()):
                # A transient infrastructure result is not task evidence. The
                # coding model must not react to it. Restore the immutable
                # source that actually ran because the model may have edited
                # the persistent workspace after receiving the RPC result.
                # Older kernels may have let GPT mutate the Controller after
                # an outage. Consecutive outage descendants are contaminated
                # by that false evidence, so return to the first immutable
                # Controller in the chain.
                replay_source=(_post_action_transient_replay_source(
                    state["iterations"]) or previous_record)
                source_iteration=int(replay_source["iteration"])
                source_evidence=replay_source.get("evidence") or {}
                snapshot=Path(str(source_evidence.get("controller_snapshot") or ""))
                if not snapshot.is_absolute():
                    snapshot=(self.root/"iterations"/
                              f"iteration_{source_iteration:03d}"/"controller.py")
                snapshot=snapshot.resolve()
                if self.root not in snapshot.parents or not snapshot.is_file():
                    raise RuntimeError("transient replay controller snapshot is unavailable")
                shutil.copy2(snapshot,controller_path)
                source_execution_path=(self.root/"iterations"/
                    f"iteration_{source_iteration:03d}"/"robot_execution.json")
                source_execution=json.loads(source_execution_path.read_text())
                replay_case=str((source_execution.get("sensor_report") or {}).get(
                    "_harness_case_id") or "")
                if replay_case:
                    select_case=getattr(self.deployment_factory,"select_case",None)
                    if not callable(select_case):
                        raise RuntimeError(
                            "deployment Factory cannot select the opaque transient replay case")
                    select_case(replay_case)
                surface.run_robot_controller("controller.py")
                agent_result={"completed":True,"error":None,"tool_results":[]}
                locked_validation=True
                locked_retry=True
                locked_retry_source_iteration=source_iteration
            elif (self.retry_locked_validation_once and previous
                    and controller_path.is_file()):
                # An infrastructure correction must replay the immutable
                # program that produced the contradicted success evidence,
                # not an arbitrary later workspace edit. A coding pass may
                # already have reacted to the stale verifier before the host
                # was restarted. Prefer the newest execution whose independent
                # sensor outcome was positive; otherwise retain the historical
                # current-controller behavior for non-vision deployments.
                for candidate in reversed(state["iterations"]):
                    candidate_evidence=candidate.get("evidence") or {}
                    outcome=((candidate_evidence.get("sensor_report") or {})
                             .get("independent_task_outcome") or {})
                    snapshot=Path(str(candidate_evidence.get("controller_snapshot") or ""))
                    if outcome.get("verified") is not True or not snapshot.is_absolute():
                        continue
                    snapshot=snapshot.resolve()
                    if (self.root not in snapshot.parents or not snapshot.is_file()):
                        continue
                    shutil.copy2(snapshot,controller_path)
                    # Startup already migrated the persistent workspace to
                    # current Adapter-owned Tool versions. Restoring an older
                    # immutable snapshot must not silently undo that binding.
                    # Rebind exact string constants only; task logic and
                    # Agent-authored analytic Tools remain byte-for-byte as
                    # authored. The binding ledger is supplied by the Adapter
                    # and retained in the replay record below.
                    binding=(self.guidance.get("deployment_dependency_binding") or {})
                    replacements={str(old):str(new) for old,new in
                                  (binding.get("replacements") or {}).items()
                                  if old and new and old!=new}
                    if replacements:
                        restored=controller_path.read_text()
                        rebound,changed=remap_controller_tool_ids(restored,replacements)
                        if changed:controller_path.write_text(rebound)
                    locked_retry_source_iteration=int(candidate["iteration"])
                    # Replay the exact opaque case that produced the
                    # contradicted evidence. The private handle is read from
                    # the full artifact and never copied into model context.
                    source_execution_path=(self.root/"iterations"/
                        f"iteration_{locked_retry_source_iteration:03d}"/
                        "robot_execution.json")
                    source_execution=json.loads(source_execution_path.read_text())
                    replay_case=str((source_execution.get("sensor_report") or {}).get(
                        "_harness_case_id") or "")
                    if replay_case:
                        select_case=getattr(self.deployment_factory,"select_case",None)
                        if not callable(select_case):
                            raise RuntimeError(
                                "deployment Factory cannot select the opaque infrastructure replay case")
                        select_case(replay_case)
                    break
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
                recovered_handoff=self._recover_uncommitted_pass_handoff(
                    artifact/"agent_trace.jsonl")
                for pass_attempt in range(1,self.max_coding_passes+1):
                    coding_pass=pass_attempt+(1 if recovered_handoff else 0)
                    instruction=dict(base_instruction);instruction["coding_pass"]=coding_pass
                    if coding_pass>1:
                        instruction["correction"]=("The prior coding pass ended without a robot "
                            "episode. Use the persisted diagnosis and prior-pass handoff; do not "
                            "create or revise another diagnosis/hypothesis note in a correction "
                            "pass. Read only the needed Controller pages, make a semantically "
                            "executable code/Tool change (prefer replace_file_lines after the "
                            "relevant page is known), test it, and call run_robot_controller "
                            "once before ending this pass.")
                        instruction["prior_coding_pass_handoff"]=(
                            self._coding_pass_handoff(passes[-1]) if passes
                            else recovered_handoff)
                    # A controller in a task workspace with no committed robot
                    # episode is necessarily unexecuted (for example after a
                    # process restart during compilation repair). Treat it as
                    # pending executable work even though there is no rejected
                    # rollout hash yet. Later iterations compare against the
                    # last failed immutable snapshot as before.
                    pending_executable=bool(controller_path.is_file() and not state["iterations"])
                    # A pre-robot task-fidelity rejection is authoritative for
                    # the exact current source. Treating that same source as a
                    # pending executable makes every correction pass compile
                    # and retry it instead of editing the semantic drift. The
                    # cached binding is model-independent evidence and becomes
                    # stale automatically when controller.py changes.
                    fidelity_rejection=self._current_task_fidelity_rejection(controller_path)
                    if fidelity_rejection is not None:pending_executable=False
                    if fidelity_rejection is not None:
                        instruction["task_fidelity_rejection"]=fidelity_rejection
                    if pending_executable:
                        instruction["pending_executable_requires_immediate_run"]={
                            "value":True,
                            "rule":(
                                "The persistent Controller already differs from the last failed "
                                "execution. Do not restart diagnosis or checkout another Skill. "
                                "Compile or make one necessary correction, then call "
                                "run_robot_controller now.")}
                    pass_limits=self._coding_pass_limits(coding_pass)
                    agent=CodingAgent(model=self.model,registry=surface.registry(),system_prompt=SYSTEM_PROMPT,
                                      trace_path=artifact/"agent_trace.jsonl",
                                      executable_pending=pending_executable,
                                      **pass_limits)
                    agent_result=agent.run(json.dumps(instruction,default=str));passes.append(agent_result)
                    if surface.last_execution is not None:break
            if surface.last_execution is None:
                raise RuntimeError(
                    f"coding agent ended {self.max_coding_passes} passes without a robot episode")
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
            transient=evidence.get("transient_infrastructure_failure")
            if transient is not None:
                record["transient_infrastructure_failure"]=transient
            if locked_validation:record["locked_generalization_validation"]=True
            if locked_retry:
                record["locked_validation_retry_after_infrastructure_change"]=True
                record["infrastructure_replay_without_model"]=True
                if automatic_post_action_replay:
                    record["transient_infrastructure_replay_without_model"]=True
                if locked_retry_source_iteration is not None:
                    record["infrastructure_replay_source_iteration"]=(
                        locked_retry_source_iteration)
                binding=(self.guidance.get("deployment_dependency_binding") or {})
                if binding.get("replacements"):
                    record["infrastructure_replay_dependency_binding"]={
                        "replacements":dict(binding["replacements"]),
                        "method":"python_ast_exact_string_constant_rebind"}
            self._save(state)
            if transient is not None:
                consecutive=0
                for candidate in reversed(state["iterations"]):
                    if candidate.get("transient_infrastructure_failure") is None:break
                    consecutive+=1
                if consecutive>=3:
                    raise RuntimeError(
                        "transient infrastructure failure repeated three times; campaign must "
                        "halt without declaring a task frontier")
                continue
            episode_count+=1
            if surface.last_execution and surface.last_execution["sensor_success_candidate"]:
                complete=self._accept_generalization_success(
                    state,record,surface.last_execution)
                if not complete:
                    continue
                return self._freeze_success(
                    state=state,record=record,index=index,task=task,
                    skill_name=skill_name,task_model=task_model,
                    execution=surface.last_execution,evidence=evidence,
                    artifact=artifact)
        return state

__all__=["EvolutionEngine","SYSTEM_PROMPT"]
