"""LLM-authored task semantics and an independent controller coverage review.

The task model is deliberately benchmark neutral.  It turns language into
causal requirements before controller authoring, without consulting an
environment evaluator or simulator state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .agent import CodingAgent
from .registry import FunctionRegistry


TASK_MODEL_PROMPT = '''You are the task-semantics planner inside Embodied Codex.
Before any robot program is written, convert the complete natural-language task
into a causal, environment-neutral execution model.  You do not control a robot
and cannot use benchmark predicates, simulator state, BDDL, rewards, task IDs,
or fixed scene coordinates.

Call submit_task_model with one structured model. Preserve every entity,
qualifier, relation, condition, and desired outcome expressed by the instruction.
Infer causal prerequisites from the instruction and observable world rather than
from benchmark conventions. Represent relevant uncertainty, state-changing
actions, reobservation, verification, and bounded recovery without prescribing a
particular robot algorithm or task family.

Each requirement needs a unique id and every requirement must be satisfied by
at least one ordered phase.  Each phase must declare the actual Robot SDK
operation types it needs: observe, use, act, verify, or record.  Reuse an
existing capability only when its contract satisfies the phase; otherwise list
the capability gap.  Do not write controller code in this stage.'''


TASK_MODEL_CRITIC_PROMPT = '''You are an independent task-model critic inside
Embodied Codex.  Compare the proposed task model against the complete original
instruction and the supplied Tool contracts.  You cannot control the robot and
must not use evaluator or simulator information.

Reject omissions or distortions of an instruction constraint, missing causal
prerequisites, unsupported assumptions, unhandled state changes, unverifiable
success claims, or reliance on an unavailable Tool. Do not demand task-specific
coordinates, benchmark conventions, or a fixed task-family flow. Call
submit_task_model_review exactly once with approved and concrete issues.
Approval requires full semantic and causal coverage. Approval does not require
every capability already to be installed: an explicit,
well-specified capability gap is a valid task-model output and will be handled by
the coding agent. Do not reject a model merely for leaving controller algorithms,
motion parameters, or Tool acquisition unresolved. Reject only if the causal
requirement itself is absent, contradicted, or falsely claimed as already proven.'''


CONTROLLER_REVIEW_PROMPT = '''You are the independent semantic preflight critic
inside Embodied Codex.  No robot episode has started.  Compare the immutable
task model, the complete controller source, and its phase-to-code binding.

Approve only when executable code (not comments or rationale) implements every
phase and preserves its causal order, conditions, observations, actions,
capability calls, recovery, and sensor evidence. A final verifier cannot
compensate for an omitted prerequisite. Reject dead code, fabricated Tool fields,
bindings that point all phases at an unrelated generic function, or a capability
name that is never called. Do not ask for task/state branches, fixed coordinates,
evaluator access, or one particular algorithm. Call submit_controller_review
exactly once.'''


TASK_FIDELITY_REVIEW_PROMPT = '''You are the lightweight task-fidelity critic
inside Embodied Codex. No robot episode has started. Compare the complete natural-
language instruction with the exact controller source.

Reject only a clear semantic substitution or omission that makes the program pursue
a different task: selecting an object by a different stated relation, dropping an
explicit identity/attribute/container qualifier, targeting a different destination,
or using an old task-specific selector that contradicts the current instruction.
Dynamic grounding through robot.instruction and a general VLM is valid evidence of
coverage. A fail-closed program may still be approved when it attempts to ground the
right task and is expected to learn from execution.

This is not a strategy-quality, API, motion-planning, recovery-completeness, or
physical-success review. Do not demand a particular algorithm, fixed phase graph,
coordinates, every possible recovery branch, or proof that the first rollout will
succeed. Comments alone do not repair executable code that plainly implements a
different relation. Call submit_task_fidelity_review exactly once.'''


CAPABILITY_INTEGRATION_REVIEW_PROMPT = '''You are the independent capability-
integration critic inside Embodied Codex. No robot episode has started. Review
the evidence-backed Capability Gap, the exact candidate Tool manifests/manuals/
source, and the exact Controller that proposes to use them.

Approve only when the executable Tool plus its executable Controller integration
materially address the diagnosed missing capability. Check that live inputs,
outputs, control flow, and post-action sensing cover the Gap contract; the
Controller must actually call and consume an approved Tool before the failed
stage. A Tool may implement one well-defined component while the Controller
provides the surrounding feedback loop, but their combined behavior must be a
credible, discriminating integration trial. Reject a renamed parameter sweep,
an unchanged failed mechanism, fixed benchmark coordinates, task/state branches,
fabricated fields, or a static proposal falsely described as closed-loop sensing.

Audit attribution as well as code. Search results are research leads, not proof
that the registered source implements the new code. Reject claims that an
external repository was adopted or adapted when the supplied implementation is
merely an original heuristic inspired by a broad idea. It is valid to label such
code original synthesis and cite sources only as background, but it must not be
presented as integration of that external algorithm. Unit tests establish a JSON
contract, not physical efficacy; do not treat exact self-authored expected output
as task validation.

This is a pre-execution causal-quality review, not an evaluator or a demand for
guaranteed success. Do not require simulator state, a particular algorithm, or
task-specific tuning. Put only defects that make the proposed integration trial
non-credible in blocking_issues. Put real but non-blocking scope limits, expected
failure modes, and missing physical efficacy evidence in limitations. Approval
requires blocking_issues=[]; limitations may remain. Call
submit_capability_integration_review exactly once.'''


class TaskModelError(RuntimeError):
    pass


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload=json.dumps(dict(value),sort_keys=True,separators=(",",":"),ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_task_model(value: Mapping[str, Any], instruction: str) -> dict[str, Any]:
    if not isinstance(value, Mapping): raise TaskModelError("task_model must be an object")
    model=dict(value)
    if model.get("instruction") != instruction:
        raise TaskModelError("task_model must preserve the exact instruction")
    entities=model.get("entities");requirements=model.get("requirements");phases=model.get("phases")
    if not isinstance(entities,list) or not entities:
        raise TaskModelError("task_model requires entities")
    if not isinstance(requirements,list) or not requirements:
        raise TaskModelError("task_model requires requirements")
    if not isinstance(phases,list) or not phases:
        raise TaskModelError("task_model requires phases")
    requirement_ids=[]
    for item in requirements:
        if not isinstance(item,Mapping) or not isinstance(item.get("id"),str) or not item["id"]:
            raise TaskModelError("every requirement needs an id")
        if not isinstance(item.get("description"),str) or not item["description"].strip():
            raise TaskModelError("every requirement needs a description")
        requirement_ids.append(item["id"])
    if len(set(requirement_ids)) != len(requirement_ids):
        raise TaskModelError("requirement ids must be unique")
    phase_ids=[]; covered=set(); allowed={"observe","use","act","verify","record"}
    for phase in phases:
        if not isinstance(phase,Mapping) or not isinstance(phase.get("id"),str) or not phase["id"]:
            raise TaskModelError("every phase needs an id")
        for key in ("purpose","depends_on","satisfies_requirements",
                    "required_robot_operations","observations","actions",
                    "success_evidence","capability_requirements"):
            if key not in phase:
                raise TaskModelError(f"phase {phase['id']} missing {key}")
        if not isinstance(phase["purpose"],str) or not phase["purpose"].strip():
            raise TaskModelError(f"phase {phase['id']} has no purpose")
        for key in ("depends_on","satisfies_requirements","required_robot_operations",
                    "observations","actions","success_evidence","capability_requirements"):
            if not isinstance(phase[key],list):
                raise TaskModelError(f"phase {phase['id']} field {key} must be a list")
        operations=set(phase["required_robot_operations"])
        if not operations.issubset(allowed):
            raise TaskModelError(f"phase {phase['id']} has an unknown Robot operation")
        covered.update(phase["satisfies_requirements"]);phase_ids.append(phase["id"])
    if len(set(phase_ids)) != len(phase_ids): raise TaskModelError("phase ids must be unique")
    phase_set=set(phase_ids)
    for phase in phases:
        if not set(phase["depends_on"]).issubset(phase_set):
            raise TaskModelError(f"phase {phase['id']} depends on an unknown phase")
        if phase["id"] in phase["depends_on"]:
            raise TaskModelError(f"phase {phase['id']} depends on itself")
        if not set(phase["satisfies_requirements"]).issubset(set(requirement_ids)):
            raise TaskModelError(f"phase {phase['id']} references an unknown requirement")
    missing=set(requirement_ids)-covered
    if missing: raise TaskModelError(f"requirements without a phase: {sorted(missing)}")
    # Enforce an acyclic phase dependency graph.
    pending={p["id"]:set(p["depends_on"]) for p in phases};resolved=set()
    while pending:
        ready=[key for key,deps in pending.items() if deps.issubset(resolved)]
        if not ready: raise TaskModelError("phase dependencies contain a cycle")
        for key in ready: resolved.add(key);pending.pop(key)
    model["protocol"]="embodied-codex-task-model-v1"
    model.setdefault("capability_gaps",[])
    if not isinstance(model["capability_gaps"],list):
        raise TaskModelError("capability_gaps must be a list")
    return model


def _free_object_schema():
    return {"type":"object","additionalProperties":True}


def _run_planner(model, instruction, context, trace_path, critique=None):
    candidate={}
    registry=FunctionRegistry()
    def submit(task_model):
        candidate["value"]=validate_task_model(task_model,instruction)
        return {"accepted_for_independent_review":True,
                "task_model_sha256":canonical_sha256(candidate["value"])}
    registry.add("submit_task_model","Submit the complete causal task model.",
                 {"type":"object","properties":{"task_model":_free_object_schema()},
                  "required":["task_model"],"additionalProperties":False},submit)
    request={"instruction":instruction,"available_capabilities":context,
             "required_schema":{
                 "size_rule":"Use concise fields: at most 10 entities, 12 requirements, 12 phases; do not add undeclared phase fields.",
                 "instruction":"exact original string",
                 "entities":"[{id, description, role}]",
                 "requirements":"[{id, kind, description}]",
                 "phases":[{"id":"string","purpose":"string","depends_on":["phase id"],
                    "satisfies_requirements":["requirement id"],
                    "required_robot_operations":["observe|use|act|verify|record"],
                    "observations":["sensor evidence needed"],
                    "actions":["conditional or physical action"],
                    "success_evidence":["sensor evidence"],
                    "capability_requirements":["contract or missing capability"]}],
                 "capability_gaps":"[description]"}}
    if critique:request["critic_issues_to_repair"]=critique
    CodingAgent(model=model,registry=registry,system_prompt=TASK_MODEL_PROMPT,
                trace_path=trace_path,max_turns=12).run(json.dumps(request,default=str))
    if "value" not in candidate: raise TaskModelError("planner did not submit a valid task model")
    return candidate["value"]


def _run_task_critic(model, instruction, context, task_model, trace_path):
    review={};registry=FunctionRegistry()
    def submit(approved,issues):
        if not isinstance(approved,bool) or not isinstance(issues,list):
            raise TaskModelError("invalid task-model review")
        review.update({"approved":approved,"issues":[str(x) for x in issues]})
        return {"recorded":True}
    registry.add("submit_task_model_review","Record the independent semantic verdict.",
        {"type":"object","properties":{"approved":{"type":"boolean"},
          "issues":{"type":"array","items":{"type":"string"}}},
         "required":["approved","issues"],"additionalProperties":False},submit)
    payload={"instruction":instruction,"available_capabilities":context,
             "proposed_task_model":task_model}
    CodingAgent(model=model,registry=registry,system_prompt=TASK_MODEL_CRITIC_PROMPT,
                trace_path=trace_path,max_turns=8).run(json.dumps(payload,default=str))
    if "approved" not in review:raise TaskModelError("critic did not submit a verdict")
    if review["approved"] and review["issues"]:
        raise TaskModelError("critic approval cannot contain issues")
    return review


def build_task_model(*, model, instruction: str, context: Any, artifact_dir: str|Path,
                     max_attempts: int=3):
    root=Path(artifact_dir);root.mkdir(parents=True,exist_ok=True);issues=None
    failures=[];semantic_attempts=0
    indices=[]
    for path in root.glob("candidate_[0-9][0-9].json"):
        try:index=int(path.stem.rsplit("_",1)[1])
        except ValueError:continue
        indices.append(index);review_path=root/f"review_{index:02d}.json"
        if not review_path.is_file():continue
        candidate=json.loads(path.read_text());review=json.loads(review_path.read_text())
        validate_task_model(candidate,instruction);semantic_attempts+=1
        if review.get("approved") is True:
            candidate.pop("task_model_sha256",None)
            candidate["task_model_sha256"]=canonical_sha256(candidate)
            return candidate
        issues=[str(item)[:700] for item in (review.get("issues") or [])[:6]]
    attempt=max(indices,default=0)+1
    # Transport/protocol failures do not consume the semantic repair budget.
    # Bound them separately so an unavailable model service still terminates.
    protocol_attempts=0;protocol_limit=max(3,max_attempts*3)
    while semantic_attempts<max_attempts and protocol_attempts<protocol_limit:
        protocol_attempts+=1
        try:
            candidate=_run_planner(model,instruction,context,
                root/f"planner_attempt_{attempt:02d}.jsonl",issues)
        except TaskModelError as exc:
            failures.append(f"planner attempt {attempt}: {exc}")
            issues=["The prior planner did not submit a valid structured model. "
                    "Call submit_task_model exactly once with a concise complete value."]
            attempt+=1
            continue
        (root/f"candidate_{attempt:02d}.json").write_text(
            json.dumps(candidate,indent=2,default=str)+"\n")
        try:
            review=_run_task_critic(model,instruction,context,candidate,
                root/f"critic_attempt_{attempt:02d}.jsonl")
        except TaskModelError as exc:
            failures.append(f"critic attempt {attempt}: {exc}")
            issues=["The independent critic failed to submit a verdict; resubmit a concise model."]
            attempt+=1
            continue
        (root/f"review_{attempt:02d}.json").write_text(json.dumps(review,indent=2)+"\n")
        if review["approved"]:
            candidate["task_model_sha256"]=canonical_sha256(candidate)
            return candidate
        semantic_attempts+=1
        # Preserve the critic artifact in full, but bound the repair context so
        # one verbose critique cannot cause the next model call to time out.
        issues=[str(item)[:700] for item in review["issues"][:6]]
        attempt+=1
    raise TaskModelError(f"task model rejected after {max_attempts} attempts: "
                         f"{issues}; protocol_failures={failures}")


def review_controller_binding(*, model, task_model: Mapping[str,Any], source: str,
                              binding: Mapping[str,Any], trace_path: str|Path):
    verdict={};registry=FunctionRegistry()
    phase_ids=[p["id"] for p in task_model["phases"]]
    def submit(approved,covered_phase_ids,issues):
        if not isinstance(approved,bool) or not isinstance(covered_phase_ids,list) or not isinstance(issues,list):
            raise TaskModelError("invalid controller review")
        verdict.update({"approved":approved,
            "covered_phase_ids":[str(x) for x in covered_phase_ids],
            "issues":[str(x) for x in issues]})
        return {"recorded":True}
    registry.add("submit_controller_review","Record semantic controller coverage.",
        {"type":"object","properties":{"approved":{"type":"boolean"},
          "covered_phase_ids":{"type":"array","items":{"type":"string"}},
          "issues":{"type":"array","items":{"type":"string"}}},
         "required":["approved","covered_phase_ids","issues"],
         "additionalProperties":False},submit)
    payload={"task_model":task_model,"phase_binding":binding,"controller_source":source}
    CodingAgent(model=model,registry=registry,system_prompt=CONTROLLER_REVIEW_PROMPT,
                trace_path=trace_path,max_turns=10).run(json.dumps(payload,default=str))
    if "approved" not in verdict:raise TaskModelError("controller critic did not submit a verdict")
    if verdict["approved"] and (set(verdict["covered_phase_ids"])!=set(phase_ids) or verdict["issues"]):
        raise TaskModelError("controller critic issued an internally inconsistent approval")
    return verdict


def review_controller_task_fidelity(*, model, instruction: str, source: str,
                                    trace_path: str|Path):
    verdict={};registry=FunctionRegistry()
    def submit(approved,issues):
        if not isinstance(approved,bool) or not isinstance(issues,list):
            raise TaskModelError("invalid task-fidelity review")
        verdict.update({"approved":approved,"issues":[str(x) for x in issues]})
        return {"recorded":True}
    registry.add("submit_task_fidelity_review","Record task semantic fidelity.",
        {"type":"object","properties":{"approved":{"type":"boolean"},
          "issues":{"type":"array","items":{"type":"string"}}},
         "required":["approved","issues"],"additionalProperties":False},submit)
    payload={"instruction":str(instruction),"controller_source":str(source)}
    CodingAgent(model=model,registry=registry,system_prompt=TASK_FIDELITY_REVIEW_PROMPT,
                trace_path=trace_path,max_turns=4).run(json.dumps(payload,default=str))
    if "approved" not in verdict:
        raise TaskModelError("task-fidelity critic did not submit a verdict")
    if verdict["approved"] and verdict["issues"]:
        raise TaskModelError("task-fidelity approval cannot contain issues")
    return verdict


def review_capability_integration(*, model, gap: Mapping[str,Any],
                                  tools: list[Mapping[str,Any]],
                                  controller_source: str, trace_path: str|Path):
    """Independently check that an acquired Tool really addresses its Gap."""
    verdict={};registry=FunctionRegistry()
    candidate_ids=[str((item.get("manifest") or {}).get("tool_id") or "")
                   for item in tools]
    def submit(approved,approved_tool_ids,covered_requirements,blocking_issues,
               limitations):
        if (not isinstance(approved,bool) or not isinstance(approved_tool_ids,list)
                or not isinstance(covered_requirements,list)
                or not isinstance(blocking_issues,list)
                or not isinstance(limitations,list)):
            raise TaskModelError("invalid capability-integration review")
        verdict.update({"approved":approved,
            "approved_tool_ids":[str(x) for x in approved_tool_ids],
            "covered_requirements":[str(x) for x in covered_requirements],
            "issues":[str(x) for x in blocking_issues],
            "limitations":[str(x) for x in limitations]})
        return {"recorded":True}
    registry.add("submit_capability_integration_review",
        "Record whether the concrete Tool/Controller integration satisfies the diagnosed Gap.",
        {"type":"object","properties":{"approved":{"type":"boolean"},
          "approved_tool_ids":{"type":"array","items":{"type":"string"}},
          "covered_requirements":{"type":"array","items":{"type":"string"}},
          "blocking_issues":{"type":"array","items":{"type":"string"}},
          "limitations":{"type":"array","items":{"type":"string"}}},
         "required":["approved","approved_tool_ids","covered_requirements",
                     "blocking_issues","limitations"],
         "additionalProperties":False},submit)
    payload={"capability_gap":dict(gap),"candidate_tools":[dict(item) for item in tools],
             "controller_source":str(controller_source)}
    CodingAgent(model=model,registry=registry,
                system_prompt=CAPABILITY_INTEGRATION_REVIEW_PROMPT,
                trace_path=trace_path,max_turns=6).run(json.dumps(payload,default=str))
    if "approved" not in verdict:
        raise TaskModelError("capability-integration critic did not submit a verdict")
    unknown=set(verdict["approved_tool_ids"])-set(candidate_ids)
    if unknown:
        raise TaskModelError("capability critic approved an unknown Tool")
    if verdict["approved"] and (not verdict["approved_tool_ids"]
            or not verdict["covered_requirements"] or verdict["issues"]):
        raise TaskModelError("capability critic issued an internally inconsistent approval")
    return verdict


__all__=["TaskModelError","build_task_model","canonical_sha256",
         "review_capability_integration","review_controller_binding","review_controller_task_fidelity",
         "validate_task_model"]
