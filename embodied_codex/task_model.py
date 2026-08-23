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

Call submit_task_model with one structured model.  Preserve every identifying
relation and destination relation in the instruction.  Include latent
accessibility conditions: an object described as in/behind/under an articulated
container may be visible yet physically unreachable, so the plan must inspect
accessibility and conditionally manipulate and verify the container before
attempting the object.  This is causal reasoning, not a benchmark-specific
rule.  Likewise represent occlusion, obstacles, containment, required tool use,
reobservation after scene-changing actions, attachment verification, placement
verification, and bounded recovery when they are relevant.

Each requirement needs a unique id and every requirement must be satisfied by
at least one ordered phase.  Each phase must declare the actual Robot SDK
operation types it needs: observe, use, act, verify, or record.  Reuse an
existing capability only when its contract satisfies the phase; otherwise list
the capability gap.  Do not write controller code in this stage.'''


TASK_MODEL_CRITIC_PROMPT = '''You are an independent task-model critic inside
Embodied Codex.  Compare the proposed task model against the complete original
instruction and the supplied Tool contracts.  You cannot control the robot and
must not use evaluator or simulator information.

Reject omissions of source identity, spatial relations, accessibility or
articulation preconditions, scene-changing-action verification, mandatory
reobservation, grasp/attachment evidence, destination relation, or final stable
verification.  Reject a plan that treats a detector label as proof of a
relation, assumes a contained object is reachable, or claims an unavailable
Tool can perform a phase.  Do not demand task-specific coordinates or a fixed
benchmark flow.  Call submit_task_model_review exactly once with approved and
concrete issues.  Approval requires full semantic and causal coverage.'''


CONTROLLER_REVIEW_PROMPT = '''You are the independent semantic preflight critic
inside Embodied Codex.  No robot episode has started.  Compare the immutable
task model, the complete controller source, and its phase-to-code binding.

Approve only when executable code (not comments or rationale) implements every
phase and preserves its causal order, conditions, reobservations, actions,
capability calls, recovery, and sensor evidence.  In particular, a program may
not proceed directly to manipulating an object whose accessibility phase is
unimplemented.  A final verifier cannot compensate for an omitted prerequisite.
Reject dead code, fabricated Tool fields, bindings that point all phases at an
unrelated generic function, or a capability name that is never called.  Do not
ask for task/state branches, fixed coordinates, evaluator access, or one
particular algorithm.  Call submit_controller_review exactly once.'''


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
    for attempt in range(1,max_attempts+1):
        candidate=_run_planner(model,instruction,context,
            root/f"planner_attempt_{attempt:02d}.jsonl",issues)
        (root/f"candidate_{attempt:02d}.json").write_text(
            json.dumps(candidate,indent=2,default=str)+"\n")
        review=_run_task_critic(model,instruction,context,candidate,
            root/f"critic_attempt_{attempt:02d}.jsonl")
        (root/f"review_{attempt:02d}.json").write_text(json.dumps(review,indent=2)+"\n")
        if review["approved"]:
            candidate["task_model_sha256"]=canonical_sha256(candidate)
            return candidate
        issues=review["issues"]
    raise TaskModelError(f"task model rejected after {max_attempts} attempts: {issues}")


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


__all__=["TaskModelError","build_task_model","canonical_sha256",
         "review_controller_binding","validate_task_model"]
