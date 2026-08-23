"""Run a resumable GPT-driven embodied controller evolution loop.

This is the public Harness orchestration entry point for a development
environment.  The environment adapter remains LIBERO-specific here, while
the evolution state machine is benchmark-neutral and stores sensor evidence
only.  A sealed evaluator is never opened by this process.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT), str(ROOT / "capability_library"),
    str(ROOT / "capability_library" / "tools"), str(ROOT / "Thea"),
]

from autonomous_evolution_loop import AutonomousEvolutionLoop, EvolutionConfig, sensor_only
from controller_program_workspace import (
    ControllerProgramWorkspace,
    verified_stage_from_evidence,
)
from stage_node_workspace import StageNodeWorkspace
from controller_graph_workspace import ControllerGraphWorkspace
from task_skill_workspace import TaskSkillWorkspace
from graph_task_skill_workspace import GraphTaskSkillWorkspace
from frontier_registrar import make_frontier_registrar
from libero_robot_sdk import (
    execute_libero_graph,
    execute_libero_program,
    libero_task_instruction,
    robot_sdk_contract,
)
from harness import Harness
from harness.models.providers import OpenAICompatLLM


SYSTEM_PROMPT = """You are the autonomous embodied Coding Agent.
Use only task instruction, RGB/RGB-D, calibration, proprioception, and action
history. Never use reward, success, done, evaluator output, BDDL, simulator
poses, object IDs, or task/state branches. You may use public task-disjoint
models and algorithms. Each controller must be generic and immutable.

You author a complete independent Python controller, not a stage specification.
Its source must define run(robot), own all loops, branches, reobservation,
verification, retries, transport, placement, and recovery, and use only
robot.instruction(), robot.observe(), robot.call_tool(name, arguments),
robot.act(action), and robot.record(event). The deployment-owned Robot SDK
executes these calls. Do not use files, print, private attributes, or imports
other than math/statistics. Never merely describe a control-flow change in a
rationale: encode it in source. Use the exact response fields in the supplied
SDK contract; do not guess alternate field names or replace a missing visual
region with a fabricated box or coordinate.
Keep each complete controller compact: prefer shared helpers and data-driven
bounded loops over repeated stage code, normally staying under 350 source lines.
Compactness must not remove required verification or recovery branches.

Preserve the full task semantics when calling select_entities. Pass the live
instruction as a complete query (or equally explicit source and destination
phrases); never replace it with a truncated token list, unordered noun bag, or
the first few words. Token-level GroundingDINO queries may be used only after
select_entities has established semantic source/target roles. A controller
that changes a previously verified full-instruction selector into word-level
queries is a regression, not a new perception strategy.
Obtain that live text inside every controller with robot.instruction(); never
copy the current task sentence into a Python string literal. Literal task text
is not a reusable controller and must fail the authoring audit.

For the first round, load the reusable closed-loop grasp/place Skills, call
list_tested_capability_tools, create_controller_program, and then execute the
program exactly once. For every later round, inspect the prior immutable
program and supplied
sensor-only failure evidence. If it indicates failure, search public resources
and register useful leads with register_public_research_lead. Provenance-check
candidate assets before using them. Then
create a new complete controller version with a generic evidence-supported change and
execute it. Do not stop after a failed round unless no legal next experiment
exists. Never claim success from evaluator output; use only the sensor-only
conclusion returned by the execution tool.

When a failure can be addressed by a small deterministic algorithm, use
describe_capability_hook_contracts before writing code, then use
create_capability_tool and test_capability_hook to implement and validate it.
You may additionally use test_capability_tool for your own semantic unit tests.
A unit-tested Tool is registered in the shared capability library and can be
called by immutable tool_id from a later program through robot.call_tool.
Treat each tested Tool as a typed interface, not a name-only hint.
The four predefined hooks are conveniences, not a capability ceiling. When no
predefined hook matches, create a generic Tool with explicit object
input_schema and output_schema, test it with test_capability_tool, and call its
immutable tool_id from the controller. Generic Tools may implement perception
post-processing, waypoint/path planning, articulation recovery, IK helpers, or
other deterministic algorithms from legal sensor and proprioceptive inputs.
Use the evaluator-isolated engineering workspace for multi-file implementation,
tests, public repository checkout, and inspection of your own sensor-only run
artifacts. A research lead without an implementation/test attempt is not an
acquired capability.
list_tested_capability_tools returns the exact deduplicated hook contracts.
Before calling an immutable tool_id, build every required input field and
consume the named output fields exactly. If legal runtime observations cannot
satisfy the contract, do not claim that Tool was applied.
The program audit requires an explicit keyed lookup for every Tool output.
Each lookup must alter the corresponding control behavior: in particular,
close_steps controls the close duration and reobserve_before_attempt guards a
fresh visual observation/relocalization before the attempt. Merely assigning,
copying, logging, or returning a Tool field is not implementation.
Never fabricate a required contract field. In particular, generate_grasps
returns each candidate's sensor/model-derived approach_world; forward that
exact vector to grasp ranking or execution Tools instead of substituting a
constant top-down vector. Also use approach_world to construct the candidate's
pregrasp waypoint; every program calling generate_grasps is audited for an
explicit approach_world lookup.
Initial grasp candidate failures may produce a ranking Tool; if ranking was
actually applied and attachment still failed, change execution/contact logic
instead of creating another equivalent ranker. Never copy executable code from
an unverified search result.

Treat Cartesian arrival as part of the closed loop. A fixed repeat count does
not imply that a target was reached. Inspect every robot.act return and continue
bounded servo calls until reached_target is true before descending, closing,
transporting, or releasing. If action_outcomes show large final errors, repair
motion convergence before changing grasp ranking or contact geometry.
Use repeat up to 20 inside each bounded servo call for distant free-space
targets; one OSC step normally advances only a few millimeters. Do not treat
millimeter-scale progress as stalled. Set position_tolerance_m at or below
0.01 for final grasp/contact commands. robot.act already returns current
proprioception, so do not acquire and save a new RGB-D frame on every servo
step; call robot.observe only when fresh visual evidence is needed.
Give every robot.act call a concise phase string such as approach, contact,
close, lift, drawer_pull, transport, release, or retreat. The SDK attaches this
label to motion diagnostics so the next authoring round can identify which
stage stalled. Before every early return, robot.record a structured phase and
reason. Do not leave failed actions or abort paths unlabeled.

Route capabilities from the complete live instruction before writing motion
code. If its source clause names a drawer or cabinet, load the
visual-articulated-drawer-open-and-retrieve Skill in addition to the generic
closed-loop grasp/place Skills. Detect a visible handle using RGB-D and public
open-vocabulary perception; derive the pull direction only from observed
handle, contained-object, or cabinet geometry. Approach, close, and pull with
bounded sensor-derived motions. Before pulling, call capture_landmark_baseline
on the detected handle. After pulling and acquiring a fresh frame, call
verify_landmark_displacement with that opaque baseline and visually require at
least 4 cm of handle displacement. Fully reobserve and re-ground the source after articulation. If
the drawer is only partially open and contact remains unreachable, first test
a farther visually verified pull before changing grasp ranking. Never infer
drawer state, collision geometry, or motion from BDDL or simulator internals.

Attachment verification has a strict physical order: approach with the
gripper open, reach the contact pose, close, lift the EEF by enough to visibly
separate the object from its source support (normally at least 0.08 m), wait
for that lift target to be reached, then acquire a fresh frame and call
verify_attachment. Never call verify_attachment immediately after closing at
the contact pose: source-vacated is intentionally required and a pre-lift
check is not evidence of attachment. Keep the deployment-owned initial source
baseline immutable across candidate retries. A failed candidate may move the
object, so relocalize the live object for the next contact pose, but do not
replace or fabricate the verifier's original source baseline.

Preserve verified stage prefixes. When the latest controller regresses below
the supplied best_prior checkpoint, inspect and branch from the best program;
copy its proven grasp/transport prefix and change only the first unverified
suffix. Never discard a strict attachment success merely to redesign placement.
Calling inspect_controller_program on that best_prior is mandatory before
claiming that its prefix was preserved. If its evidence shows an earlier
candidate failed and a later candidate attached, preserve the bounded
multi-candidate loop and its ordering; trying only candidate zero is a known
regression. Do not paraphrase a proven prefix from memory.

For placement, never rely only on the initial target xyz. Keep the target
bbox from select_entities and, after transport and before every release or
correction, call segment_box on a fresh RGB-D frame for that destination. Use
the returned support-region xyz as the current target center, then verify the
object-to-support relation from subsequent fresh frames.
Before grasping, also call segment_box once on the unobstructed destination
and retain its returned opaque mask_id. Pass that target_mask_id to every
post-release verify_support_relation call. Placement success must use the
object SAM footprint's containment and clearance inside this pre-placement
support mask, plus height and temporal stability; do not reduce an "on"
relation to a fixed 3-D center-distance threshold.
Require two consecutive fresh post-release verify_support_relation calls to
pass, with the object world position stable between them, before returning
success. Do not return after the first verified frame; one frame cannot prove
post-release temporal stability.
"""

GRAPH_SYSTEM_PROMPT = """You are an autonomous embodied Coding Agent operating
a structured Controller Graph, never a monolithic controller script. Use only
language, RGB/RGB-D, calibration, proprioception, action history, and public
task-disjoint capabilities. Never use reward, done, evaluator output, BDDL,
simulator poses/IDs, task/state branches, or fixed absolute geometry.

Create immutable Stage Nodes with create_stage_node. Each node defines
run_stage(robot, context), declares required context fields, declared outputs
for each outcome, and owns exactly one coherent stage such as observation,
grounding, articulation, grasp planning, grasp execution, verification,
transport, or placement. Compose node IDs with create_controller_graph. Every
declared outcome must have an explicit edge and all context requirements must
be guaranteed by predecessor outputs. Execute exactly one graph per autonomous
round. Terminal edge targets are the exact literals $success and $failure;
never use bare success/failure names and never create terminal Stage Nodes.
The Robot Adapter supplies exactly one initial context field: task_instruction.
Do not declare observations, detections, poses, boxes, masks, or plans as
initial_context_fields; an entry Stage Node must produce them from live calls.

Every return from run_stage must be exactly an object shaped
{"outcome": "declared_outcome", "updates": {declared fields...}}. Never return
a Python tuple such as ``outcome, updates``. At every return statement, outcome
must be a literal string and updates must be an object literal whose keys
exactly equal provides_by_outcome for that outcome. Use explicit if branches
with separate literal returns; do not compute outcome using a variable or a
conditional expression. Implement every declared outcome. Keep action and verification
separate: a verification Stage Node may observe and call one adapter-owned
verify_* Tool type (including repeated fresh calls required for temporal
stability) but cannot call robot.act; motion nodes cannot call verify_*.
Declare the verified outcome of each such node as a checkpoint_outcome. For a
drawer retrieval task this means distinct articulation verification,
post-lift attachment verification, and final stable support verification
nodes rather than burying verification inside motion nodes.
After create_controller_graph, call preflight_controller_graph. Execute only an
eligible preflight result; repair compile errors before consuming the single
Robot Adapter execution allowed in the round.

On later rounds inspect the best prior graph and relevant nodes. The Harness
supplies required frozen aliases from adapter-owned visual checkpoints. Keep
those immutable node IDs and incident edges; replace only the first failed node
or its unfrozen successors. Do not copy, compare, or rewrite successful source
prefixes. A checkpoint may be declared only by a verification Stage Node that
calls an adapter-owned verify_* Tool. Use public search and register or test a
new generic Tool when failure evidence identifies a missing capability.

All motion geometry must be computed from live context, sensor observations,
or model/Tool outputs. Each act result must gate stage outcomes. Reobserve at
physical phase boundaries and route failed verification to bounded recovery or
$failure. Never claim graph success without the required attachment and support
verification nodes.
"""

ACQUISITION_SYSTEM_PROMPT = """You are the capability-acquisition phase of an
autonomous embodied Coding Agent. You cannot create or execute robot
controllers in this phase. Use only the supplied sensor-only failure history.
Search public task-disjoint resources where useful, record provenance, and
turn a small deterministic remedy into an audited reusable Tool. Before coding
a runtime Tool, call describe_capability_hook_contracts; after coding, call
test_capability_hook. A failed contract test must be diagnosed and repaired
within this session when possible. Never use evaluator output, reward, BDDL,
simulator state, task-specific training data, or benchmark answers.
If none of the predefined hooks fits the diagnosed mechanism, do not stop at
"missing hook". Create a generic Tool with explicit object input_schema and
output_schema, validate representative behavior with test_capability_tool, and
preserve it for the next authoring round.
Use write_engineering_file and run_engineering_command when implementation or
dependency testing needs a real workspace. A paper/URL registration alone is
not completion. Before stopping without a usable capability, preserve a
concrete failed engineering, installation, contract, unit-test, or smoke-test
result.
Treat prior capability outcomes as experimental evidence. Do not create a
semantic duplicate of a Tool whose hook was applied while the same stage
failure persisted. Change the capability family or state a genuinely distinct,
sensor-supported hypothesis.
"""


def _tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": e.get("type"), "name": e.get("name"), "success": e.get("success")}
        for e in events
        if e.get("type") in {"tool_call", "tool_result"}
    ]


def _run_visible_harness_stream(
    harness: Harness,
    instruction: str,
    *,
    live_trace: Path,
    max_turns: int,
    failure_budget: int,
    system_prompt_override: str,
) -> list[dict[str, Any]]:
    """Persist Agent events immediately and expose concise progress milestones."""
    events: list[dict[str, Any]] = []
    live_trace.parent.mkdir(parents=True, exist_ok=True)
    with live_trace.open("a") as stream_file:
        for raw_event in harness.run_stream(
            instruction,
            max_turns=max_turns,
            failure_budget=failure_budget,
            system_prompt_override=system_prompt_override,
        ):
            event = dict(raw_event)
            events.append(event)
            stream_file.write(json.dumps(event, default=str) + "\n")
            stream_file.flush()
            event_type = str(event.get("type") or "")
            if event_type in {"tool_call", "tool_result", "model_error", "done"}:
                name = str(event.get("name") or "")
                success = event.get("success")
                detail = (
                    f"success={str(bool(success)).lower()}"
                    if success is not None else ""
                )
                print(
                    f"[embodied-codex] {event_type} {name} {detail}".rstrip(),
                    flush=True,
                )
    return events


def _extract_execution(events: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    """Return the last real execution, ignoring rejected budget retries."""
    controller_id = None
    evidence: dict[str, Any] = {}
    for event in events:
        if event.get("type") != "tool_result" or event.get("name") not in {
            "execute_controller_script", "execute_controller_program",
            "execute_controller_graph",
        }:
            continue
        result = event.get("result") or {}
        if result.get("sensor_evidence"):
            controller_id = (
                result.get("graph_id") or result.get("controller_id") or controller_id
            )
            evidence = sensor_only(result["sensor_evidence"])
    return controller_id, evidence


def compact_sensor_evidence_for_prompt(
    evidence: Mapping[str, Any], *, max_action_samples: int = 12,
) -> dict[str, Any]:
    """Keep full evidence on disk while bounding the reasoning-model context."""
    compact = dict(sensor_only(evidence))
    compact.pop("action_outcome_samples", None)
    compact.pop("action_outcomes_total", None)
    compact.pop("action_outcomes_omitted", None)
    outcomes = [
        item for item in compact.pop("action_outcomes", [])
        if isinstance(item, Mapping)
    ]
    if outcomes:
        edge = max(1, min(3, max_action_samples // 3))
        ranked = sorted(
            range(len(outcomes)),
            key=lambda index: float(outcomes[index].get("final_error_m") or -1.0),
            reverse=True,
        )
        selected = set(range(min(edge, len(outcomes))))
        selected.update(range(max(0, len(outcomes) - edge), len(outcomes)))
        for index in ranked:
            if len(selected) >= min(max_action_samples, len(outcomes)):
                break
            selected.add(index)
        samples = [dict(outcomes[index]) for index in sorted(selected)]
        compact["action_outcome_samples"] = samples[:max_action_samples]
        compact["action_outcomes_total"] = len(outcomes)
        compact["action_outcomes_omitted"] = max(
            0, len(outcomes) - len(compact["action_outcome_samples"])
        )
    methods = compact.pop("rpc_methods", None)
    if isinstance(methods, list):
        compact["rpc_method_counts"] = dict(Counter(str(item) for item in methods))
    return compact


def _model() -> OpenAICompatLLM:
    return OpenAICompatLLM(
        api_key=os.environ["APEX_API_KEY"],
        base_url="https://api.apexin.ai/v1",
        model="gpt-5.6-sol",
        temperature=0,
        # Complete embodied programs are substantially larger than skill-routing
        # tool calls.  The OpenAI-compatible client defaults to 60 seconds,
        # which consistently terminated the first full-program generation.
        timeout=300.0,
        max_tokens=5000,
        # The compatible gateway closes long generations near 60 seconds.
        # Low effort is sufficient because Skills and the exact SDK contract
        # carry the domain procedure, and keeps the complete tool call bounded.
        reasoning_effort="low",
        max_retries=5,
        provider_label="apex-openai-compat",
    )


def _suggest_capability_hook(failure: Mapping[str, Any]) -> str | None:
    failure_class = str(failure.get("failure_class") or "")
    ineffective = {
        str(item.get("hook"))
        for item in failure.get("capability_outcomes") or ()
        if isinstance(item, Mapping) and item.get("outcome") == "failure_persisted"
    }
    if failure_class == "attachment_not_verified":
        if "grasp_execution_profile" in ineffective:
            return None
        return (
            "grasp_execution_profile"
            if "grasp_retry_ranking" in ineffective
            else "grasp_retry_ranking"
        )
    if failure_class in {"contact_convergence_failed", "contact_unreachable"}:
        return (
            None
            if "grasp_execution_profile" in ineffective
            else "grasp_execution_profile"
        )
    if failure_class == "transport_not_verified":
        return None if "transport_profile" in ineffective else "transport_profile"
    if failure_class == "placement_not_verified":
        return (
            None
            if "support_relation_profile" in ineffective
            else "support_relation_profile"
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--state", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-turns-per-round", type=int, default=16)
    parser.add_argument("--max-turns-acquisition", type=int, default=14)
    parser.add_argument("--acquisition-after-same-failure", type=int, default=2)
    parser.add_argument(
        "--force-acquisition-next-round", action="store_true",
        help=(
            "Re-run acquisition once after a Harness capability upgrade without "
            "deleting or rewriting prior evidence."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-workspace", type=Path, required=True)
    parser.add_argument(
        "--controller-interface", choices=("graph", "program", "spec"), default="graph",
        help="graph is the canonical typed-node interface; program/spec are legacy.",
    )
    parser.add_argument(
        "--stage-node-workspace", type=Path, default=None,
        help="Immutable Stage Node store; defaults to OUTPUT/stage_nodes.",
    )
    parser.add_argument(
        "--capability-workspace",
        type=Path,
        default=None,
        help="Persistent audited Tool store; defaults to OUTPUT/capability_tools.",
    )
    parser.add_argument(
        "--task-skill-workspace",
        type=Path,
        default=ROOT / "capability_library" / "task_skills",
        help="Shared immutable learned Task Skill store.",
    )
    args = parser.parse_args()
    if not os.environ.get("APEX_API_KEY"):
        raise OSError("APEX_API_KEY is required")
    args.output.mkdir(parents=True, exist_ok=True)
    task_instruction = libero_task_instruction("libero_spatial", args.task)
    workspace = args.controller_workspace
    stage_node_workspace = args.stage_node_workspace or (args.output / "stage_nodes")
    capability_workspace = args.capability_workspace or (args.output / "capability_tools")
    state_path = args.output / "evolution_state.json"
    ledger_path = args.output / "capability_acquisition.jsonl"
    (args.output / "run_config.json").write_text(json.dumps({
        "protocol": "embodied-codex-autonomous-evolution-v2",
        "model": "gpt-5.6-sol",
        "base_url": "https://api.apexin.ai/v1",
        "task_adapter": "libero_spatial",
        "task_selector": args.task,
        "task_instruction": task_instruction,
        "state_selector": args.state,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "acquisition_after_same_failure": args.acquisition_after_same_failure,
        "force_acquisition_next_round": args.force_acquisition_next_round,
        "controller_workspace": str(workspace.resolve()),
        "stage_node_workspace": str(stage_node_workspace.resolve()),
        "controller_interface": args.controller_interface,
        "capability_workspace": str(capability_workspace.resolve()),
        "task_skill_workspace": str(args.task_skill_workspace.resolve()),
        "evaluator_available_to_process": False,
        "api_key_persisted": False,
    }, indent=2) + "\n")

    def acquire_capabilities(round_id: int, failure: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run a controller-free acquisition session before the next authoring session."""
        acquisition_dir = args.output / f"acquisition_before_round_{round_id:03d}"
        if (acquisition_dir / "report.json").exists():
            attempt = 2
            while (
                args.output
                / f"acquisition_before_round_{round_id:03d}_attempt_{attempt:03d}"
            ).exists():
                attempt += 1
            acquisition_dir = (
                args.output
                / f"acquisition_before_round_{round_id:03d}_attempt_{attempt:03d}"
            )
        acquisition_dir.mkdir(parents=True, exist_ok=True)
        registrar = make_frontier_registrar(
            [f"libero_spatial:task_{args.task}"],
            ledger_path=str(ledger_path),
            state_path=str(args.output / "self_evolution_state.json"),
            capability_workspace=capability_workspace,
            include_controller_tools=False,
            engineering_workspace=args.output / "engineering_workspace",
            engineering_read_roots={
                "runs": args.output,
                "controllers": workspace,
                "capabilities": capability_workspace,
            },
        )
        harness = Harness(
            {"servers": [], "llm": {"provider": "mock", "model": "gpt-5.6-sol"},
             "skills": {"dir": str(ROOT / "capability_library" / "skills")},
             "context": {"compaction": {"enabled": False}}},
            model=_model(), builtin_registrar=registrar,
        )
        failure_class = str(failure.get("failure_class") or "unknown_failure")
        suggested_hook = _suggest_capability_hook(failure)
        failure_for_prompt = dict(sensor_only(failure))
        if isinstance(failure_for_prompt.get("latest_sensor_evidence"), Mapping):
            failure_for_prompt["latest_sensor_evidence"] = compact_sensor_evidence_for_prompt(
                failure_for_prompt["latest_sensor_evidence"]
            )
        instruction = (
            "A Harness phase gate was triggered by repeated sensor-only failure. "
            f"The live public task instruction is: {task_instruction!r}. "
            f"Failure evidence: {json.dumps(failure_for_prompt, sort_keys=True)}\n"
            "Load the autonomous-capability-acquisition Skill. Search public resources "
            "when external knowledge is relevant and register useful leads. "
        )
        if suggested_hook:
            instruction += (
                f"The closest existing runtime contract is {suggested_hook}. Call "
                "describe_capability_hook_contracts, implement a generic remedy if the "
                "evidence supports one, and call test_capability_hook. "
            )
        instruction += (
            "Finish by recording what was acquired, rejected, or remains missing. "
            "Do not create a semantic duplicate of a previously acquired Tool, and "
            "do not retry a hook marked failure_persisted unless the new hypothesis "
            "changes the mechanism rather than only weights or naming. "
            "Do not create or execute a controller."
        )
        try:
            live_trace = acquisition_dir / "thea_trace_live.jsonl"
            events = _run_visible_harness_stream(
                harness, instruction,
                live_trace=live_trace,
                max_turns=args.max_turns_acquisition,
                failure_budget=5,
                system_prompt_override=ACQUISITION_SYSTEM_PROMPT,
            )
            engineering_names = {
                "write_engineering_file", "run_engineering_command",
                "create_capability_tool", "test_capability_tool",
                "test_capability_hook",
            }
            attempted = any(
                event.get("type") == "tool_call"
                and event.get("name") in engineering_names
                for event in events
            )
            if not attempted:
                followup = (
                    "The prior acquisition stopped after research without an "
                    "implementation or executable test, so the phase gate is not "
                    "satisfied. Do not repeat the search-only summary. Use the "
                    "evaluator-isolated engineering workspace now. Implement and "
                    "test a deterministic generic schema Tool for the diagnosed "
                    "failure, or attempt installation/smoke testing of the public "
                    "asset and preserve the concrete failure. Do not create or "
                    "execute a robot controller."
                )
                events.extend(_run_visible_harness_stream(
                    harness, followup,
                    live_trace=live_trace,
                    max_turns=max(10, args.max_turns_acquisition // 2),
                    failure_budget=4,
                    system_prompt_override=ACQUISITION_SYSTEM_PROMPT,
                ))
        finally:
            harness.close()
        trace_path = acquisition_dir / "thea_trace.json"
        trace_path.write_text(json.dumps(events, indent=2, default=str) + "\n")
        hook_tests = [
            sensor_only(event.get("result") or {})
            for event in events
            if event.get("type") == "tool_result"
            and event.get("name") == "test_capability_hook"
        ]
        generic_tests = [
            sensor_only(event.get("result") or {})
            for event in events
            if event.get("type") == "tool_result"
            and event.get("name") == "test_capability_tool"
        ]
        engineering_events = [
            {
                "type": event.get("type"), "name": event.get("name"),
                "success": event.get("success"),
            }
            for event in events
            if event.get("type") in {"tool_call", "tool_result"}
            and event.get("name") in {
                "write_engineering_file", "read_engineering_file",
                "list_engineering_files", "inspect_experiment_artifact",
                "run_engineering_command", "create_capability_tool",
                "test_capability_tool", "test_capability_hook",
            }
        ]
        report = {
            "acquisition_completed": True,
            "failure_class": failure_class,
            "suggested_hook": suggested_hook,
            "hook_tests": hook_tests,
            "generic_tests": generic_tests,
            "engineering_events": engineering_events,
            "implementation_attempted": any(
                event.get("type") == "tool_call"
                and event.get("name") in {
                    "write_engineering_file", "run_engineering_command",
                    "create_capability_tool", "test_capability_tool",
                    "test_capability_hook",
                }
                for event in events
            ),
            "tool_events": _tool_events(events),
            "trace_path": str(trace_path),
            "controller_tools_exposed": False,
            "evaluator_visible_to_agent": False,
        }
        (acquisition_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def author_round(round_id: int, prior: Mapping[str, Any]) -> Mapping[str, Any]:
        round_dir = args.output / f"round_{round_id:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        best_prior = sensor_only(prior.get("best_prior") or {})
        best_evidence = compact_sensor_evidence_for_prompt(
            best_prior.get("sensor_evidence") or {}
        )
        protected_stage = verified_stage_from_evidence(
            best_prior.get("sensor_evidence") or {}
        )
        required_revision = (
            {
                "base_program_id": str(best_prior["controller_id"]),
                "stage": protected_stage,
            }
            if args.controller_interface == "program"
            and best_prior.get("controller_id")
            and protected_stage is not None
            else None
        )
        graph_verified_aliases = list(
            ((best_prior.get("sensor_evidence") or {}).get("controller_graph") or {})
            .get("verified_prefix_aliases") or []
        )
        required_graph_revision = (
            {
                "base_graph_id": str(best_prior["controller_id"]),
                "frozen_node_aliases": graph_verified_aliases,
            }
            if args.controller_interface == "graph"
            and best_prior.get("controller_id")
            and graph_verified_aliases
            else None
        )
        program_store = None
        program_executions = 0
        program_executor = None
        graph_store = None
        graph_executions = 0
        graph_executor = None
        if args.controller_interface == "graph":
            node_store = StageNodeWorkspace(
                stage_node_workspace,
                python="/data/zxy/envs/vla-report/bin/python",
                timeout_sec=900, max_rpc_calls=10000,
                capability_workspace=capability_workspace,
            )
            graph_store = ControllerGraphWorkspace(
                workspace, nodes=node_store,
                required_revision=required_graph_revision,
                require_checkpoint_success=True,
                available_initial_context_fields={"task_instruction"},
            )

            def graph_executor(graph_id: str) -> Mapping[str, Any]:
                nonlocal graph_executions
                if graph_executions >= 1:
                    return {
                        "execution_completed": False,
                        "reason": "one controller graph execution is allowed per round",
                    }
                graph_executions += 1
                execution_index = graph_executions
                while (round_dir / f"graph_execution_{execution_index:03d}").exists():
                    execution_index += 1
                return execute_libero_graph(
                    graph_store, graph_id, suite="libero_spatial",
                    task=args.task, state=args.state, seed=args.seed,
                    output=round_dir / f"graph_execution_{execution_index:03d}",
                    capability_workspace=capability_workspace,
                )
        elif args.controller_interface == "program":
            program_store = ControllerProgramWorkspace(
                workspace, python="/data/zxy/envs/vla-report/bin/python",
                timeout_sec=900, max_rpc_calls=10000,
                capability_workspace=capability_workspace,
                required_revision=required_revision,
            )

            def program_executor(program_id: str) -> Mapping[str, Any]:
                nonlocal program_executions
                if program_executions >= 1:
                    return {
                        "execution_completed": False,
                        "reason": "one controller execution is allowed per autonomous round",
                    }
                program_executions += 1
                execution_index = program_executions
                # A killed/restarted outer loop may leave a complete or partial
                # immutable execution directory in this round. Never overwrite
                # evidence; allocate the next free attempt directory instead.
                while (round_dir / f"program_execution_{execution_index:03d}").exists():
                    execution_index += 1
                return execute_libero_program(
                    program_store, program_id, suite="libero_spatial",
                    task=args.task, state=args.state, seed=args.seed,
                    output=round_dir / f"program_execution_{execution_index:03d}",
                    capability_workspace=capability_workspace,
                )

        registrar = make_frontier_registrar(
            [f"libero_spatial:task_{args.task}"],
            ledger_path=str(ledger_path),
            state_path=str(args.output / "self_evolution_state.json"),
            controller_workspace=workspace if args.controller_interface == "spec" else None,
            controller_program_workspace=workspace if args.controller_interface == "program" else None,
            controller_program_executor=program_executor,
            required_controller_revision=required_revision,
            stage_node_workspace=(
                stage_node_workspace if args.controller_interface == "graph" else None
            ),
            controller_graph_workspace=(
                workspace if args.controller_interface == "graph" else None
            ),
            controller_graph_executor=graph_executor,
            required_graph_revision=required_graph_revision,
            controller_timeout_sec=900,
            max_controller_executions=1,
            capability_workspace=capability_workspace,
            engineering_workspace=args.output / "engineering_workspace",
            engineering_read_roots={
                "runs": args.output,
                "controllers": workspace,
                "capabilities": capability_workspace,
            },
        )
        harness = Harness(
            {"servers": [], "llm": {"provider": "mock", "model": "gpt-5.6-sol"},
             "skills": {"dir": str(ROOT / "capability_library" / "skills")},
             "context": {"compaction": {"enabled": False}}},
            model=_model(), builtin_registrar=registrar,
        )
        previous = compact_sensor_evidence_for_prompt(
            prior.get("sensor_evidence") or {}
        )
        reanalysis = None
        prior_round_id = prior.get("round")
        if prior_round_id is not None:
            reanalysis_path = (
                args.output / f"round_{int(prior_round_id):03d}"
                / "locked_attachment_reanalysis.json"
            )
            if reanalysis_path.is_file():
                reanalysis = sensor_only(json.loads(reanalysis_path.read_text()))
                if reanalysis.get("hardened_attachment_verified") is False:
                    previous = {
                        **previous,
                        "attachment_verified": False,
                        "placement_verified": False,
                        "sensor_only_conclusion": "attachment_not_verified",
                        "harness_sensor_reanalysis": reanalysis,
                    }
            motion_reanalysis_path = (
                args.output / f"round_{int(prior_round_id):03d}"
                / "sensor_only_motion_reanalysis.json"
            )
            if motion_reanalysis_path.is_file():
                motion_reanalysis = sensor_only(
                    json.loads(motion_reanalysis_path.read_text())
                )
                previous = {
                    **previous,
                    "action_outcomes": motion_reanalysis.get("action_outcomes", []),
                    "control_diagnostics": motion_reanalysis.get("control_diagnostics", {}),
                    "harness_motion_reanalysis": {
                        "protocol": motion_reanalysis.get("protocol"),
                        "source": motion_reanalysis.get("source"),
                        "evaluator_used": False,
                    },
                }
                previous = compact_sensor_evidence_for_prompt(previous)
        failure_history = sensor_only(prior.get("failure_history") or [])
        latest_acquisition = sensor_only(prior.get("latest_acquisition") or {})
        instruction = (
            f"This is autonomous evolution round {round_id} for development task "
            f"libero_spatial:task_{args.task}, state {args.state}, seed {args.seed}. "
            "The selectors are only for launching the adapter and must not appear "
            "in controller logic. "
            f"The live public task instruction is: {task_instruction!r}. Preserve "
            "every source, containment, destination, and spatial-relation clause. "
        )
        if args.controller_interface == "graph":
            instruction += (
                "The deployment Robot SDK contract is: "
                + json.dumps(robot_sdk_contract(), sort_keys=True)
                + "\nAuthor independent typed run_stage(robot, context) nodes and "
                "compose them as a Controller Graph. Do not create a monolithic "
                "run(robot) program. "
            )
        elif args.controller_interface == "program":
            instruction += (
                "The deployment Robot SDK contract is: "
                + json.dumps(robot_sdk_contract(), sort_keys=True)
                + "\nWrite actual run(robot) Python source. Use small helper functions "
                "and bounded loops. robot.act takes one action object with a sensor-derived "
                "target_eef_xyz, gripper, orientation, gains, and repeat count. "
            )
        if previous:
            instruction += (
                "The previous controller produced this complete sensor-only evidence. "
                "Treat it as the only allowed prior observation and improve it: "
                + json.dumps(previous, sort_keys=True) + "\n"
            )
            if reanalysis is not None:
                instruction += (
                    "A later Harness verifier audit found that the prior program could "
                    "supply its own motion baseline. The immutable sensor-only reanalysis "
                    "above supersedes that affected attachment claim. Do not optimize a "
                    "downstream stage until attachment passes the adapter-owned locked "
                    "baseline and source-vacated checks.\n"
                )
            if (
                args.controller_interface == "program"
                and previous.get("sensor_only_conclusion") == "controller_program_error"
                and program_store is not None
                and prior.get("controller_id")
            ):
                try:
                    inspected = program_store.inspect(str(prior["controller_id"]))
                    last_run = (inspected.get("runs") or [])[-1]
                    failed_rpc = next((
                        event for event in reversed(last_run.get("rpc_events") or [])
                        if event.get("error")
                    ), None)
                    diagnostics = sensor_only({
                        "error": last_run.get("error"),
                        "failed_rpc": failed_rpc,
                    })
                    instruction += (
                        "The deployment's legal program/RPC diagnostic is: "
                        + json.dumps(diagnostics, sort_keys=True) + "\n"
                    )
                except (FileNotFoundError, IndexError, KeyError, ValueError):
                    pass
        else:
            instruction += "There is no prior controller; start by loading both reusable Skills.\n"
        if (
            best_prior.get("controller_id")
        ):
            instruction += (
                "The persistent Harness stage memory identifies this earlier controller "
                "as the best strict sensor checkpoint: "
                + json.dumps({
                    "round": best_prior.get("round"),
                    "controller_id": best_prior.get("controller_id"),
                    "sensor_evidence": best_evidence,
                }, sort_keys=True)
                + "\nThe latest version regressed. Inspect that best controller and preserve "
                "its verified prefix; change only its unverified suffix.\n"
            )
            if required_revision is not None and program_store is not None:
                constraint = program_store.revision_constraint()
                instruction += (
                    "This is now enforced structurally, not merely requested: every new "
                    "program must preserve the exact source prefix from "
                    f"{constraint['base_program_id']} through line "
                    f"{constraint['protected_through_line']} "
                    f"({constraint['stage']} checkpoint, sha256 "
                    f"{constraint['protected_prefix_sha256']}). Inspect that program, "
                    "preserve the protected executable AST (comments and formatting may "
                    "differ), and edit only code after it.\n"
                )
            if required_graph_revision is not None:
                instruction += (
                    "The graph runtime structurally requires base_graph_id "
                    f"{required_graph_revision['base_graph_id']} and frozen node aliases "
                    f"{json.dumps(required_graph_revision['frozen_node_aliases'])}. "
                    "Inspect the graph and those node IDs. Reuse them directly; create "
                    "new node versions only for unfrozen failed stages. No source or AST "
                    "prefix comparison is used.\n"
                )
        if failure_history:
            instruction += (
                "The complete compact sensor-only failure history is: "
                + json.dumps(failure_history, sort_keys=True) + "\n"
            )
        if latest_acquisition:
            instruction += (
                "A separate Harness-owned capability acquisition phase just completed. "
                "Its audited result is: " + json.dumps(latest_acquisition, sort_keys=True)
                + "\nCall list_tested_capability_tools and bind a relevant newly tested "
                "Tool when its contract matches the failure.\n"
            )
        instruction += (
            "Run the full authoring workflow now. If the previous result failed, "
            "use public search and register useful leads when relevant. Call "
            "list_tested_capability_tools. "
        )
        if args.controller_interface == "graph":
            instruction += (
                "Create only the necessary immutable Stage Nodes, compose one complete "
                "typed Controller Graph, and execute that graph exactly once. Never "
                "create or execute a second graph in this round. "
            )
        else:
            instruction += (
                "Create one immutable complete controller and execute it exactly once. "
                "Never create or execute a second controller in this round. "
            )
        instruction += (
            "The persistent outer loop owns the next version. Use remaining turns only "
            "for failure research and capability creation/testing. Do not read or infer "
            "evaluator results."
        )
        try:
            live_attempt = 1
            while (round_dir / f"thea_trace_live_{live_attempt:03d}.jsonl").exists():
                live_attempt += 1
            events = _run_visible_harness_stream(
                harness, instruction,
                live_trace=round_dir / f"thea_trace_live_{live_attempt:03d}.jsonl",
                max_turns=args.max_turns_per_round,
                failure_budget=4,
                system_prompt_override=(
                    GRAPH_SYSTEM_PROMPT
                    if args.controller_interface == "graph" else SYSTEM_PROMPT
                ),
            )
        finally:
            harness.close()
        controller_id, evidence = _extract_execution(events)
        if controller_id is None or not evidence:
            attempt = 1
            while (round_dir / f"thea_trace_authoring_failure_{attempt:03d}.json").exists():
                attempt += 1
            failure_trace = round_dir / f"thea_trace_authoring_failure_{attempt:03d}.json"
            failure_report = round_dir / f"report_authoring_failure_{attempt:03d}.json"
            failure_trace.write_text(json.dumps(events, indent=2, default=str) + "\n")
            last_error = next((
                event.get("error") or event.get("final_text")
                for event in reversed(events)
                if event.get("type") in {"model_error", "done"}
                and (event.get("error") or event.get("final_text"))
            ), "authoring session ended without controller sensor evidence")
            failure_report.write_text(json.dumps({
                "round": round_id,
                "authoring_attempt": attempt,
                "controller_id": controller_id,
                "reason": str(last_error),
                "consumed_as_experiment_round": False,
                "trace_path": str(failure_trace),
                "evaluator_visible_to_agent": False,
            }, indent=2) + "\n")
            raise RuntimeError(
                "authoring produced no executed controller evidence; "
                f"preserved at {failure_trace}: {last_error}"
            )
        trace_path = round_dir / "thea_trace.json"
        trace_path.write_text(json.dumps(events, indent=2, default=str) + "\n")
        asset_events = []
        for event in events:
            if event.get("type") != "tool_result":
                continue
            result = event.get("result") or {}
            if event.get("name") in {"register_capability_asset", "register_public_research_lead", "self_evolve_from_failure"}:
                asset_events.append(sensor_only({"name": event.get("name"), "result": result}))
        report = {
            "round": round_id, "controller_id": controller_id,
            "sensor_evidence": evidence, "tool_events": _tool_events(events),
            "asset_events": asset_events, "trace_path": str(trace_path),
            "evaluator_visible_to_agent": False,
        }
        (round_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    result = AutonomousEvolutionLoop(
        EvolutionConfig(
            f"libero_spatial:task_{args.task}",
            max_rounds=args.max_rounds,
            acquisition_after_same_failure=args.acquisition_after_same_failure,
            force_acquisition_next_round=args.force_acquisition_next_round,
        ),
        state_path=state_path,
        author_round=author_round,
        acquire_capabilities=acquire_capabilities,
    ).run()
    if (
        args.controller_interface == "program"
        and result.get("status") == "sensor_success"
        and result.get("rounds")
    ):
        winning = result["rounds"][-1]
        program_id = winning.get("controller_id")
        evidence = winning.get("sensor_evidence") or {}
        if program_id:
            slug = re.sub(r"[^a-z0-9]+", "_", task_instruction.casefold()).strip("_")
            skill_name = ("learned_" + slug)[:80].rstrip("_")
            skill_programs = ControllerProgramWorkspace(
                workspace, python="/data/zxy/envs/vla-report/bin/python",
                timeout_sec=900, max_rpc_calls=10000,
                capability_workspace=capability_workspace,
            )
            task_skills = TaskSkillWorkspace(
                args.task_skill_workspace,
                controller_workspace=skill_programs,
                capability_workspace=capability_workspace,
                library_path=ROOT / "capability_library" / "library.json",
            )
            candidate = task_skills.create_candidate(
                name=skill_name,
                description=(
                    "Agent-learned sensor-grounded Tool orchestration for: "
                    + task_instruction
                ),
                semantic_task=task_instruction,
                program_id=str(program_id),
                development_evidence=evidence,
                development_context={
                    "environment": "libero_spatial",
                    "task_selector": args.task,
                    "state_selector": args.state,
                    "state_key": f"task-{args.task}:state-{args.state}:seed-{args.seed}",
                    "seed": args.seed,
                },
            )
            result["task_skill_candidate"] = candidate
            (args.output / "task_skill_candidate.json").write_text(
                json.dumps(candidate, indent=2) + "\n"
            )
    elif (
        args.controller_interface == "graph"
        and result.get("status") == "sensor_success"
        and result.get("rounds")
    ):
        winning = result["rounds"][-1]
        graph_id = winning.get("controller_id")
        evidence = winning.get("sensor_evidence") or {}
        if graph_id:
            slug = re.sub(r"[^a-z0-9]+", "_", task_instruction.casefold()).strip("_")
            skill_name = ("learned_graph_" + slug)[:80].rstrip("_")
            node_store = StageNodeWorkspace(
                stage_node_workspace,
                python="/data/zxy/envs/vla-report/bin/python",
                capability_workspace=capability_workspace,
            )
            graph_store = ControllerGraphWorkspace(workspace, nodes=node_store)
            task_skills = GraphTaskSkillWorkspace(
                args.task_skill_workspace, graph_workspace=graph_store,
                capability_workspace=capability_workspace,
                library_path=ROOT / "capability_library" / "library.json",
            )
            candidate = task_skills.create_candidate(
                name=skill_name,
                description="Agent-learned typed Controller Graph for: " + task_instruction,
                semantic_task=task_instruction, graph_id=str(graph_id),
                development_evidence=evidence,
                development_context={
                    "environment": "libero_spatial",
                    "task_selector": args.task,
                    "state_selector": args.state,
                    "state_key": f"task-{args.task}:state-{args.state}:seed-{args.seed}",
                    "seed": args.seed,
                },
            )
            result["task_skill_candidate"] = candidate
            (args.output / "task_skill_candidate.json").write_text(
                json.dumps(candidate, indent=2) + "\n"
            )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
