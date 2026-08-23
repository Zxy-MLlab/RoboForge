"""Let Thea author and execute an immutable LIBERO controller candidate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "capability_library"),
    str(ROOT / "capability_library" / "tools"),
    str(ROOT / "Thea"),
]

from frontier_registrar import make_frontier_registrar
from harness import Harness
from harness.models.providers import OpenAICompatLLM


SYSTEM_PROMPT = """## System Prompt
You are the controller-authoring agent for a zero-cheating embodied benchmark.
Act like a Coding Agent: compose reusable capabilities into an immutable
standalone controller, execute it, inspect sensor-only evidence, and revise it
when justified. Choose one tool per turn. A controller specification must be
generic across task IDs and states. Never encode benchmark coordinates,
task/state branches, reward, done, evaluator results, BDDL, simulator object
poses, or segmentation IDs. Use only RGB/RGB-D, calibration, language,
proprioception, public task-disjoint models, and action history. Evaluator
success is hidden and cannot drive revisions. Do not claim that a requested
stage is implemented when run evidence reports it unsupported. Before creating
a controller, load and follow the reusable Skill named
autonomous-closed-loop-grasp-place-recovery and then load
visual-articulated-drawer-open-and-retrieve. Treat their validated
parameter/result memory as stronger evidence than guessing a previously failed
configuration. Articulation stages remain language-conditioned and must never
branch on a task selector.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=3)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-workspace", type=Path)
    parser.add_argument("--prior-controller")
    parser.add_argument("--sensor-evidence", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("APEX_API_KEY", "")
    if not api_key:
        raise OSError("APEX_API_KEY is required")
    model = OpenAICompatLLM(
        api_key=api_key,
        base_url="https://api.apexin.ai/v1",
        model="gpt-5.6-sol",
        temperature=0,
        max_tokens=1600,
        reasoning_effort="medium",
        max_retries=2,
        provider_label="apex-openai-compat",
    )
    workspace = args.controller_workspace or (args.output / "controllers")
    registrar = make_frontier_registrar(
        [f"libero_spatial:task_{args.task}"],
        ledger_path=str(args.output / "capability_acquisition.jsonl"),
        state_path=str(args.output / "self_evolution_state.json"),
        controller_workspace=workspace,
    )
    config = {
        "servers": [],
        "llm": {"provider": "mock", "model": "gpt-5.6-sol"},
        "skills": {"dir": str(ROOT / "capability_library" / "skills")},
        "context": {"compaction": {"enabled": False}},
    }
    harness = Harness(config, model=model, builtin_registrar=registrar)
    events: list[dict] = []
    prior_context = ""
    if args.prior_controller:
        evidence = ""
        if args.sensor_evidence and args.sensor_evidence.is_file():
            evidence = args.sensor_evidence.read_text()
        prior_context = (
            f"A prior immutable controller {args.prior_controller} in an external "
            "frozen workspace produced the following evidence. It is intentionally "
            "not resolvable by inspect_controller_script in this new workspace; do "
            "not spend a tool call trying to inspect it. Treat the supplied sensor "
            "evidence as the complete allowed prior observation. It "
            "is sensor-only. Create a new version "
            "whose generic parameters address only evidence-supported failures, "
            "then execute the new version. Never overwrite the prior version. "
            f"Evidence: {evidence}\n"
        )
    instruction = prior_context + (
        "First load the autonomous-closed-loop-grasp-place-recovery Skill and "
        "then load the visual-articulated-drawer-open-and-retrieve Skill. Then "
        "create and execute a generic closed-loop pick-and-place controller for "
        f"the LIBERO-Spatial development environment, task selector {args.task}, "
        f"state selector {args.state}, seed {args.seed}. The selector is only for "
        "launching the environment and must not appear in controller logic. Use "
        "live open-vocabulary detection, physical-region selection, conditional "
        "visual stages detect_articulated_handle, open_drawer, verify_articulation, "
        "reobserve_after_articulation, SAM, ranked "
        "GraspNet candidates, guarded execution, visual attachment verification, "
        "visual placement verification, and bounded correction. First create one "
        "controller script, then execute it once. Inspect the sensor evidence and "
        "accurately report unsupported stages. Do not use evaluator feedback."
    )
    try:
        events.extend(
            harness.run_stream(
                instruction,
                max_turns=args.max_turns,
                failure_budget=3,
                system_prompt_override=SYSTEM_PROMPT,
            )
        )
    finally:
        harness.close()

    (args.output / "thea_trace.json").write_text(
        json.dumps(events, indent=2, default=str) + "\n"
    )
    report = {
        "protocol": "embodied-coding-agent-harness-v1",
        "task_surface": "development_only",
        "model": "gpt-5.6-sol",
        "task": args.task,
        "state": args.state,
        "event_types": [event.get("type") for event in events],
        "tool_calls": [
            {"name": event.get("name"), "arguments": event.get("arguments")}
            for event in events
            if event.get("type") == "tool_call"
        ],
        "errors": [
            event
            for event in events
            if event.get("type") in {"harness_error", "tool_error", "model_error"}
        ],
        "evaluator_visible_to_agent": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
