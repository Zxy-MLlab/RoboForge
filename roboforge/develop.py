"""Canonical multi-state OpenHands development entry point."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .harness.campaign import load_campaign_config
from .workspace.project import ProjectWorkspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roboforge develop")
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("ROBOFORGE_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--base-url", default=os.getenv("ROBOFORGE_MODEL_BASE_URL", "https://api.apexin.ai/v1"))
    parser.add_argument("--api-key-env", default=os.getenv("ROBOFORGE_MODEL_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=os.getenv("ROBOFORGE_REASONING_EFFORT", "medium"),
                        choices=["low", "medium", "high", "xhigh"])
    parser.add_argument("--adapter-python", default=os.getenv("ROBOFORGE_ADAPTER_PYTHON", "/root/autodl-tmp/mj311/bin/python"))
    args = parser.parse_args(argv)
    config = load_campaign_config(args.config)
    run = args.run_dir.resolve(); workspace = run / "workspace"
    controller = ProjectWorkspace(workspace).initialize()
    manifest = {
        "schema_version": "roboforge-canonical-campaign-v1",
        "runtime": config.runtime, "task": config.task,
        "development_states": list(config.development_states),
        "held_out_split": config.held_out_split,
        "max_valid_trials": config.max_valid_trials,
        "max_iterations": args.max_iterations or config.max_iterations,
        "controller_mode": config.controller_mode,
        "controller": str(controller), "started_unix": time.time(),
        "termination_reason": None,
    }
    (run / "campaign-manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (run / "campaign-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    ledger = run / "development-trial-ledger.json"
    ledger.write_text(json.dumps({"schema_version": 1, "allowed_states": list(config.development_states),
                                  "valid_trial_budget": config.max_valid_trials, "records": []},
                                 indent=2, sort_keys=True) + "\n")

    from openhands.sdk import LLM
    from . import create_openhands_conversation
    key = os.getenv(args.api_key_env)
    if not key:
        raise SystemExit(f"missing model credential environment variable: {args.api_key_env}")
    llm = LLM(model=args.model, api_key=key, base_url=args.base_url,
              api_mode="chat" if args.base_url.rstrip("/").endswith("/v1") else "auto",
              max_output_tokens=12000, reasoning_effort=args.reasoning_effort,
              usage_id=f"roboforge:develop:{run.name}")
    state_list = ", ".join(str(item) for item in config.development_states)
    prompt = f"""Develop the LIBERO task {config.task} as one reusable program in this workspace.
The Harness allowlist permits development initial-state indices {state_list}; they are supplied
only to ordinary Terminal commands and must never be read by the Controller. Run an allowed
state with:
python -m roboforge run controllers/controller.py --runtime libero --task {config.task}
--seed STATE --run-dir experiments/state-STATE --adapter-python $ROBOFORGE_ADAPTER_PYTHON
Inspect stdout and experiment evidence after every run, edit the Controller and the
workspace robot_sdk/capabilities/runtime_adapters code as needed, and regress across
multiple states. Hidden evaluation states are verifier-private and are not present in this
workspace. Continue
until the development budget is exhausted or a single unified candidate is ready, then
write a final candidate manifest. Use only ordinary Editor, Terminal, search, Git and
the other generic OpenHands tools; there are no robot-specific LLM tools.
This is an execution task, not a request for a plan or acknowledgement. Your first response
must invoke a public workspace or Terminal tool to inspect the current project and task
context. Do not finish with prose before performing at least one tool call and running the
current Controller once. After every tool observation, continue with another public tool
call; a text-only progress update is not a valid stopping condition. Only stop after the
current Controller has actually been run and its result has been inspected, or when the
OpenHands SDK reports an explicit budget, error, or FinishTool termination."""
    conversation = create_openhands_conversation(
        llm=llm, workspace=workspace, persistence_dir=run / "openhands",
        service=None, controller_path=controller,
        max_iterations=args.max_iterations or config.max_iterations,
        terminal_env={"ROBOFORGE_WORKSPACE": str(workspace), "PYTHONPATH": str(Path(__file__).parents[1]),
                      "ROBOFORGE_ADAPTER_PYTHON": args.adapter_python,
                      "ROBOFORGE_DEVELOPMENT_STATES": json.dumps(list(config.development_states)),
                      "ROBOFORGE_VALID_TRIAL_BUDGET": str(config.max_valid_trials),
                      "ROBOFORGE_CAMPAIGN_LEDGER": str(ledger)},
    )
    try:
        conversation.send_message(prompt)
        conversation.run()
        manifest["termination_reason"] = "conversation_complete"
    finally:
        conversation.close()
        manifest["finished_unix"] = time.time()
        (run / "campaign-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


__all__ = ["main"]
