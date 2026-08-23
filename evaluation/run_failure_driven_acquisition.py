"""Let local Qwen drive Thea's public capability search from a dev failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "capability_library"),
    str(ROOT / "capability_library" / "tools"),
    str(ROOT / "Thea"),
]

from frontier_registrar import make_frontier_registrar
from harness import Harness
from qwen_local_model import LocalQwenVL
from harness.models.providers import OpenAICompatLLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--failure", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model", default="/data/zxy/cache/qwen2_5_vl")
    parser.add_argument("--provider", choices=("qwen", "apex"), default="qwen")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    registrar = make_frontier_registrar(
        [args.task],
        ledger_path=str(args.output / "capability_acquisition.jsonl"),
        state_path=str(args.output / "self_evolution_state.json"),
    )
    config = {
        "servers": [],
        "llm": {"provider": "mock", "model": "local-qwen"},
        "skills": {"dir": str(ROOT / "capability_library" / "skills")},
        "context": {"compaction": {"enabled": False}},
    }
    model = (
        OpenAICompatLLM(
            api_key=os.environ.get("APEX_API_KEY", ""),
            base_url="https://api.apexin.ai/v1",
            model="gpt-5.6-sol",
            temperature=0,
            max_tokens=1200,
            reasoning_effort="medium",
            max_retries=2,
            provider_label="apex-openai-compat",
        )
        if args.provider == "apex"
        else LocalQwenVL(args.model, device=args.device, max_new_tokens=350)
    )
    harness = Harness(config, model=model, builtin_registrar=registrar)
    events = []
    prompt = (
        f"Development task {args.task} failed: {args.failure}. "
        "Analyze the missing capability and use the public embodied resource search tool. "
        "Look for general pretrained policies, algorithms, or reusable tools, not task answers. "
        "Do not use reward, success, privileged simulator state, demonstrations from this task, "
        "or any sealed evaluation result."
    )
    try:
        events.extend(harness.run_stream(prompt, max_turns=3))
    finally:
        harness.close()

    tool_calls = []
    for event in events:
        if event.get("type") != "tool_call":
            continue
        result = next(
            (
                candidate
                for candidate in events
                if candidate.get("type") == "tool_result"
                and candidate.get("name") == event.get("name")
                and candidate.get("turn") == event.get("turn")
            ),
            {},
        )
        tool_calls.append(
            {
                "name": event.get("name"),
                "arguments": event.get("arguments", {}),
                "success": result.get("success"),
                "result": result.get("result"),
            }
        )
    report = {
        "protocol": "harness-acquired-task-zero-shot-v2",
        "model": "gpt-5.6-sol" if args.provider == "apex" else "Qwen2.5-VL-7B-Instruct-local",
        "task": args.task,
        "surface": "development_only",
        "failure": args.failure,
        "tool_calls": tool_calls,
        "errors": [
            {"type": event.get("type"), "error": event.get("error")}
            for event in events
            if event.get("type") in {"harness_error", "tool_error"}
        ],
        "sealed_results_consumed": False,
    }
    (args.output / "trace.json").write_text(json.dumps(events, indent=2, default=str) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
