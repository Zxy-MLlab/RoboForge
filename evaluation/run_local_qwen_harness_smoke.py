"""Run a real local-Qwen -> Thea Tool -> result self-evolution smoke."""

from __future__ import annotations

import json
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


def main() -> None:
    output = ROOT / "artifacts" / "local_qwen_harness_smoke"
    output.mkdir(parents=True, exist_ok=True)
    registrar = make_frontier_registrar(
        ["libero_object:task_3"],
        ledger_path=str(output / "capability_acquisition.jsonl"),
        state_path=str(output / "self_evolution_state.json"),
    )
    config = {
        "servers": [],
        "llm": {"provider": "mock", "model": "local-qwen"},
        "skills": {"dir": str(ROOT / "capability_library" / "skills")},
        "context": {"compaction": {"enabled": False}},
    }
    model = LocalQwenVL(
        str(Path("/data/zxy/cache/qwen2_5_vl")),
        device="cuda:0",
        max_new_tokens=300,
    )
    harness = Harness(config, model=model, builtin_registrar=registrar)
    events = []
    try:
        for event in harness.run_stream(
            "A development LIBERO episode failed because a bowl slipped during grasp. "
            "Analyze the failure, search public resources for general solutions, and "
            "record useful candidates. Do not use benchmark answers, success signals, "
            "or models trained on this task.",
            max_turns=3,
        ):
            events.append(event)
    finally:
        harness.close()
    report = {
        "protocol": "harness-acquired-task-zero-shot-v2",
        "model": "Qwen2.5-VL-7B-Instruct-local",
        "task_surface": "development_only",
        "sealed_results_consumed": False,
        "event_types": [event.get("type") for event in events],
        "errors": [
            {"type": event.get("type"), "error": event.get("error")}
            for event in events
            if event.get("type") in {"harness_error", "tool_error"}
        ],
        "tool_calls": [
            {
                "name": event.get("name"),
                "arguments": event.get("arguments", {}),
                "result_success": next(
                    (
                        result_event.get("success")
                        for result_event in events
                        if result_event.get("type") == "tool_result"
                        and result_event.get("id") == event.get("id")
                    ),
                    None,
                ),
                "result_keys": next(
                    (
                        sorted((result_event.get("result") or {}).keys())
                        for result_event in events
                        if result_event.get("type") == "tool_result"
                        and result_event.get("id") == event.get("id")
                    ),
                    [],
                ),
            }
            for event in events
            if event.get("type") == "tool_call"
        ],
    }
    (output / "trace.json").write_text(json.dumps(events, indent=2, default=str) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
