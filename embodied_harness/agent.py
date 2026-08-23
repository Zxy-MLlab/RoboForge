"""Persistent standalone tool-calling Agent loop."""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

from .model import Model
from .tool_registry import ToolRegistry


class Agent:
    def __init__(
        self, *, model: Model, tools: ToolRegistry, system_prompt: str,
        trace_path: str | Path, max_turns: int = 40,
        max_model_attempts: int = 3, model_retry_delay_seconds: float = 2.0,
    ) -> None:
        self.model = model; self.tools = tools; self.system_prompt = system_prompt
        self.trace_path = Path(trace_path); self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_turns = int(max_turns)
        self.max_model_attempts = max(1, int(max_model_attempts))
        self.model_retry_delay_seconds = max(0.0, float(model_retry_delay_seconds))

    def _emit(self, event: Mapping[str, Any]) -> None:
        with self.trace_path.open("a") as stream:
            stream.write(json.dumps({"unix": time.time(), **dict(event)}, default=str) + "\n")

    def _resume_after_transport_error(
        self, instruction: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int] | None:
        """Rebuild the last unfinished conversation without replaying its tools."""
        if not self.trace_path.is_file():
            return None
        try:
            events = [json.loads(line) for line in self.trace_path.read_text().splitlines()]
        except (OSError, json.JSONDecodeError):
            return None
        task_indices = [
            index for index, event in enumerate(events)
            if event.get("type") == "task" and event.get("instruction") == instruction
        ]
        if not task_indices:
            return None
        segment = events[task_indices[-1] + 1:]
        meaningful = [event for event in segment if event.get("type") != "resume"]
        if not meaningful or meaningful[-1].get("type") != "model_error":
            return None
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": instruction},
        ]
        tool_results: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        last_turn = 0
        for event in segment:
            kind = event.get("type")
            if kind == "model":
                calls = list(event.get("tool_calls") or [])
                content = str(event.get("content") or "")
                assistant: dict[str, Any] = {"role": "assistant", "content": content}
                if calls:
                    assistant["tool_calls"] = [
                        {"id": call["id"], "type": "function",
                         "function": {"name": call["name"],
                                      "arguments": call.get("arguments") or "{}"}}
                        for call in calls
                    ]
                messages.append(assistant)
                pending = list(calls)
                last_turn = max(last_turn, int(event.get("turn") or 0))
            elif kind == "tool_result":
                if not pending:
                    return None
                call_index = next(
                    (i for i, call in enumerate(pending)
                     if call.get("name") == event.get("name")), 0,
                )
                call = pending.pop(call_index)
                if event.get("ok") is True:
                    payload = {"ok": True, "result": event.get("result")}
                else:
                    payload = {"ok": False, "error": event.get("error")}
                tool_results.append({"name": event.get("name"), **payload})
                messages.append({
                    "role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(payload, default=str),
                })
            elif kind == "model_error":
                last_turn = max(last_turn, int(event.get("turn") or 0) - 1)
        if pending:
            return None
        return messages, tool_results, last_turn + 1

    def run(self, instruction: str) -> dict[str, Any]:
        resumed = self._resume_after_transport_error(instruction)
        if resumed is None:
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": instruction},
            ]
            tool_results: list[dict[str, Any]] = []
            start_turn = 1
            self._emit({"type": "task", "instruction": instruction})
        else:
            messages, tool_results, start_turn = resumed
            self._emit({"type": "resume", "turn": start_turn,
                        "recovered_tool_results": len(tool_results)})
        for turn in range(start_turn, self.max_turns + 1):
            decision = None
            for attempt in range(1, self.max_model_attempts + 1):
                try:
                    decision = dict(self.model.decide(
                        messages=messages, tools=self.tools.schemas,
                    ))
                    break
                except Exception as exc:
                    error = f"model_transport_error: {type(exc).__name__}: {exc}"
                    self._emit({"type": "model_error", "turn": turn,
                                "attempt": attempt,
                                "will_retry": attempt < self.max_model_attempts,
                                "error": error})
                    if attempt < self.max_model_attempts:
                        time.sleep(self.model_retry_delay_seconds * attempt)
            if decision is None:
                return {"completed": False, "error": error,
                        "tool_results": tool_results, "turns": turn}
            calls = list(decision.get("tool_calls") or [])
            content = str(decision.get("content") or "")
            self._emit({"type": "model", "turn": turn, "content": content,
                        "tool_calls": calls})
            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if calls:
                assistant_message["tool_calls"] = [
                    {"id": call["id"], "type": "function",
                     "function": {"name": call["name"],
                                  "arguments": call.get("arguments") or "{}"}}
                    for call in calls
                ]
            messages.append(assistant_message)
            if not calls:
                return {"completed": True, "final_text": content,
                        "tool_results": tool_results, "turns": turn}
            # Execute every model-issued call in order. The standalone protocol
            # does not silently discard parallel tool calls.
            for call in calls:
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                    result = self.tools.call(str(call["name"]), arguments)
                    payload = {"ok": True, "result": result}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                tool_results.append({"name": call["name"], **payload})
                self._emit({"type": "tool_result", "turn": turn,
                            "name": call["name"], **payload})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(payload, default=str)})
        return {"completed": False, "error": "agent turn budget exhausted",
                "tool_results": tool_results, "turns": self.max_turns}


__all__ = ["Agent"]
