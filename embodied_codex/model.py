from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
import signal
import threading
import time
from typing import Any, Mapping, Protocol


class Model(Protocol):
    def decide(self, *, messages: list[Mapping[str, Any]],
               tools: list[Mapping[str, Any]]): ...


class ModelResponseTimeout(TimeoutError):
    pass


class ModelResponseIncomplete(RuntimeError):
    pass


@contextmanager
def _total_deadline(seconds: float):
    """Bound a stream even when a proxy keeps resetting read timeouts."""
    if (seconds <= 0 or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "setitimer")):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(_signum, _frame):
        raise ModelResponseTimeout(
            f"model response exceeded {float(seconds):g} seconds")

    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


class ResponsesHistory:
    """Client-owned, replayable Responses history grouped by causal turn."""

    protocol = "responses-stateless-history-v1"

    def __init__(self, *, max_chars: int = 300_000):
        self.max_chars = int(max_chars)
        self.system: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []
        self.compacted_turns: list[dict[str, Any]] = []
        self.current_state_fingerprint: str | None = None

    @staticmethod
    def _plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): ResponsesHistory._plain(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ResponsesHistory._plain(item) for item in value]
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            return ResponsesHistory._plain(dump())
        return value

    @staticmethod
    def _value(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    @classmethod
    def normalize_output_item(cls, item: Any) -> dict[str, Any]:
        """Convert a Response output item to the documented input shape."""
        item_type = cls._value(item, "type")
        if item_type == "reasoning":
            result = {"type": "reasoning"}
            for key in ("id", "summary", "content", "encrypted_content"):
                value = cls._value(item, key)
                if value is not None:
                    result[key] = cls._plain(value)
            return result
        if item_type == "function_call":
            result = {"type": "function_call"}
            for key in ("id", "call_id", "name", "arguments", "caller",
                        "namespace"):
                value = cls._value(item, key)
                if value is not None:
                    result[key] = cls._plain(value)
            for required in ("call_id", "name", "arguments"):
                if not result.get(required):
                    raise RuntimeError(
                        f"Responses function_call is missing {required}")
            return result
        if item_type == "message":
            role = str(cls._value(item, "role", "assistant"))
            content = []
            for part in cls._value(item, "content", []) or []:
                part_type = cls._value(part, "type")
                if part_type in {"output_text", "text", "input_text"}:
                    # Replayed assistant messages remain model output items.
                    # Apex accepts output_text (or refusal) for this role, not
                    # the input_text shape used by user/current-state messages.
                    content.append({"type": "output_text",
                                    "text": str(cls._value(part, "text", ""))})
                elif part_type in {"input_image", "image_url"}:
                    image_url = cls._value(part, "image_url")
                    if isinstance(image_url, Mapping):
                        image_url = image_url.get("url")
                    content.append({"type": "input_image", "image_url": image_url})
                else:
                    raise RuntimeError(
                        f"Unsupported Responses message content type: {part_type}")
            return {"type": "message", "role": role, "content": content}
        raise RuntimeError(f"Unsupported Responses output item type: {item_type}")

    @staticmethod
    def _response_content(content: Any) -> Any:
        if not isinstance(content, list):
            return content if content is not None else ""
        parts = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind in {"text", "input_text"}:
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif kind in {"image_url", "input_image"}:
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, Mapping) else image
                parts.append({"type": "input_image", "image_url": url})
            else:
                raise RuntimeError(f"Unsupported model input content type: {kind}")
        return parts

    def set_authoritative_messages(
            self, messages: list[Mapping[str, Any]]) -> dict[str, Any]:
        system = next((item for item in messages
                       if isinstance(item, Mapping)
                       and item.get("role") == "system"), None)
        state = next((item for item in messages
                      if isinstance(item, Mapping)
                      and item.get("role") == "user"), None)
        if system is not None:
            self.system = {"role": "system",
                           "content": self._response_content(system.get("content"))}
        if state is None:
            state = {"role": "user", "content": ""}
        current = {"role": "user",
                   "content": self._response_content(state.get("content"))}
        encoded = json.dumps(current, default=str, sort_keys=True,
                             separators=(",", ":"))
        self.current_state_fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        return current

    @staticmethod
    def _turn_complete(turn: Mapping[str, Any]) -> bool:
        calls = turn.get("calls") or {}
        return all(isinstance(value, Mapping)
                   and value.get("status") in {"completed", "failed"}
                   for value in calls.values())

    def append_response(self, *, response_id: str,
                        output: list[Any]) -> list[dict[str, Any]]:
        normalized = [self.normalize_output_item(item) for item in output]
        calls: dict[str, dict[str, str]] = {}
        for item in normalized:
            if item["type"] != "function_call":
                continue
            call_id = str(item["call_id"])
            if call_id in calls or any(
                    call_id in (turn.get("calls") or {}) for turn in self.turns):
                raise RuntimeError(f"Duplicate Responses function call id: {call_id}")
            calls[call_id] = {"name": str(item["name"]), "status": "pending"}
        self.turns.append({"response_id": str(response_id),
                           "output_items": normalized,
                           "continuation_items": [], "calls": calls})
        return normalized

    def record_tool_output(self, call_id: str, output: str, *,
                           multimodal_inputs: list[str] | None = None,
                           failed: bool = False) -> None:
        call_id = str(call_id)
        owner = next((turn for turn in reversed(self.turns)
                      if call_id in (turn.get("calls") or {})), None)
        if owner is None:
            raise RuntimeError(f"Unknown Responses function call id: {call_id}")
        call = owner["calls"][call_id]
        if call.get("status") != "pending":
            raise RuntimeError(f"Responses function call already resolved: {call_id}")
        owner["continuation_items"].append({
            "type": "function_call_output", "call_id": call_id,
            "output": str(output)})
        images = list(multimodal_inputs or [])
        if images:
            owner["continuation_items"].append({
                "role": "user", "content": [
                    {"type": "input_text", "text": "Selected sensor artifact."},
                    *({"type": "input_image", "image_url": url} for url in images),
                ]})
        call["status"] = "failed" if failed else "completed"

    def pending_call_ids(self) -> list[str]:
        return [str(call_id) for turn in self.turns
                for call_id, value in (turn.get("calls") or {}).items()
                if value.get("status") == "pending"]

    def _compaction_item(self) -> dict[str, Any] | None:
        if not self.compacted_turns:
            return None
        rows = self.compacted_turns[-32:]
        value = {"type": "responses_history_compaction",
                 "total_turns": len(self.compacted_turns), "turns": rows}
        return {"role": "user", "content": [{"type": "input_text",
            "text": json.dumps(value, sort_keys=True, separators=(",", ":"))}]}

    @staticmethod
    def _turn_items(turn: Mapping[str, Any]) -> list[dict[str, Any]]:
        continuation = list(turn.get("continuation_items", []))
        call_outputs = [item for item in continuation
                        if item.get("type") == "function_call_output"]
        additional_inputs = [item for item in continuation
                             if item.get("type") != "function_call_output"]
        return ([copy.deepcopy(item) for item in turn.get("output_items", [])]
                + [copy.deepcopy(item) for item in call_outputs]
                + [copy.deepcopy(item) for item in additional_inputs])

    def _serialized_size(self, current_state: Mapping[str, Any]) -> int:
        return len(json.dumps(self.serialize(current_state, compact=False),
                              default=str, separators=(",", ":")).encode())

    @staticmethod
    def _routing_references(turn: Mapping[str, Any]) -> dict[str, Any]:
        references: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, str):
                if value.startswith(("artifact://", "evidence://", "run://")):
                    references.append(value)
                return
            if isinstance(value, Mapping):
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        for item in turn.get("continuation_items", []):
            if item.get("type") != "function_call_output":
                continue
            output = item.get("output")
            try:
                visit(json.loads(output) if isinstance(output, str) else output)
            except (TypeError, ValueError):
                visit(output)
        unique = list(dict.fromkeys(references))
        result: dict[str, Any] = {"references": unique[:16]}
        if len(unique) > 16:
            result["reference_count"] = len(unique)
            result["omitted_reference_count"] = len(unique) - 16
        return result

    def _compact(self, current_state: Mapping[str, Any]) -> None:
        # Remove only whole, resolved turns and always retain the newest causal
        # chain. Pending calls can never be separated from their outputs.
        while self._serialized_size(current_state) > self.max_chars:
            removable = next((index for index, turn in enumerate(self.turns[:-1])
                              if self._turn_complete(turn)), None)
            if removable is None:
                break
            turn = self.turns.pop(removable)
            self.compacted_turns.append({
                "response_id": turn.get("response_id"),
                "calls": [{"call_id": call_id, "name": value.get("name"),
                           "status": value.get("status")}
                          for call_id, value in (turn.get("calls") or {}).items()],
                **self._routing_references(turn)})

    def serialize(self, current_state: Mapping[str, Any], *,
                  compact: bool = True) -> list[dict[str, Any]]:
        if compact:
            self._compact(current_state)
        result = []
        if self.system is not None:
            result.append(copy.deepcopy(self.system))
        result.append(copy.deepcopy(dict(current_state)))
        summary = self._compaction_item()
        if summary is not None:
            result.append(summary)
        for turn in self.turns:
            result.extend(self._turn_items(turn))
        return result

    def replay_counts(self, input_items: list[Mapping[str, Any]]) -> dict[str, int]:
        def image_count(value: Any) -> int:
            if isinstance(value, Mapping):
                return (1 if value.get("type") == "input_image" else 0) + sum(
                    image_count(item) for item in value.values())
            if isinstance(value, list):
                return sum(image_count(item) for item in value)
            return 0

        return {
            "reasoning_items_replayed": sum(
                item.get("type") == "reasoning" for item in input_items),
            "function_calls_replayed": sum(
                item.get("type") == "function_call" for item in input_items),
            "function_outputs_replayed": sum(
                item.get("type") == "function_call_output" for item in input_items),
            "multimodal_inputs_replayed": image_count(input_items),
        }

    def state(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "max_chars": self.max_chars,
                "system": copy.deepcopy(self.system),
                "turns": copy.deepcopy(self.turns),
                "compacted_turns": copy.deepcopy(self.compacted_turns),
                "current_state_fingerprint": self.current_state_fingerprint}

    def restore(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            return
        protocol = state.get("protocol")
        if protocol != self.protocol:
            raise RuntimeError(
                f"Unsupported Responses transport checkpoint protocol: {protocol}")
        self.max_chars = int(state.get("max_chars", self.max_chars))
        system = state.get("system")
        self.system = copy.deepcopy(dict(system)) if isinstance(system, Mapping) else None
        self.turns = copy.deepcopy(list(state.get("turns") or []))
        self.compacted_turns = copy.deepcopy(
            list(state.get("compacted_turns") or []))
        self.current_state_fingerprint = (
            str(state["current_state_fingerprint"])
            if state.get("current_state_fingerprint") else None)
        pending = self.pending_call_ids()
        if pending:
            raise RuntimeError(
                "Responses checkpoint contains unresolved function calls: "
                + ",".join(pending))


@dataclass
class OpenAIModel:
    api_key: str
    base_url: str
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"
    reasoning_context: str = "all_turns"
    max_tokens: int = 8000
    timeout: float = 120
    total_response_timeout: float = 120
    retry_delays: tuple[float, ...] = (1.0, 2.0)
    provider: str | None = None
    max_history_chars: int = 300_000

    def __post_init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=self.timeout, max_retries=0)
        self.history = ResponsesHistory(max_chars=self.max_history_chars)
        self.audit_log: list[dict[str, Any]] = []

    @staticmethod
    def _transient(error: BaseException) -> bool:
        status = getattr(error, "status_code", None)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        return type(error).__name__ in {
            "APIConnectionError", "APITimeoutError", "RateLimitError",
            "InternalServerError", "TimeoutError", "ConnectionError"}

    @staticmethod
    def _value(item: Any, name: str, default: Any = None) -> Any:
        return ResponsesHistory._value(item, name, default)

    @classmethod
    def _response_tool(cls, tool: Mapping[str, Any]) -> dict[str, Any]:
        """Convert the Kernel's Chat-style schema to the Responses schema."""
        function = tool.get("function") if isinstance(
            tool.get("function"), Mapping) else tool
        result = {"type": "function", "name": function.get("name"),
                  "description": function.get("description", ""),
                  "parameters": function.get("parameters", {"type": "object"})}
        if "strict" in function:
            result["strict"] = function["strict"]
        return result

    def transport_state(self) -> dict[str, Any]:
        """Serializable canonical history; encrypted reasoning stays opaque."""
        return self.history.state()

    def restore_transport_state(self, state: Mapping[str, Any] | None) -> None:
        self.history.restore(state)

    def record_tool_output(self, call_id: str, output: str, *,
                           multimodal_inputs: list[str] | None = None,
                           failed: bool = False) -> None:
        self.history.record_tool_output(
            call_id, output, multimodal_inputs=multimodal_inputs, failed=failed)

    def _usage(self, response: Any) -> Any:
        usage = self._value(response, "usage")
        if usage is not None and not isinstance(usage, Mapping):
            dump = getattr(usage, "model_dump", None)
            if callable(dump):
                return dump()
            to_dict = getattr(usage, "to_dict", None)
            if callable(to_dict):
                return to_dict()
            return {key: self._value(usage, key) for key in
                    ("input_tokens", "output_tokens", "total_tokens",
                     "prompt_tokens", "completion_tokens",
                     "input_tokens_details", "output_tokens_details")
                    if self._value(usage, key) is not None}
        return usage

    def _audit(self, response: Any, *, tool_call_count: int,
               input_items: list[Mapping[str, Any]]) -> dict[str, Any]:
        reasoning = self._value(response, "reasoning")
        effective_context = (self._value(reasoning, "context")
                             if reasoning is not None
                             else self._value(response, "reasoning_context"))
        usage = self._usage(response)
        output_details = (usage.get("output_tokens_details")
                          if isinstance(usage, Mapping) else None)
        reasoning_tokens = (output_details.get("reasoning_tokens")
                            if isinstance(output_details, Mapping) else None)
        response_status = (self._value(response, "status")
                           or self._value(response, "finish_reason"))
        encoded = json.dumps(input_items, default=str,
                             separators=(",", ":")).encode()
        audit = {
            "provider": self.provider or (
                "apex" if "apexin.ai" in self.base_url else "openai"),
            "requested_model": self.model,
            "effective_model": self._value(response, "model"),
            "reasoning_effort": self.reasoning_effort,
            "reasoning_context": self.reasoning_context,
            "response_id": self._value(response, "id"),
            "previous_response_id_used": None,
            "usage": usage,
            "reasoning_tokens": reasoning_tokens,
            "response_status": response_status,
            "finish_status": response_status,
            "effective_reasoning_context": effective_context,
            "tool_call_count": tool_call_count,
            "history_item_count": len(input_items),
            "serialized_history_bytes": len(encoded),
            "estimated_history_tokens": max(1, len(encoded) // 4),
            **self.history.replay_counts(input_items),
        }
        self.audit_log.append(audit)
        return audit

    def _decide_once(self, *, messages, tools):
        if (self.provider or "").lower() == "apex":
            return self._decide_chat(messages=messages, tools=tools)
        if not hasattr(self.client, "responses"):
            raise RuntimeError("OpenAI SDK/client does not support the Responses API")
        pending = self.history.pending_call_ids()
        if pending:
            raise RuntimeError(
                "Responses continuation has unexecuted function calls: "
                + ",".join(pending))
        current_state = self.history.set_authoritative_messages(list(messages))
        input_items = self.history.serialize(current_state)
        with _total_deadline(self.total_response_timeout):
            response = self.client.responses.create(
                model=self.model,
                input=input_items,
                tools=[self._response_tool(tool) for tool in tools],
                reasoning={"effort": self.reasoning_effort,
                           "context": self.reasoning_context},
                include=["reasoning.encrypted_content"],
                max_output_tokens=self.max_tokens)
        output = self._value(response, "output", []) or []
        calls = []
        text_parts = []
        output_text = self._value(response, "output_text")
        for item in output:
            item_type = self._value(item, "type")
            if item_type == "function_call":
                calls.append({
                    "id": self._value(item, "call_id")
                          or self._value(item, "id") or "",
                    "name": self._value(item, "name") or "",
                    "arguments": self._value(item, "arguments") or "{}"})
            elif item_type == "message" and not output_text:
                for part in self._value(item, "content", []) or []:
                    if self._value(part, "type") in {"output_text", "text"}:
                        value = self._value(part, "text")
                        if value:
                            text_parts.append(str(value))
        if output_text:
            text_parts = [str(output_text)]
        response_id = self._value(response, "id")
        if not response_id:
            raise RuntimeError("Responses API returned no response id for audit/history")
        status = self._value(response, "status")
        if status != "completed":
            self._audit(response, tool_call_count=len(calls),
                        input_items=input_items)
            raise ModelResponseIncomplete(
                f"Responses API returned non-completed status: {status}")
        normalized = self.history.append_response(
            response_id=str(response_id), output=list(output))
        audit = self._audit(response, tool_call_count=len(calls),
                            input_items=input_items)
        return {"content": "".join(text_parts), "tool_calls": calls,
                "response_items": normalized, "audit": audit}

    @staticmethod
    def _chat_messages(messages):
        converted = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            item = {"role": str(message.get("role", "user"))}
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, Mapping):
                        continue
                    kind = part.get("type")
                    if kind in {"text", "input_text", "output_text"}:
                        parts.append({"type": "text", "text": str(part.get("text", ""))})
                    elif kind in {"image_url", "input_image"}:
                        image = part.get("image_url")
                        url = image.get("url") if isinstance(image, Mapping) else image
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                item["content"] = parts
            else:
                item["content"] = str(content)
            if item["role"] == "assistant" and message.get("tool_calls"):
                item["tool_calls"] = list(message["tool_calls"])
            if item["role"] == "tool":
                item["tool_call_id"] = str(message.get("tool_call_id", ""))
            converted.append(item)
        return converted

    def _decide_chat(self, *, messages, tools):
        if not hasattr(self.client, "chat"):
            raise RuntimeError("OpenAI SDK/client does not support Chat Completions")
        current_state = self.history.set_authoritative_messages(list(messages))
        with _total_deadline(self.total_response_timeout):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._chat_messages(messages),
                tools=[{"type": "function", "function": dict(tool.get("function") or tool)}
                       for tool in tools],
                tool_choice="auto",
                max_tokens=self.max_tokens)
        choice = (self._value(response, "choices") or [None])[0]
        message = self._value(choice, "message")
        if message is None:
            raise ModelResponseIncomplete("Chat Completions returned no message")
        calls = []
        history_output = []
        for item in self._value(message, "tool_calls", []) or []:
            function = self._value(item, "function")
            call_id = self._value(item, "id") or ""
            name = self._value(function, "name") or ""
            arguments = self._value(function, "arguments") or "{}"
            calls.append({"id": call_id, "name": name, "arguments": arguments})
            history_output.append({"type": "function_call", "call_id": call_id,
                                   "name": name, "arguments": arguments})
        content = self._value(message, "content") or ""
        if content:
            history_output.insert(0, {"type": "message", "role": "assistant",
                                      "content": [{"type": "text", "text": str(content)}]})
        response_id = self._value(response, "id") or hashlib.sha256(
            json.dumps({"messages": messages, "response": response},
                       default=str, sort_keys=True).encode()).hexdigest()
        self.history.append_response(response_id=str(response_id), output=history_output)
        usage = self._usage(response)
        audit = {"provider": self.provider or "apex", "requested_model": self.model,
                 "effective_model": self._value(response, "model"),
                 "reasoning_effort": self.reasoning_effort,
                 "response_id": self._value(response, "id"),
                 "previous_response_id_used": None, "usage": usage,
                 "response_status": self._value(choice, "finish_reason"),
                 "finish_status": self._value(choice, "finish_reason"),
                 "tool_call_count": len(calls),
                 "history_item_count": len(messages),
                 "serialized_history_bytes": len(json.dumps(messages, default=str).encode()),
                 "estimated_history_tokens": max(1, len(json.dumps(messages, default=str)) // 4)}
        self.audit_log.append(audit)
        return {"content": str(content), "tool_calls": calls, "audit": audit}

    def decide(self, *, messages, tools):
        for attempt in range(len(self.retry_delays) + 1):
            try:
                return self._decide_once(messages=messages, tools=tools)
            except Exception as exc:
                if attempt >= len(self.retry_delays) or not self._transient(exc):
                    raise
                time.sleep(max(0.0, float(self.retry_delays[attempt])))
        raise AssertionError("unreachable")


__all__ = ["Model", "OpenAIModel", "ResponsesHistory",
           "ModelResponseTimeout", "ModelResponseIncomplete"]
