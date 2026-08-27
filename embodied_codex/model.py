from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import signal
import threading
import time
import hashlib
import json
from typing import Any, Mapping, Protocol


class Model(Protocol):
    def decide(self, *, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]]): ...


class ModelResponseTimeout(TimeoutError):
    pass


class ModelResponseIncomplete(RuntimeError):
    pass


@contextmanager
def _total_deadline(seconds: float):
    """Bound a stream even when a proxy keeps resetting read timeouts."""
    if (seconds<=0 or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal,"setitimer")):
        yield;return
    previous_handler=signal.getsignal(signal.SIGALRM)
    def expired(_signum,_frame):
        raise ModelResponseTimeout(f"model response exceeded {float(seconds):g} seconds")
    signal.signal(signal.SIGALRM,expired)
    previous_timer=signal.setitimer(signal.ITIMER_REAL,float(seconds))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,*previous_timer)
        signal.signal(signal.SIGALRM,previous_handler)


@dataclass
class OpenAIModel:
    api_key: str; base_url: str; model: str = "gpt-5.6-sol"
    reasoning_effort: str = "high"; max_tokens: int = 8000; timeout: float = 120
    total_response_timeout: float = 120
    retry_delays: tuple[float, ...] = (1.0, 2.0)
    provider: str | None = None
    def __post_init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=self.timeout, max_retries=0)
        self.previous_response_id: str | None = None
        self._pending_call_ids: list[str] = []
        self._sent_call_ids: set[str] = set()
        self._sent_extra_inputs: set[str] = set()
        self._last_state_fingerprint: str | None = None
        self.audit_log: list[dict[str, Any]] = []
    @staticmethod
    def _transient(error: BaseException) -> bool:
        status = getattr(error, "status_code", None)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        name = type(error).__name__
        return name in {"APIConnectionError", "APITimeoutError", "RateLimitError",
                        "InternalServerError", "TimeoutError", "ConnectionError"}

    @staticmethod
    def _value(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    @classmethod
    def _response_tool(cls, tool: Mapping[str, Any]) -> dict[str, Any]:
        """Convert the Kernel's Chat-style schema to the Responses schema."""
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        result = {"type": "function", "name": function.get("name"),
                  "description": function.get("description", ""),
                  "parameters": function.get("parameters", {"type": "object"})}
        if "strict" in function:
            result["strict"] = function["strict"]
        return result

    @classmethod
    def _response_content(cls, content: Any) -> Any:
        if not isinstance(content, list):
            return content if content is not None else ""
        parts = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            kind = part.get("type")
            if kind == "text":
                parts.append({"type": "input_text", "text": part.get("text", "")})
            elif kind == "image_url":
                image = part.get("image_url")
                url = image.get("url") if isinstance(image, Mapping) else image
                parts.append({"type": "input_image", "image_url": url})
            else:
                parts.append(dict(part))
        return parts

    @classmethod
    def _message_input(cls, message: Mapping[str, Any]) -> list[dict[str, Any]]:
        role = str(message.get("role", "user"))
        if role == "tool":
            return [{"type": "function_call_output",
                     "call_id": str(message.get("tool_call_id", "")),
                     "output": str(message.get("content", ""))}]
        if role == "assistant" and message.get("tool_calls"):
            # Previous Responses already owns these function calls.  Keeping
            # them in a new input would duplicate the model's own output.
            return []
        return [{"role": role, "content": cls._response_content(message.get("content"))}]

    @staticmethod
    def _fingerprint(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _continuation_input(self, messages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Build continuation input from stable call ids, never list positions."""
        if self.previous_response_id is None:
            if any(str(m.get("role")) == "tool" for m in messages if isinstance(m, Mapping)):
                raise RuntimeError("Responses transport state is missing; cannot resume tool continuation safely")
            return [item for m in messages if isinstance(m, Mapping)
                    for item in self._message_input(m)]

        users = [m for m in messages if isinstance(m, Mapping) and m.get("role") == "user"]
        state = users[0] if users else {"role": "user", "content": ""}
        result = self._message_input(state)
        outputs = {}
        extras = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "tool":
                call_id = str(message.get("tool_call_id", ""))
                if call_id:
                    outputs[call_id] = message
            elif message.get("role") == "user" and message is not state:
                # User messages emitted for selected multimodal artifacts are
                # content-addressed, so compaction/reordering cannot duplicate
                # or lose them.
                fingerprint = self._fingerprint(message)
                if fingerprint not in self._sent_extra_inputs:
                    extras.append((fingerprint, message))
        missing = [call_id for call_id in self._pending_call_ids if call_id not in outputs]
        if missing:
            raise RuntimeError(
                "Responses continuation has unexecuted function calls: " + ",".join(missing))
        for call_id in self._pending_call_ids:
            result.extend(self._message_input(outputs[call_id]))
            self._sent_call_ids.add(call_id)
        for fingerprint, message in extras:
            result.extend(self._message_input(message))
            self._sent_extra_inputs.add(fingerprint)
        self._last_state_fingerprint = self._fingerprint(state)
        return result

    def transport_state(self) -> dict[str, Any]:
        """Serializable non-secret state required to resume Responses linkage."""
        return {"previous_response_id": self.previous_response_id,
                "pending_call_ids": list(self._pending_call_ids),
                "sent_call_ids": sorted(self._sent_call_ids),
                "sent_extra_inputs": sorted(self._sent_extra_inputs),
                "last_state_fingerprint": self._last_state_fingerprint}

    def restore_transport_state(self, state: Mapping[str, Any] | None) -> None:
        if not isinstance(state, Mapping):
            return
        self.previous_response_id = (str(state["previous_response_id"])
                                     if state.get("previous_response_id") else None)
        self._pending_call_ids = [str(x) for x in state.get("pending_call_ids") or []]
        self._sent_call_ids = {str(x) for x in state.get("sent_call_ids") or []}
        self._sent_extra_inputs = {str(x) for x in state.get("sent_extra_inputs") or []}
        self._last_state_fingerprint = (str(state["last_state_fingerprint"])
                                        if state.get("last_state_fingerprint") else None)

    def _audit(self, response: Any, *, tool_call_count: int,
               previous_response_id_used: str | None = None) -> dict[str, Any]:
        usage = self._value(response, "usage")
        if usage is not None and not isinstance(usage, Mapping):
            dump = getattr(usage, "model_dump", None)
            if callable(dump):
                usage = dump()
            else:
                to_dict = getattr(usage, "to_dict", None)
                usage = to_dict() if callable(to_dict) else {
                    key: self._value(usage, key) for key in
                    ("input_tokens", "output_tokens", "total_tokens",
                     "prompt_tokens", "completion_tokens",
                     "input_tokens_details", "output_tokens_details")
                    if self._value(usage, key) is not None}
        reasoning = self._value(response, "reasoning")
        effective_context = (self._value(reasoning, "context")
                             if reasoning is not None else self._value(response, "reasoning_context"))
        audit = {"provider": self.provider or ("apex" if "apexin.ai" in self.base_url else "openai"),
                 "requested_model": self.model,
                 "effective_model": self._value(response, "model"),
                 "reasoning_effort": self.reasoning_effort,
                 "response_id": self._value(response, "id"),
                 "previous_response_id_used": previous_response_id_used,
                 "usage": usage,
                 "finish_status": self._value(response, "status") or self._value(response, "finish_reason"),
                 "effective_reasoning_context": effective_context,
                 "tool_call_count": tool_call_count}
        self.audit_log.append(audit)
        return audit

    def _decide_once(self, *, messages, tools):
        if not hasattr(self.client, "responses"):
            raise RuntimeError("OpenAI SDK/client does not support the Responses API")
        previous_response_id_used = self.previous_response_id
        input_items = self._continuation_input(list(messages))
        with _total_deadline(self.total_response_timeout):
            kwargs = {"model": self.model, "input": input_items,
                      "tools": [self._response_tool(tool) for tool in tools],
                      "reasoning": {"effort": self.reasoning_effort},
                      "max_output_tokens": self.max_tokens}
            if self.previous_response_id is not None:
                kwargs["previous_response_id"] = self.previous_response_id
            response = self.client.responses.create(**kwargs)
        output = self._value(response, "output", []) or []
        calls = []
        text_parts = []
        output_text = self._value(response, "output_text")
        if output_text:
            text_parts.append(str(output_text))
        for item in output:
            item_type = self._value(item, "type")
            if item_type == "function_call":
                calls.append({"id": self._value(item, "call_id") or self._value(item, "id") or "",
                              "name": self._value(item, "name") or "",
                              "arguments": self._value(item, "arguments") or "{}"})
            elif item_type == "message":
                content = self._value(item, "content", []) or []
                for part in content:
                    if self._value(part, "type") in {"output_text", "text"}:
                        value = self._value(part, "text")
                        if value: text_parts.append(str(value))
        response_id = self._value(response, "id")
        if not response_id:
            raise RuntimeError("Responses API returned no response id; continuation cannot be guaranteed")
        status = self._value(response, "status")
        if status != "completed":
            self._audit(response, tool_call_count=len(calls),
                        previous_response_id_used=previous_response_id_used)
            raise ModelResponseIncomplete(f"Responses API returned non-completed status: {status}")
        self.previous_response_id = str(response_id)
        self._pending_call_ids = [str(call["id"]) for call in calls if call.get("id")]
        audit = self._audit(response, tool_call_count=len(calls),
                            previous_response_id_used=previous_response_id_used)
        return {"content": "".join(text_parts), "tool_calls": calls, "audit": audit}

    def decide(self, *, messages, tools):
        for attempt in range(len(self.retry_delays) + 1):
            try:
                return self._decide_once(messages=messages, tools=tools)
            except Exception as exc:
                if attempt >= len(self.retry_delays) or not self._transient(exc):
                    raise
                time.sleep(max(0.0, float(self.retry_delays[attempt])))
        raise AssertionError("unreachable")

__all__ = ["Model", "OpenAIModel", "ModelResponseTimeout", "ModelResponseIncomplete"]
