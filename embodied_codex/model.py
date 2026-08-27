from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import signal
import threading
import time
from typing import Any, Mapping, Protocol


class Model(Protocol):
    def decide(self, *, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]]): ...


class ModelResponseTimeout(TimeoutError):
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
        self._sent_message_count = 0
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
            # them in a new input would duplicate calls, so only text is sent.
            content = message.get("content")
            return ([{"role": "assistant", "content": cls._response_content(content)}]
                    if content else [])
        return [{"role": role, "content": cls._response_content(message.get("content"))}]

    def _incremental_input(self, messages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if self.previous_response_id is None or self._sent_message_count == 0:
            selected = list(messages)
        else:
            # The Kernel replaces the current state message while appending the
            # new assistant/tool transcript.  Re-send that state plus only the
            # transcript emitted since the previous response.
            selected = ([messages[1]] if len(messages) > 1 else [])
            selected.extend(messages[self._sent_message_count:])
        result = []
        for message in selected:
            if isinstance(message, Mapping):
                result.extend(self._message_input(message))
        return result

    def _audit(self, response: Any, *, tool_call_count: int) -> dict[str, Any]:
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
        audit = {"provider": self.provider or ("apex" if "apexin.ai" in self.base_url else "openai"),
                 "requested_model": self.model,
                 "effective_model": self._value(response, "model"),
                 "reasoning_effort": self.reasoning_effort,
                 "response_id": self._value(response, "id"),
                 "usage": usage,
                 "finish_status": self._value(response, "status") or self._value(response, "finish_reason"),
                 "tool_call_count": tool_call_count}
        self.audit_log.append(audit)
        return audit

    def _decide_once(self, *, messages, tools):
        if not hasattr(self.client, "responses"):
            raise RuntimeError("OpenAI SDK/client does not support the Responses API")
        input_items = self._incremental_input(list(messages))
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
        self.previous_response_id = str(response_id)
        self._sent_message_count = len(messages)
        audit = self._audit(response, tool_call_count=len(calls))
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

__all__ = ["Model", "OpenAIModel", "ModelResponseTimeout"]
