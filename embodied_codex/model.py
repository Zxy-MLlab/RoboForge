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
    def __post_init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=self.timeout, max_retries=0)
    @staticmethod
    def _transient(error: BaseException) -> bool:
        status = getattr(error, "status_code", None)
        if status in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        name = type(error).__name__
        return name in {"APIConnectionError", "APITimeoutError", "RateLimitError",
                        "InternalServerError", "TimeoutError", "ConnectionError"}

    def _decide_once(self, *, messages, tools):
        with _total_deadline(self.total_response_timeout):
            stream = self.client.chat.completions.create(
                model=self.model, messages=list(messages), tools=list(tools),
                tool_choice="auto", temperature=0, max_tokens=self.max_tokens, stream=True,
                extra_body={"reasoning_effort": self.reasoning_effort},
            )
            content=[]; calls={}
            for chunk in stream:
                if not chunk.choices: continue
                delta=chunk.choices[0].delta
                if delta.content: content.append(delta.content)
                for item in (delta.tool_calls or []):
                    call=calls.setdefault(item.index,{"id":"","name":"","arguments":""})
                    if item.id: call["id"] += item.id
                    if item.function:
                        if item.function.name: call["name"] += item.function.name
                        if item.function.arguments: call["arguments"] += item.function.arguments
        return {"content":"".join(content),"tool_calls":[calls[i] for i in sorted(calls)]}

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
