"""Direct OpenAI-compatible model client; no external Harness dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class Model(Protocol):
    def decide(
        self, *, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...


@dataclass
class OpenAICompatibleModel:
    api_key: str
    base_url: str
    model: str = "gpt-5.6-sol"
    timeout: float = 300
    # Retries are owned by Agent so they are visible in its durable trace and
    # do not turn one opaque SDK call into many five-minute waits.
    max_retries: int = 0
    max_tokens: int = 6000
    reasoning_effort: str = "low"
    stream: bool = True

    def __post_init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def decide(self, *, messages, tools):
        response = self._client.chat.completions.create(
            model=self.model, messages=list(messages), tools=list(tools),
            tool_choice="auto", temperature=0, max_tokens=self.max_tokens,
            extra_body={"reasoning_effort": self.reasoning_effort},
            stream=self.stream,
        )
        if self.stream:
            return self._collect_stream(response)
        choice = response.choices[0].message
        return {
            "content": choice.content or "",
            "tool_calls": [
                {"id": item.id, "name": item.function.name,
                 "arguments": item.function.arguments}
                for item in (choice.tool_calls or [])
            ],
        }

    @staticmethod
    def _collect_stream(chunks) -> Mapping[str, Any]:
        """Reassemble OpenAI tool calls while consuming the response stream."""
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        for chunk in chunks:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            if getattr(delta, "content", None):
                content.append(delta.content)
            for item in (getattr(delta, "tool_calls", None) or []):
                index = int(item.index)
                call = calls.setdefault(index, {
                    "id": "", "name": "", "arguments": "",
                })
                if getattr(item, "id", None):
                    call["id"] += item.id
                function = getattr(item, "function", None)
                if function is not None:
                    if getattr(function, "name", None):
                        call["name"] += function.name
                    if getattr(function, "arguments", None):
                        call["arguments"] += function.arguments
        return {
            "content": "".join(content),
            "tool_calls": [calls[index] for index in sorted(calls)],
        }


__all__ = ["Model", "OpenAICompatibleModel"]
