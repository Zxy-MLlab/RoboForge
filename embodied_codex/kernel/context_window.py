"""Bounded context history for long-running coding-agent sessions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping


@dataclass
class ContextWindowManager:
    max_tokens: int = 30_000
    max_tool_result_chars: int = 24_000
    chars_per_token: float = 4.0

    @staticmethod
    def _encoded(value: Any) -> str:
        text = json.dumps(value, default=str)
        return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image>", text)

    def token_estimate(self, messages: list[Mapping[str, Any]], tools=None) -> int:
        characters = len(self._encoded(messages)) + len(self._encoded(tools or []))
        return int(characters / max(self.chars_per_token, 1.0)) + 1

    @property
    def max_message_chars(self) -> int:
        return int(self.max_tokens * self.chars_per_token)

    def bound_tool_payload(self, payload: Mapping[str, Any]) -> str:
        content = json.dumps(payload, default=str)
        if len(content) <= self.max_tool_result_chars:
            return content
        return json.dumps({"ok": payload.get("ok"), "truncated": True,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "preview": content[:self.max_tool_result_chars]})

    def compact(self, messages: list[dict[str, Any]], tools=None) -> list[dict[str, Any]]:
        if self.token_estimate(messages, tools) <= self.max_tokens:
            return messages
        fixed = messages[:2]
        groups: list[list[dict[str, Any]]] = []
        for message in messages[2:]:
            if message.get("role") == "assistant" or not groups:
                groups.append([message])
            else:
                groups[-1].append(message)
        selected: list[list[dict[str, Any]]] = []
        omitted = 0
        for group in reversed(groups):
            flattened = [item for row in selected for item in row]
            if self.token_estimate([*fixed, *group, *flattened], tools) > self.max_tokens:
                omitted += 1
                continue
            selected.insert(0, group)
        flattened = [item for row in selected for item in row]
        if omitted:
            summary = {"role": "user", "content": json.dumps({
                "history_compacted": True, "omitted_tool_call_groups": omitted,
                "current_state_is_authoritative": True})}
            candidate = [*fixed, summary, *flattened]
            if self.token_estimate(candidate, tools) <= self.max_tokens:
                return candidate
        return [*fixed, *flattened]


__all__ = ["ContextWindowManager"]
