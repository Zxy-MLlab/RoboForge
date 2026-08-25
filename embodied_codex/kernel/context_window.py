"""Bounded context history for long-running coding-agent sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import os
import uuid
from typing import Any, Mapping


@dataclass(frozen=True)
class ResourceBudgets:
    max_task_chars: int = 12_000
    max_adapter_chars: int = 24_000
    max_workspace_chars: int = 20_000
    max_assets_chars: int = 24_000
    max_state_chars: int = 16_000
    max_evidence_chars: int = 20_000
    max_context_chars: int = 100_000
    max_tool_calls_per_turn: int = 4
    max_image_bytes: int = 4 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    max_total_image_pixels: int = 24_000_000
    max_images_per_turn: int = 4
    max_video_bytes: int = 512 * 1024 * 1024
    max_video_frames: int = 4
    max_array_bytes: int = 512 * 1024 * 1024
    max_array_elements: int = 64_000_000
    max_point_cloud_header_bytes: int = 64 * 1024
    max_process_output_bytes: int = 1024 * 1024
    max_artifact_bytes: int = 2 * 1024 * 1024 * 1024


@dataclass
class ContextWindowManager:
    max_tokens: int = 30_000
    max_tool_result_chars: int = 24_000
    chars_per_token: float = 4.0
    budgets: ResourceBudgets = field(default_factory=ResourceBudgets)

    @staticmethod
    def _encoded(value: Any) -> str:
        text = json.dumps(value, default=str)
        return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image>", text)

    def token_estimate(self, messages: list[Mapping[str, Any]], tools=None) -> int:
        characters = len(self._encoded(messages)) + len(self._encoded(tools or []))
        images = self._image_count(messages)
        return int(characters / max(self.chars_per_token, 1.0)) + 1 + images * 2048

    @classmethod
    def _image_count(cls, value: Any) -> int:
        if isinstance(value, Mapping):
            return (1 if value.get("type") == "image_url" else 0) + sum(
                cls._image_count(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(cls._image_count(item) for item in value)
        return 0

    @property
    def max_message_chars(self) -> int:
        return int(self.max_tokens * self.chars_per_token)

    def bound_tool_payload(self, payload: Mapping[str, Any]) -> str:
        content = json.dumps(payload, default=str, sort_keys=True)
        if len(content) <= self.max_tool_result_chars:
            return content
        return json.dumps({"ok": payload.get("ok"), "truncated": True,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "preview": content[:self.max_tool_result_chars]})

    def _write_context_artifact(self, root: Path, encoded: str) -> str:
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        directory = root / "context"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest}.json"
        if not target.exists():
            current = sum(path.stat().st_size for path in root.rglob("*")
                          if path.is_file()) if root.is_dir() else 0
            required = len(encoded.encode()) + 1
            if current + required > self.budgets.max_artifact_bytes:
                raise RuntimeError("run artifact disk quota exceeded")
            temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
            try:
                with temporary.open("x") as stream:
                    stream.write(encoded); stream.write("\n")
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return f"run://artifacts/context/{target.name}"

    def _bound_field(self, value: Any, limit: int, artifact_root: Path) -> Any:
        encoded = json.dumps(value, default=str, sort_keys=True)
        if len(encoded) <= limit:
            return value
        return {"truncated": True, "original_chars": len(encoded),
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                "preview": encoded[:max(0, limit // 2)],
                "artifact_uri": self._write_context_artifact(artifact_root, encoded)}

    def bound_context(self, context: Mapping[str, Any], *,
                      artifact_root: str | Path) -> dict[str, Any]:
        root = Path(artifact_root).resolve()
        result = dict(context)
        limits = {"task": self.budgets.max_task_chars,
                  "adapter": self.budgets.max_adapter_chars,
                  "workspace": self.budgets.max_workspace_chars,
                  "assets": self.budgets.max_assets_chars,
                  "latest_evidence": self.budgets.max_evidence_chars,
                  "state": self.budgets.max_state_chars}
        for key, limit in limits.items():
            if key in result:
                result[key] = self._bound_field(result[key], int(limit), root)
        encoded = json.dumps(result, default=str)
        if len(encoded) > self.budgets.max_context_chars:
            # Preserve routing fields and replace the complete fixed state with
            # one recoverable reference. This handles a single current state
            # that exceeds the budget even when history is empty.
            reference = self._bound_field(dict(context), 512, root)
            result = {"system": result.get("system"), "current_context": reference}
        return result

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


__all__ = ["ContextWindowManager", "ResourceBudgets"]
