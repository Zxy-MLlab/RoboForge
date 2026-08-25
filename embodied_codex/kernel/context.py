"""Progressive context construction for the function-calling coding agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class MinimalSystemPrompt:
    text = (
        "You are an autonomous embodied coding agent. Work directly in the persistent workspace. "
        "Understand the task and Adapter SDK, write and test a complete controller, inspect "
        "structured sensor evidence, and choose the next action. Use indexed assets first; load "
        "manuals or source only when needed. Never claim completion without executing and checking evidence."
    )


@dataclass
class ContextBuilder:
    adapter_index: Mapping[str, Any]
    asset_registry: Any
    workspace: Any
    system_prompt: str = MinimalSystemPrompt.text
    top_k: int = 5

    def build(self, *, task: str, latest_evidence: Any = None,
              retrieved_assets: Any = None, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        assets = retrieved_assets
        if assets is None and self.asset_registry:
            assets = self.asset_registry.search(task, limit=self.top_k)
        return {"system": self.system_prompt, "task": str(task), "adapter": dict(self.adapter_index),
                "workspace": self.workspace.index(limit=100) if self.workspace else [],
                "controller": self._controller_summary(), "assets": assets or {},
                "latest_evidence": latest_evidence, "state": dict(state or {})}

    def _controller_summary(self):
        if not self.workspace or not self.workspace.controller.is_file(): return None
        path = self.workspace.controller
        return {"path": "controller.py", "bytes": path.stat().st_size}

    def load_selected_detail(self, selection: Any):
        return self.asset_registry.inspect(str(selection)) if self.asset_registry else None


__all__ = ["ContextBuilder", "MinimalSystemPrompt"]
