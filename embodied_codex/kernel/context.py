"""Progressive context construction; detail is loaded only by model request."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class MinimalSystemPrompt:
    text = (
        "You are an autonomous embodied coding agent. Understand the task and adapter SDK, "
        "write and test a complete controller in the workspace, inspect structured evidence, "
        "and choose your next engineering action. Follow the adapter contract and safety limits. "
        "Use assets by searching their summaries first and request detail only when useful."
    )


@dataclass
class ContextBuilder:
    adapter_index: Mapping[str, Any]
    asset_registry: Any
    workspace: Any
    system_prompt: str = MinimalSystemPrompt.text
    top_k: int = 5

    def build(self, *, task: str, latest_evidence: Any = None,
              retrieved_assets: Any = None) -> dict[str, Any]:
        assets = retrieved_assets
        if assets is None:
            assets = self.asset_registry.search(task, limit=self.top_k) if self.asset_registry else {}
        return {
            "system": self.system_prompt,
            "task": str(task),
            "adapter": dict(self.adapter_index),
            "workspace": self.workspace.index(limit=100) if self.workspace else [],
            "controller": self._controller_summary(),
            "assets": assets,
            "latest_evidence": latest_evidence,
        }

    def _controller_summary(self):
        if not self.workspace or not self.workspace.controller.is_file(): return None
        path = self.workspace.controller
        return {"path": "controller.py", "bytes": path.stat().st_size}

    def load_selected_detail(self, selection: Any):
        if not self.asset_registry: return None
        return self.asset_registry.inspect(selection)


__all__ = ["ContextBuilder", "MinimalSystemPrompt"]
