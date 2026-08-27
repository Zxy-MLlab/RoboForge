"""Progressive context construction for the function-calling coding agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evidence import AgentEvidence, is_routing_reference


_PRIVATE_DIGEST_KEYS = {"reward", "done", "env.check_success", "env_check_success",
                       "check_success", "hidden_evaluator", "hidden evaluator", "evaluator"}


class MinimalSystemPrompt:
    text = (
        "You are an autonomous embodied coding agent. Work directly in the persistent workspace. "
        "Understand the task and Adapter SDK, write and test a complete controller, inspect "
        "structured sensor evidence, and choose the next action. Use indexed assets first; load "
        "manuals or source only when needed. Stay within the sandbox and Adapter safety contract. "
        "Never claim completion without executing and checking evidence."
    )


@dataclass
class ContextBuilder:
    adapter_index: Mapping[str, Any]
    asset_registry: Any
    workspace: Any
    initial_observation: Any = None
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
                "initial_observation": self._observation_summary(self.initial_observation),
                "latest_evidence": self._evidence_summary(latest_evidence), "state": dict(state or {})}

    @staticmethod
    def _observation_summary(value: Any):
        if not isinstance(value, Mapping):
            return value
        summary = {key: value.get(key) for key in
                   ("frame_id", "step", "proprioception", "rgb_path", "depth_path",
                    "point_cloud_path", "video_path") if key in value
                   and (not str(value.get(key)).startswith("/")
                        and not str(value.get(key)).startswith("\\"))}
        cameras = value.get("cameras")
        if isinstance(cameras, Mapping):
            summary["cameras"] = {str(name): {key: item.get(key) for key in
                ("rgb_path", "rgb_sha256", "depth_path", "depth_sha256", "shape",
                 "depth_range_m") if key in item and
                (not key.endswith("_path") or not str(item.get(key)).startswith(("/", "\\")))}
                for name, item in cameras.items() if isinstance(item, Mapping)}
        return summary

    @staticmethod
    def _bounded_diagnostic(value: Any, *, depth: int = 0):
        if depth >= 3:
            return "<nested diagnostic omitted>"
        if isinstance(value, str):
            return value if len(value) <= 512 else value[:512] + "..."
        if isinstance(value, Mapping):
            return {str(key): ContextBuilder._bounded_diagnostic(item, depth=depth + 1)
                    for key, item in list(value.items())[:32]}
        if isinstance(value, (list, tuple)):
            return {"entries": len(value), "preview": [
                ContextBuilder._bounded_diagnostic(item, depth=depth + 1)
                for item in list(value)[:4]]}
        return value

    @staticmethod
    def _evidence_summary(value: Any):
        if isinstance(value, AgentEvidence):
            value = value.as_dict()
        if not isinstance(value, Mapping):
            return value
        execution = value.get("execution") if isinstance(value.get("execution"), Mapping) else {}
        diagnostics = (value.get("diagnostics")
                       if isinstance(value.get("diagnostics"), Mapping) else {})
        digest = (value.get("digest")
                  if isinstance(value.get("digest"), Mapping) else {})
        # This is a positive projection of the AgentEvidence API. Harness and
        # evaluator metadata are not filtered by naming convention because
        # they are never accepted into this view in the first place.
        result = {"execution": {key: execution.get(key) for key in ("completed", "error")},
                  "diagnostics": ContextBuilder._bounded_diagnostic(diagnostics),
                  "digest": ContextBuilder._bounded_digest(digest)}
        if isinstance(value.get("evidence_ref"), str):
            result["evidence_ref"] = value["evidence_ref"]
        if isinstance(value.get("decision_id"), str):
            result["decision_id"] = value["decision_id"]
        return result

    @staticmethod
    def _bounded_digest(value: Any, *, depth: int = 0):
        """Preserve compact digest records while bounding nested public data."""
        if isinstance(value, str):
            if is_routing_reference(value):
                return value
            return value if len(value) <= 512 else value[:512] + "..."
        if not isinstance(value, (Mapping, list, tuple)):
            return value
        if depth >= 6:
            return "<nested digest omitted>"
        if isinstance(value, Mapping):
            items = ((key, item) for key, item in value.items()
                     if str(key).lower() not in _PRIVATE_DIGEST_KEYS)
            return {str(key): ContextBuilder._bounded_digest(item, depth=depth + 1)
                    for key, item in list(items)[:24]}
        if isinstance(value, (list, tuple)):
            entries = list(value)
            if len(entries) <= 16:
                return [ContextBuilder._bounded_digest(item, depth=depth + 1)
                        for item in entries]
            head_count = 4
            tail_count = 12
            return {
                "total_count": len(entries),
                "head": [ContextBuilder._bounded_digest(item, depth=depth + 1)
                         for item in entries[:head_count]],
                "tail": [ContextBuilder._bounded_digest(item, depth=depth + 1)
                         for item in entries[-tail_count:]],
                "omitted_count": len(entries) - head_count - tail_count,
            }
        return value

    def _controller_summary(self):
        if not self.workspace or not self.workspace.controller.is_file(): return None
        path = self.workspace.controller
        return {"path": "controller.py", "bytes": path.stat().st_size}

    def load_selected_detail(self, selection: Any):
        return self.asset_registry.inspect(str(selection)) if self.asset_registry else None


__all__ = ["ContextBuilder", "MinimalSystemPrompt"]
