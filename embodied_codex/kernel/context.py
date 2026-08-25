"""Progressive context construction for the function-calling coding agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
                    "point_cloud_path", "video_path") if key in value}
        cameras = value.get("cameras")
        if isinstance(cameras, Mapping):
            summary["cameras"] = {str(name): {key: item.get(key) for key in
                ("rgb_path", "rgb_sha256", "depth_path", "depth_sha256", "shape",
                 "depth_range_m") if key in item}
                for name, item in cameras.items() if isinstance(item, Mapping)}
        return summary

    @staticmethod
    def _evidence_summary(value: Any):
        if not isinstance(value, Mapping):
            return value
        execution = value.get("execution") if isinstance(value.get("execution"), Mapping) else {}
        report = value.get("sensor_report") if isinstance(value.get("sensor_report"), Mapping) else {}
        return {"artifact_uri": value.get("artifact_uri"),
                "controller_sha256": value.get("controller_sha256"),
                "case_handle": value.get("case_handle"),
                "environment_identity": value.get("environment_identity"),
                "verification_receipt": value.get("verification_receipt"),
                "execution": {key: execution.get(key) for key in
                    ("completed", "error", "sensor_verification_observed", "program_sha256")},
                "sensor_report": {key: report.get(key) for key in report
                    if key in {"sensor_success", "sensor_success_candidate", "success", "verified",
                               "sensor_verification_passed", "independent_task_outcome",
                               "trace_path", "rollout_path", "final_step", "proprioception",
                               "final_proprioception", "action_log"}}}

    def _controller_summary(self):
        if not self.workspace or not self.workspace.controller.is_file(): return None
        path = self.workspace.controller
        return {"path": "controller.py", "bytes": path.stat().st_size}

    def load_selected_detail(self, selection: Any):
        return self.asset_registry.inspect(str(selection)) if self.asset_registry else None


__all__ = ["ContextBuilder", "MinimalSystemPrompt"]
