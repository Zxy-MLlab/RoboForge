"""Minimal autonomous loop. The model chooses actions; the kernel executes them."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping

from .context import ContextBuilder
from .events import EventStore
from .recovery import save_checkpoint


@dataclass
class AgentDecision:
    requests_asset_search: str | None = None
    requests_web_search: str | None = None
    requests_more_detail: Any = None
    changes: Mapping[str, Any] | list[Mapping[str, Any]] | None = None
    executes_controller: bool = False
    acquires_capability: Mapping[str, Any] | None = None
    finishes: bool = False
    message: str = ""

    @classmethod
    def from_value(cls, value: Any) -> "AgentDecision":
        if isinstance(value, cls): return value
        if not isinstance(value, Mapping): raise TypeError("agent step must return a decision object")
        return cls(
            requests_asset_search=value.get("requests_asset_search") or value.get("search_assets"),
            requests_web_search=value.get("requests_web_search") or value.get("search_web"),
            requests_more_detail=value.get("requests_more_detail") or value.get("load_detail"),
            changes=value.get("changes") or value.get("workspace_changes"),
            executes_controller=bool(value.get("executes_controller") or value.get("execute")),
            acquires_capability=value.get("acquires_capability") or value.get("capability"),
            finishes=bool(value.get("finishes") or value.get("finish")),
            message=str(value.get("message") or ""),
        )


@dataclass
class LoopBudget:
    max_steps: int = 60
    max_executions: int = 20
    timeout_seconds: float = 3600
    started: float = field(default_factory=time.monotonic)
    steps: int = 0
    executions: int = 0

    def exhausted(self) -> bool:
        return (self.steps >= self.max_steps or self.executions >= self.max_executions
                or time.monotonic() - self.started >= self.timeout_seconds)


class AgentLoop:
    def __init__(self, *, agent: Any, workspace: Any, adapter: Any,
                 context_builder: ContextBuilder, asset_registry: Any = None,
                 runtime: Any = None, event_store: EventStore | None = None,
                 budget: LoopBudget | None = None, root: str | None = None,
                 web_search: Any = None):
        self.agent, self.workspace, self.adapter = agent, workspace, adapter
        self.context_builder, self.asset_registry = context_builder, asset_registry
        self.runtime = runtime
        self.web_search = web_search
        self.event_store = event_store or EventStore(workspace.root)
        self.budget = budget or LoopBudget()
        self.root = root or str(workspace.root)
        self.latest_evidence = None
        self.retrieved_assets = None

    def _step(self, context):
        if hasattr(self.agent, "step"):
            return self.agent.step(context)
        if hasattr(self.agent, "decide"):
            return self.agent.decide(context=context)
        raise TypeError("agent must expose step(context) or decide(context=...)")

    def run(self, task: str | None = None) -> dict[str, Any]:
        task = str(task or getattr(self.adapter, "instruction", ""))
        while not self.budget.exhausted():
            self.budget.steps += 1
            context = self.context_builder.build(task=task,
                latest_evidence=self.latest_evidence, retrieved_assets=self.retrieved_assets)
            decision = AgentDecision.from_value(self._step(context))
            self.event_store.commit("agent_decision", {"step": self.budget.steps,
                "message": decision.message, "decision": decision.__dict__})
            if decision.requests_asset_search:
                self.retrieved_assets = self.asset_registry.search(decision.requests_asset_search)
            if decision.requests_web_search and self.web_search is not None:
                web_results = self.web_search(decision.requests_web_search)
                current = dict(self.retrieved_assets or {})
                current["web_search"] = web_results
                self.retrieved_assets = current
            if decision.requests_more_detail is not None:
                selection = decision.requests_more_detail
                if isinstance(selection, Mapping):
                    selection = selection.get("asset_id") or selection.get("id") or selection.get("tool_id")
                detail = self.context_builder.load_selected_detail(selection)
                if detail is not None:
                    current = dict(self.retrieved_assets or {})
                    current["detail"] = detail
                    self.retrieved_assets = current
            if decision.changes:
                self.workspace.apply(decision.changes)
            if decision.executes_controller:
                if self.runtime is None: raise RuntimeError("controller runtime is not configured")
                self.budget.executions += 1
                try:
                    result = self.runtime.execute(self.workspace.controller, self.adapter)
                    report = self.adapter.sensor_report(result)
                    self.latest_evidence = {"execution": result, "sensor_report": report}
                    self.event_store.commit("execution", self.latest_evidence)
                except Exception as exc:
                    self.latest_evidence = {"error": f"{type(exc).__name__}: {exc}"}
                    self.event_store.commit("execution_error", self.latest_evidence)
            if decision.acquires_capability:
                saved = self.asset_registry.save(decision.acquires_capability)
                self.event_store.commit("asset_saved", {"asset": saved})
            save_checkpoint(self.root, {"steps": self.budget.steps,
                "executions": self.budget.executions,
                "latest_evidence": self.latest_evidence})
            if decision.finishes: break
        return {"steps": self.budget.steps, "executions": self.budget.executions,
                "budget_exhausted": self.budget.exhausted(), "latest_evidence": self.latest_evidence}


__all__ = ["AgentDecision", "AgentLoop", "LoopBudget"]
