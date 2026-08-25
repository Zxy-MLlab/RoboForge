"""Canonical model-driven agent loop with structured function calling."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from .capability_manager import CapabilityManager
from .context import ContextBuilder
from .events import EventStore
from .recovery import load_checkpoint, save_checkpoint
from .tools import ToolRegistry


@dataclass
class LoopBudget:
    max_steps: int = 60; max_executions: int = 20; timeout_seconds: float = 3600
    started: float = field(default_factory=time.monotonic); steps: int = 0; executions: int = 0
    def exhausted(self):
        return (self.steps >= self.max_steps or self.executions >= self.max_executions
                or time.monotonic() - self.started >= self.timeout_seconds)


class ProtocolError(RuntimeError): pass


class AgentLoop:
    def __init__(self, *, model: Any, workspace: Any, adapter: Any,
                 context_builder: ContextBuilder, capability_manager: CapabilityManager,
                 runtime: Any = None, event_store: EventStore | None = None,
                 budget: LoopBudget | None = None, root: str | Path | None = None,
                 web_search: Any = None, policies: list[Any] | None = None,
                 resume: bool = True):
        self.model, self.workspace, self.adapter = model, workspace, adapter
        self.context_builder, self.capability_manager = context_builder, capability_manager
        self.runtime = runtime; self.event_store = event_store or EventStore(workspace.root)
        self.budget = budget or LoopBudget(); self.root = Path(root or workspace.root).resolve()
        self.web_search = web_search; self.policies = list(policies or [])
        self.latest_evidence = None; self.retrieved_assets = None; self.messages: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {"finished": False, "last_tool_call": None}
        if resume:
            checkpoint = load_checkpoint(self.root)
            if checkpoint:
                self.budget.steps = int(checkpoint.get("steps", 0)); self.budget.executions = int(checkpoint.get("executions", 0))
                self.latest_evidence = checkpoint.get("latest_evidence"); self.state.update(checkpoint.get("state") or {})
                if checkpoint.get("snapshot_id"):
                    self.workspace.restore(checkpoint["snapshot_id"])
                self.retrieved_assets = checkpoint.get("retrieved_assets")
        self.tools = self._build_tools()

    def _schema(self, properties=None, required=()):
        return {"type": "object", "properties": dict(properties or {}),
                "required": list(required), "additionalProperties": False}

    def _build_tools(self):
        registry = ToolRegistry(); ws = self.workspace; cap = self.capability_manager
        obj = {"type": "object"}; string = {"type": "string"}; integer = {"type": "integer"}
        registry.add("list_files", "List files in the persistent workspace.", self._schema({"pattern": string}),
                     lambda pattern="**/*": ws.list_files(pattern))
        registry.add("read_file", "Read a bounded line range from a workspace file.", self._schema(
            {"path": string, "start_line": integer, "end_line": integer}, ["path"]),
            lambda path, start_line=1, end_line=400: ws.read_file(path, start_line, end_line))
        registry.add("write_file", "Atomically write one workspace file.", self._schema(
            {"path": string, "content": string}, ["path", "content"]), ws.write_file)
        registry.add("replace_file_lines", "Atomically replace an inspected line range.", self._schema(
            {"path": string, "start_line": integer, "end_line": integer, "new_content": string,
             "expected_old_sha256": string}, ["path", "start_line", "end_line", "new_content"]), ws.replace_file_lines)
        registry.add("run_command", "Run a bounded engineering/test command in the workspace.", self._schema(
            {"argv": {"type": "array", "items": string}, "timeout_seconds": {"type": "number"}}, ["argv"]), ws.run_command)
        registry.add("search_assets", "Search shared Tool, Skill, Experience and Gap summaries.", self._schema(
            {"query": string, "limit": integer}, ["query"]), cap.search)
        registry.add("inspect_asset", "Load selected asset manual/contract detail.", self._schema({"asset_id": string}, ["asset_id"]), cap.inspect)
        registry.add("load_tool_source", "Explicitly load a Tool implementation after manual inspection.", self._schema({"tool_id": string}, ["tool_id"]), cap.load_tool_source)
        registry.add("search_web", "Search public web sources for a capability.", self._schema({"query": string, "limit": integer}, ["query"]), cap.web_search)
        registry.add("fetch_web_page", "Open one HTTPS public page.", self._schema({"url": string, "max_chars": integer}, ["url"]), cap.fetch_page)
        registry.add("download_public_asset", "Download one HTTPS asset into the workspace with optional SHA256.", self._schema(
            {"url": string, "filename": string, "sha256": string}, ["url", "filename"]), cap.download)
        registry.add("unpack_public_asset", "Safely unpack a downloaded archive inside the workspace.", self._schema(
            {"path": string, "destination": string}, ["path", "destination"]), cap.unpack)
        open_object = {"type": "object", "additionalProperties": True}
        registry.add("register_tool", "Register an immutable Tool version; call test_tool before it can be bound.", open_object, cap.register_tool)
        registry.add("test_tool", "Run JSON contract tests against a registered Tool.", self._schema(
            {"tool_id": string, "cases": {"type": "array"}}, ["tool_id", "cases"]), cap.test_tool)
        registry.add("register_skill", "Persist a successful reusable Skill.", open_object, cap.register_skill)
        registry.add("register_experience", "Persist an evidence-backed Experience.", open_object, cap.register_experience)
        registry.add("record_gap", "Persist an unresolved capability Gap.", open_object, cap.record_gap)
        registry.add("run_controller", "Execute the current controller once and return sensor evidence.", self._schema(), self._run_controller)
        registry.add("inspect_execution", "Inspect the latest committed execution evidence.", self._schema(), lambda: self.latest_evidence or {})
        registry.add("view_sensor_artifact", "Read a bounded sensor/evidence artifact path.", self._schema({"path": string, "max_chars": integer}, ["path"]), self._view_artifact)
        registry.add("finish", "Finish only after the task is actually verified.", self._schema({"summary": string}, ["summary"]), self._finish)
        return registry

    def _view_artifact(self, path: str, max_chars: int = 12000):
        candidate = Path(path)
        if not candidate.is_absolute(): candidate = self.workspace.root / candidate
        candidate = candidate.resolve()
        if self.workspace.root not in candidate.parents or not candidate.is_file(): raise ProtocolError("artifact outside workspace")
        return {"path": str(candidate.relative_to(self.workspace.root)), "content": candidate.read_text(errors="replace")[:max_chars]}

    def _finish(self, summary: str): self.state.update({"finished": True, "summary": summary}); return self.state

    def _run_controller(self):
        if self.runtime is None: raise RuntimeError("controller runtime is not configured")
        controller_sha = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest()
        case_handle = getattr(getattr(self.adapter, "episode", None), "case_handle", None)
        execution_key = hashlib.sha256(f"{controller_sha}:{case_handle or 'single-case'}".encode()).hexdigest()
        for row in self.event_store.events():
            if row.get("kind") == "execution" and row.get("payload", {}).get("execution_key") == execution_key:
                return {"reused_committed_execution": True, **row["payload"]}
        self.budget.executions += 1
        result = self.runtime.execute(self.workspace.controller, self.adapter)
        report = self.adapter.sensor_report(result)
        evidence = {"execution": result, "sensor_report": report, "controller_sha256": controller_sha,
                    "execution_key": execution_key, "case_handle": case_handle}
        self.latest_evidence = evidence; self.event_store.commit("execution", evidence)
        return evidence

    def _messages(self, task: str):
        context = self.context_builder.build(task=task, latest_evidence=self.latest_evidence,
                                             retrieved_assets=self.retrieved_assets, state=self.state)
        if not self.messages:
            self.messages = [{"role": "system", "content": context["system"]},
                             {"role": "user", "content": json.dumps(context, default=str)}]
        else:
            self.messages.append({"role": "user", "content": json.dumps(context, default=str)})
        return self.messages

    def run(self, task: str | None = None):
        task = str(task or getattr(self.adapter, "instruction", ""))
        for policy in self.policies:
            before = getattr(policy, "before_run", None)
            if callable(before): before(self)
        while not self.budget.exhausted() and not self.state.get("finished"):
            self.budget.steps += 1
            messages = self._messages(task)
            response = self.model.decide(messages=messages, tools=self.tools.schemas)
            calls = response.get("tool_calls") if isinstance(response, Mapping) else None
            if not isinstance(calls, list):
                self.event_store.commit("protocol_error", {"error": "model response must contain tool_calls", "response": response})
                self.messages.append({"role": "user", "content": "Protocol error: return a valid function call using the provided tools."})
                self._checkpoint(); continue
            assistant = {"role": "assistant", "content": response.get("content", ""), "tool_calls": calls}
            self.messages.append(assistant)
            for call in calls:
                try:
                    name = str(call.get("name") or call.get("function", {}).get("name") or "")
                    raw = call.get("arguments") or call.get("function", {}).get("arguments") or "{}"
                    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    result = self.tools.invoke(name, arguments)
                    if name == "search_assets": self.retrieved_assets = result
                    payload = {"ok": True, "result": result}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                self.messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                      "content": json.dumps(payload, default=str)})
                self.event_store.commit("tool_result", {"name": name, "payload": payload})
            self._checkpoint()
        result = {"steps": self.budget.steps, "executions": self.budget.executions,
                  "budget_exhausted": self.budget.exhausted(), "finished": self.state.get("finished", False),
                  "latest_evidence": self.latest_evidence}
        for policy in self.policies:
            after = getattr(policy, "after_run", None)
            if callable(after): after(self, result)
        return result

    def _checkpoint(self):
        snapshot = self.workspace.snapshot()
        save_checkpoint(self.root, {"steps": self.budget.steps, "executions": self.budget.executions,
            "latest_evidence": self.latest_evidence, "snapshot_id": snapshot.snapshot_id,
            "retrieved_assets": self.retrieved_assets, "state": self.state})


__all__ = ["AgentLoop", "LoopBudget", "ProtocolError"]
