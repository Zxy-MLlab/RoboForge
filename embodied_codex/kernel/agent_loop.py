"""Canonical model-driven agent loop with structured function calling."""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hashlib
import json
import mimetypes
import re
from pathlib import Path
import time
from typing import Any, Mapping

from .capability_manager import CapabilityManager
from .context import ContextBuilder
from .context_window import ContextWindowManager
from .evidence import AgentEvidence, HarnessMetadata
from .events import EventStore
from .recovery import load_checkpoint, save_checkpoint
from .tools import ToolRegistry


@dataclass
class LoopBudget:
    max_steps: int = 60; max_executions: int = 20; timeout_seconds: float = 3600
    started: float = field(default_factory=time.monotonic); steps: int = 0; executions: int = 0
    elapsed_before: float = 0.0
    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started
    def exhausted(self):
        return (self.steps >= self.max_steps or self.executions >= self.max_executions
                or self.elapsed() >= self.timeout_seconds)


class ProtocolError(RuntimeError): pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AgentLoop:
    def __init__(self, *, model: Any, workspace: Any, adapter: Any,
                 context_builder: ContextBuilder, capability_manager: CapabilityManager,
                 runtime: Any = None, event_store: EventStore | None = None,
                 budget: LoopBudget | None = None, root: str | Path | None = None,
                 web_search: Any = None,
                 resume: bool = True,
                 context_window: ContextWindowManager | None = None,
                 max_evidence_bytes: int = 20 * 1024 * 1024 * 1024):
        self.model, self.workspace, self.adapter = model, workspace, adapter
        self.context_builder, self.capability_manager = context_builder, capability_manager
        self.runtime = runtime; self.root = Path(root or workspace.root.parent).resolve()
        self.event_store = event_store or EventStore(self.root / "events", protect=True)
        self.budget = budget or LoopBudget()
        self.web_search = web_search
        self.latest_evidence = None; self.retrieved_assets = None; self.messages: list[dict[str, Any]] = []
        self._agent_latest_evidence: dict[str, Any] | None = None
        self.context_window = context_window or ContextWindowManager()
        self.max_context_chars = self.context_window.max_message_chars
        self.max_tool_result_chars = self.context_window.max_tool_result_chars
        self.max_evidence_bytes = int(max_evidence_bytes)
        self._evidence_bytes = sum(path.stat().st_size for path in
            (self.root / "evidence").glob("*.json")) if (self.root / "evidence").is_dir() else 0
        self.checkpoint_task = None
        self.session_index = 1
        self.cumulative_steps = 0
        self.cumulative_executions = 0
        self.cumulative_elapsed = 0.0
        self.research_state: dict[str, Any] = {"summary": "", "attempts": []}
        self.state: dict[str, Any] = {"finished": False, "last_tool_call": None,
                                      "completion_valid": False, "successful_cases": 0}
        self._active_tool_call_id: str | None = None
        self._artifact_handles: dict[str, Path] = {}
        self._recovery_mode = False
        restored_tool_groups: list[str] = []
        restored_tool_bindings: list[str] = []
        restored_case: str | None = None
        if resume:
            checkpoint = load_checkpoint(self.root)
            if checkpoint:
                cumulative = dict(checkpoint.get("cumulative") or {})
                self.cumulative_steps = int(cumulative.get("steps", checkpoint.get("steps", 0)))
                self.cumulative_executions = int(cumulative.get("executions", checkpoint.get("executions", 0)))
                self.cumulative_elapsed = float(cumulative.get("elapsed_seconds",
                    checkpoint.get("elapsed_seconds", 0.0)))
                self.session_index = int((checkpoint.get("session") or {}).get("index", 1)) + 1
                self.checkpoint_task = checkpoint.get("task")
                self.latest_evidence = self._load_evidence_reference(
                    checkpoint.get("latest_evidence"))
                self.state.update(checkpoint.get("state") or {})
                self._recovery_mode = (isinstance(self.state.get("completed_execution"), Mapping)
                                       or isinstance(self.state.get("pending_execution"), Mapping))
                self.research_state = self._bound_research_state(
                    checkpoint.get("research_state") or self.research_state)
                if checkpoint.get("snapshot_id"):
                    self.workspace.restore(checkpoint["snapshot_id"])
                self.retrieved_assets = checkpoint.get("retrieved_assets")
                restored_tool_groups = [str(item) for item in
                                        checkpoint.get("active_tool_groups") or []]
                restored_tool_bindings = [str(item) for item in
                                          checkpoint.get("active_shared_tools") or []]
                restored_case = (str(checkpoint["selected_case"])
                                 if checkpoint.get("selected_case") is not None else None)
        if restored_case is not None:
            selector = getattr(self.adapter, "select", None)
            if callable(selector):
                selector(restored_case)
        self.tools = self._build_tools()
        for group in restored_tool_groups:
            if group != "core":
                self.tools.activate(group)
        for tool_id in restored_tool_bindings:
            self.capability_manager.restore_tool_binding(tool_id)
        if self.latest_evidence is not None:
            self._agent_latest_evidence = self._agent_evidence(self.latest_evidence)
            protocol = self._resume_protocol()
            identity = self._execution_identity()
            validator = getattr(self.adapter, "validate_execution_receipt", None)
            valid = bool(protocol and self.latest_evidence.get("resume_token") == protocol.get("resume_token")
                         and self.latest_evidence.get("environment_identity") == identity
                         and callable(validator)
                         and validator(self.latest_evidence.get("verification_receipt") or {}) is True)
            self.state["restored_evidence_unverified"] = not valid
            if not valid:
                self.state["finished"] = False
                self.state["completion_valid"] = False
        initial = getattr(self.adapter, "initial_observation", None)
        if callable(initial) and getattr(self.context_builder, "initial_observation", None) is None:
            self.context_builder.initial_observation = initial()
        self.context_builder.initial_observation = self._register_artifacts(
            self.context_builder.initial_observation)

    def _schema(self, properties=None, required=()):
        return {"type": "object", "properties": dict(properties or {}),
                "required": list(required), "additionalProperties": False}

    def _register_artifacts(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): self._register_artifacts(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._register_artifacts(item) for item in value]
        if not isinstance(value, str):
            return value
        candidate = None
        if value.startswith("artifact://adapter/"):
            candidate = Path(getattr(self.adapter, "artifact_dir", "")) / value.removeprefix("artifact://adapter/")
        elif Path(value).is_absolute():
            candidate = Path(value)
        if candidate is None:
            return value
        try:
            candidate = candidate.resolve()
            if not candidate.is_file():
                return value
            allowed_roots = [self.workspace.root.resolve(), self.root.resolve()]
            adapter_root = getattr(self.adapter, "artifact_dir", None)
            if adapter_root:
                allowed_roots.append(Path(adapter_root).resolve())
            if not any(candidate == root or root in candidate.parents for root in allowed_roots):
                return value
            handle = "artifact://agent/" + hashlib.sha256(str(candidate).encode()).hexdigest()[:24] + "/" + candidate.name
            self._artifact_handles[handle] = candidate
            self._artifact_handles[value] = candidate
            return handle
        except OSError:
            return value

    def _artifact_path(self, uri: str) -> Path:
        if not str(uri).startswith("run://"):
            raise ProtocolError("evidence artifact URI must use run://")
        relative = Path(str(uri).removeprefix("run://"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolError("evidence artifact URI is invalid")
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise ProtocolError("evidence artifact escapes the run root")
        return path

    def _load_evidence_reference(self, value: Any) -> Any:
        if not isinstance(value, Mapping) or not value.get("artifact_uri"):
            return value
        path = self._artifact_path(str(value["artifact_uri"]))
        if not path.is_file() or _file_sha256(path) != value.get("artifact_sha256"):
            raise ProtocolError("checkpoint evidence artifact is missing or corrupt")
        try:
            evidence = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError("checkpoint evidence artifact cannot be decoded") from exc
        return {**evidence, "artifact_uri": value["artifact_uri"],
                "artifact_sha256": value["artifact_sha256"]}

    def _evidence_reference(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        summary = self._agent_evidence(evidence)
        encoded_summary = json.dumps(summary, sort_keys=True, default=str)
        if len(encoded_summary) > self.context_window.budgets.max_evidence_chars:
            summary = {"truncated": True,
                       "original_chars": len(encoded_summary),
                       "sha256": hashlib.sha256(encoded_summary.encode()).hexdigest(),
                       "preview": encoded_summary[:8000]}
        return HarnessMetadata.from_evidence(evidence).as_reference(summary=summary)

    def _agent_evidence(self, evidence: Mapping[str, Any]) -> dict[str, Any]:
        value = evidence.get("agent_evidence")
        if isinstance(value, Mapping):
            return self.context_builder._evidence_summary(value)
        execution = evidence.get("execution") if isinstance(evidence.get("execution"), Mapping) else {}
        # Legacy evidence has no explicit public projection. Expose execution
        # status only; never infer public diagnostics from internal reports.
        return AgentEvidence.from_execution(execution).as_dict()

    @staticmethod
    def _bound_research_state(value: Any) -> dict[str, Any]:
        state = dict(value) if isinstance(value, Mapping) else {}
        attempts = list(state.get("attempts") or [])[-32:]
        summary = str(state.get("summary") or "")[:8000]
        return {"summary": summary, "attempts": attempts}

    def _record_attempt(self, name: str, arguments: Mapping[str, Any], ok: bool) -> None:
        safe_arguments = {key: value for key, value in dict(arguments).items()
                          if key in {"query", "path", "directory", "tool_id",
                                     "asset_id", "name"}}
        attempts = list(self.research_state.get("attempts") or [])
        attempts.append({"session": self.session_index, "step": self.budget.steps,
                         "tool": str(name), "ok": bool(ok),
                         "arguments": safe_arguments})
        self.research_state = {"summary": str(self.research_state.get("summary") or "")[:8000],
                               "attempts": attempts[-32:]}

    def _build_tools(self):
        registry = ToolRegistry(); ws = self.workspace; cap = self.capability_manager
        registry.declare_group("source_inspection",
            "Selective Tool implementation inspection after reading its manual and schema.")
        registry.declare_group("web_acquisition",
            "Public web search, verified download, safe unpack, and isolated build tools.")
        registry.declare_group("asset_authoring",
            "Register, test, revise, promote, and persist Tool/Skill/Experience/Gap assets.")
        string = {"type": "string", "minLength": 1}; integer = {"type": "integer"}
        schema_document = {"type": "object", "properties": {
            "type": {"type": "string"}, "properties": {"type": "object"},
            "required": {"type": "array", "items": string},
            "additionalProperties": {"oneOf": [{"type": "boolean"}, {"type": "object"}]},
            "items": {"oneOf": [{"type": "object"}, {"type": "array"}]},
            "description": {"type": "string"}, "enum": {"type": "array"},
            "oneOf": {"type": "array", "items": {"type": "object"}},
            "anyOf": {"type": "array", "items": {"type": "object"}},
            "minimum": {"type": "number"}, "maximum": {"type": "number"},
            "minItems": integer, "maxItems": integer,
            "pattern": {"type": "string"}},
            "required": ["type"], "additionalProperties": False}
        manual_schema = {"type": "object", "properties": {
            "purpose": string, "when_to_use": {"type": "array", "items": string},
            "inputs": {"type": "object"}, "outputs": {"type": "object"},
            "examples": {"type": "array"}, "failure_modes": {"type": "array", "items": string},
            "limitations": {"type": "array", "items": string}},
            "additionalProperties": False}
        test_case_schema = {"type": "object", "properties": {
            "input": {"type": "object"}, "expected": {}},
            "required": ["input", "expected"], "additionalProperties": False}
        python_runtime_schema = {"type": "object", "properties": {
            "implementation": string, "version": string, "abi": string},
            "required": ["implementation", "version", "abi"],
            "additionalProperties": False}
        platform_runtime_schema = {"type": "object", "properties": {
            "system": string, "machine": string},
            "required": ["system", "machine"], "additionalProperties": False}
        wheel_artifact_schema = {"type": "object", "properties": {
            "path": string, "filename": {"type": "string", "pattern": "^[^/\\\\]+\\.whl$"},
            "kind": {"type": "string", "enum": ["wheel"]},
            "sha256": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"}},
            "required": ["path", "filename", "kind", "sha256"],
            "additionalProperties": False}
        runtime_dependency_schema = {"type": "object", "properties": {
            "name": string, "version": string, "artifact": wheel_artifact_schema},
            "required": ["name", "version", "artifact"],
            "additionalProperties": False}
        cuda_runtime_schema = {"type": "object", "properties": {
            "toolkit": string, "minimum_driver": string},
            "required": ["toolkit", "minimum_driver"],
            "additionalProperties": False}
        runtime_environment_schema = {"type": "object", "properties": {
            "python": python_runtime_schema,
            "dependencies": {"type": "array", "items": runtime_dependency_schema},
            "accelerator": {"type": "string", "enum": ["cpu", "cuda"]},
            "platform": platform_runtime_schema, "cuda": cuda_runtime_schema},
            "required": ["python", "dependencies", "accelerator", "platform"],
            "additionalProperties": False}
        registry.add("list_tool_groups", "List optional Tool groups without loading their schemas.",
                     self._schema(), registry.group_index)
        registry.add("activate_tool_group", "Explicitly activate one optional Tool schema group.",
                     self._schema({"group": {"type": "string", "enum": [
                         "source_inspection", "web_acquisition", "asset_authoring"]}}, ["group"]),
                     registry.activate)
        registry.add("deactivate_tool_group", "Deactivate an optional Tool schema group.",
                     self._schema({"group": {"type": "string", "enum": [
                         "source_inspection", "web_acquisition", "asset_authoring"]}}, ["group"]),
                     registry.deactivate)
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
            {"argv": {"type": "array", "items": string, "minItems": 1},
             "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 600}}, ["argv"]), ws.run_command)
        registry.add("search_assets", "Search promoted shared Tool, Skill and Experience summaries.", self._schema(
            {"query": string, "limit": integer, "include_gaps": {"type": "boolean"}}, ["query"]), cap.search)
        registry.add("inspect_asset", "Load selected asset manual/contract detail.", self._schema({"asset_id": string}, ["asset_id"]), cap.inspect)
        registry.add("activate_shared_tool", "Bind one inspected promoted Tool to the current Adapter.",
                     self._schema({"tool_id": string}, ["tool_id"]), cap.activate_tool)
        registry.add("load_tool_source", "Explicitly load a Tool implementation after manual inspection.",
                     self._schema({"tool_id": string}, ["tool_id"]), cap.load_tool_source,
                     group="source_inspection")
        registry.add("materialize_skill", "Materialize a selected Skill controller into the workspace.",
                     self._schema({"skill_id": string}, ["skill_id"]), cap.materialize_skill,
                     group="source_inspection")
        registry.add("search_web", "Search public web sources for a capability.",
                     self._schema({"query": string, "limit": integer}, ["query"]),
                     cap.web_search, group="web_acquisition")
        registry.add("fetch_web_page", "Open one HTTPS public page.",
                     self._schema({"url": string, "max_chars": integer}, ["url"]),
                     cap.fetch_page, group="web_acquisition")
        registry.add("download_public_asset", "Download one HTTPS asset into the workspace with optional SHA256.", self._schema(
            {"url": string, "filename": string, "sha256": string}, ["url", "filename"]),
            cap.download, group="web_acquisition")
        registry.add("unpack_public_asset", "Safely unpack a downloaded archive inside the workspace.", self._schema(
            {"path": string, "destination": string}, ["path", "destination"]),
            cap.unpack, group="web_acquisition")
        registry.add("build_capability", "Build or compile-check an acquired capability bundle in isolation.", self._schema(
            {"directory": string, "argv": {"type": "array", "items": string}}, ["directory"]),
            cap.build, group="web_acquisition")
        tool_schema = self._schema({"name": string, "source_path": string, "description": string,
            "input_schema": schema_document, "output_schema": schema_document,
            "source_urls": {"type": "array", "items": string},
            "runtime_spec": runtime_environment_schema,
            "manual": manual_schema}, ["name", "source_path", "description", "input_schema", "output_schema"])
        package_spec_schema = {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["algorithm", "perception", "planner", "policy", "model"]},
            "entrypoint": string, "accelerator": {"type": "string", "enum": ["cpu", "cuda"]},
            "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 600},
            "runtime_requirements": {"type": "array", "items": string},
            "runtime": runtime_environment_schema,
            "checkpoint_sha256": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"}}},
            "required": ["kind", "entrypoint"], "additionalProperties": False}
        package_schema = self._schema({"name": string, "bundle_path": string, "description": string,
            "input_schema": schema_document, "output_schema": schema_document,
            "package_spec": package_spec_schema, "source_urls": {"type": "array", "items": string}},
            ["name", "bundle_path", "description", "input_schema", "output_schema", "package_spec"])
        registry.add("register_tool", "Register an immutable Tool version; call test_tool before it can be bound.",
                     tool_schema, cap.register_tool, group="asset_authoring")
        registry.add("register_capability_package", "Register an acquired bundle for isolated execution.",
                     package_schema, cap.register_package, group="asset_authoring")
        registry.add("revise_tool_manual", "Update a Tool manual using explicit execution evidence.", self._schema(
            {"tool_id": string, "manual": manual_schema, "evidence_paths": {"type": "array", "items": string, "minItems": 1}},
            ["tool_id", "manual", "evidence_paths"]), cap.revise_manual,
            group="asset_authoring")
        registry.add("test_tool", "Run JSON contract tests against a registered Tool.", self._schema(
            {"tool_id": string, "cases": {"type": "array", "items": test_case_schema, "minItems": 1}}, ["tool_id", "cases"]),
            cap.test_tool, group="asset_authoring")
        registry.add("register_skill", "Persist a successful reusable Skill.", self._schema(
            {"name": string, "task": string, "controller": string, "tool_ids": {"type": "array", "items": string},
             "evidence_paths": {"type": "array", "items": string}, "evidence": {"type": "object"}},
            ["name", "task", "controller", "tool_ids", "evidence_paths"]), cap.register_skill,
            group="asset_authoring")
        registry.add("register_experience", "Persist an evidence-backed Experience.", self._schema(
            {"name": string, "summary": string, "applicability": string, "keywords": {"type": "array", "items": string},
             "evidence_paths": {"type": "array", "items": string}},
            ["name", "summary", "applicability", "evidence_paths"]), cap.register_experience,
            group="asset_authoring")
        registry.add("promote_asset", "Promote a verified asset after successful integration evidence.", self._schema(
            {"asset_id": string, "evidence_paths": {"type": "array", "items": string, "minItems": 1},
             "applicability": {"type": "object"}}, ["asset_id", "evidence_paths"]), cap.promote_asset,
            group="asset_authoring")
        registry.add("record_gap", "Persist an unresolved capability Gap.", self._schema(
            {"name": string, "task": string, "failure_summary": string, "evidence_paths": {"type": "array", "items": string},
             "attempted_methods": {"type": "array", "items": string}, "missing_capability": string,
             "blocked_reason": string, "next_steps": {"type": "array", "items": string}},
            ["name", "task", "failure_summary", "evidence_paths", "attempted_methods",
             "missing_capability", "blocked_reason", "next_steps"]), cap.record_gap,
            group="asset_authoring")
        registry.add("run_controller", "Execute the current controller once and return sensor evidence.",
                     self._schema(), lambda: self._agent_evidence(self._run_controller()))
        registry.add("reset_case", "Create a fresh episode for the selected case when the Adapter supports it.",
                     self._schema(), self._reset_case)
        registry.add("inspect_execution", "Inspect the latest committed execution evidence summary.",
                     self._schema({"evidence_ref": string}), self._inspect_execution)
        registry.add("list_executions", "List opaque references to prior Controller experiments.",
                     self._schema(), self._list_executions)
        registry.add("list_artifacts", "List AgentArtifact handles registered for an execution.",
                     self._schema({"evidence_ref": string}, ["evidence_ref"]), self._list_artifacts)
        registry.add("view_sensor_artifact", "Read a bounded sensor/evidence artifact path.", self._schema(
            {"path": string, "max_chars": {"type": "integer", "minimum": 1,
                "maximum": 20000}, "offset_bytes": {"type": "integer", "minimum": 0},
              "frame_indices": {"type": "array", "items": {"type": "integer", "minimum": 0},
                                "maxItems": 16}},
            ["path"]), self._view_artifact)
        registry.add("finish", "Finish only after the task is actually verified.", self._schema({"summary": string}, ["summary"]), self._finish)
        case_ids = getattr(self.adapter, "case_ids", None)
        selector = getattr(self.adapter, "select", None)
        if case_ids is not None and callable(selector):
            registry.add("list_cases", "List model-selectable environment cases.",
                         self._schema(), self._list_cases)
            registry.add("select_case", "Select the case used by the next Controller execution.",
                         self._schema({"case_id": string}, ["case_id"]), self._select_case)
        return registry

    def _list_cases(self):
        return {"cases": [str(item) for item in getattr(self.adapter, "case_ids", ())],
                "selected": str(getattr(self.adapter, "active_case", ""))}

    def _list_executions(self):
        refs = []
        for row in self.event_store.events():
            if row.get("kind") != "execution":
                continue
            payload = row.get("payload") or {}
            if payload.get("artifact_uri"):
                refs.append({"evidence_ref": payload.get("artifact_uri"),
                             "summary": payload.get("summary") or {}})
        return {"executions": refs[-64:]}

    def _inspect_execution(self, evidence_ref: str | None = None):
        if evidence_ref:
            for row in self.event_store.events():
                payload = row.get("payload") or {}
                if payload.get("artifact_uri") == evidence_ref:
                    return self._agent_evidence(self._load_evidence_reference(payload))
            raise ProtocolError("unknown evidence reference")
        return self._agent_evidence(self.latest_evidence) if isinstance(self.latest_evidence, Mapping) else {}

    def _list_artifacts(self, evidence_ref: str):
        evidence = self._load_evidence_reference({"artifact_uri": evidence_ref,
                                                  "artifact_sha256": next(
                                                      (row.get("payload", {}).get("artifact_sha256")
                                                       for row in self.event_store.events()
                                                       if row.get("payload", {}).get("artifact_uri") == evidence_ref), None)})
        self._register_artifacts(evidence)
        handles = []
        def visit(value):
            if isinstance(value, Mapping):
                for item in value.values(): visit(item)
            elif isinstance(value, list):
                for item in value: visit(item)
            elif isinstance(value, str) and value in self._artifact_handles:
                handles.append(value)
        visit(evidence.get("execution"))
        return {"evidence_ref": evidence_ref, "artifacts": sorted(set(handles))}

    def _select_case(self, case_id: str):
        selector = getattr(self.adapter, "select", None)
        if not callable(selector):
            raise ProtocolError("Adapter does not provide selectable cases")
        selector(str(case_id))
        observe = getattr(self.adapter, "initial_observation", None)
        observation = observe() if callable(observe) else None
        self.context_builder.initial_observation = self._register_artifacts(observation)
        # Keep full previous evidence for recovery/audit, but remove it from
        # the replaceable model view after an explicit environment switch.
        self._agent_latest_evidence = None
        return {"selected": str(getattr(self.adapter, "active_case", case_id)),
                "observation": self.context_builder._observation_summary(observation)}

    def _reset_case(self):
        reset = getattr(self.adapter, "reset_case", None)
        if not callable(reset):
            reset = getattr(self.adapter, "restart_episode", None)
        if not callable(reset):
            raise ProtocolError("Adapter does not provide reset_case/restart_episode")
        observation = reset()
        if observation is None:
            observe = getattr(self.adapter, "initial_observation", None)
            observation = observe() if callable(observe) else None
        self.latest_evidence = None
        self._agent_latest_evidence = None
        self.state.update({"completion_valid": False, "finished": False,
                           "restored_evidence_unverified": False})
        self.context_builder.initial_observation = observation
        return {"reset": True, "observation": self.context_builder._observation_summary(observation)}

    def _view_artifact(self, path: str, max_chars: int = 12000,
                       offset_bytes: int = 0, frame_indices: list[int] | None = None):
        candidate = Path(path)
        if not candidate.is_absolute():
            if str(path).startswith("artifact://adapter/"):
                candidate = Path(getattr(self.adapter, "artifact_dir", "")) / str(path).removeprefix("artifact://adapter/")
            elif str(path).startswith("workspace://"):
                candidate = self.workspace.root / str(path).removeprefix("workspace://")
            elif str(path).startswith("run://"):
                candidate = self.root / str(path).removeprefix("run://")
            else:
                candidate = self.workspace.root / candidate
        handle = str(path)
        if handle.startswith(("artifact://", "run://")):
            candidate = self._artifact_handles.get(handle)
            if candidate is None:
                raise ProtocolError("artifact handle is not registered")
        else:
            candidate = (self.workspace.root / handle).resolve()
            if self.workspace.root not in candidate.parents:
                raise ProtocolError("sensor artifacts require a registered opaque artifact handle")
        if not candidate.is_file():
            raise ProtocolError("registered artifact is missing")
        relative = handle
        suffix = candidate.suffix.casefold()
        budgets = self.context_window.budgets
        size = candidate.stat().st_size
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            if size > budgets.max_image_bytes:
                raise ProtocolError("image artifact exceeds the multimodal size limit")
            data = candidate.read_bytes()
            import cv2
            import numpy as np
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if decoded is None:
                raise ProtocolError("image artifact cannot be decoded")
            pixels = int(decoded.shape[0]) * int(decoded.shape[1])
            if pixels > budgets.max_image_pixels or pixels > budgets.max_total_image_pixels:
                raise ProtocolError("image artifact exceeds the pixel limit")
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            return {"path": relative, "kind": "image", "bytes": len(data),
                    "shape": list(decoded.shape), "pixels": pixels,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}
        if suffix == ".npy":
            if size > budgets.max_array_bytes:
                raise ProtocolError("array artifact exceeds the byte limit")
            import numpy as np
            import cv2
            array = np.load(candidate, mmap_mode="r", allow_pickle=False)
            if int(array.size) > budgets.max_array_elements:
                raise ProtocolError("array artifact exceeds the element limit")
            if array.ndim == 0:
                sample = np.asarray(array)
            elif array.ndim == 1:
                stride = max(1, int(array.shape[0] / 1_000_000) + 1)
                sample = np.asarray(array[::stride])
            else:
                target_pixels = min(budgets.max_image_pixels, 1_000_000)
                stride = max(1, int((int(array.shape[0]) * int(array.shape[1])
                                     / max(1, target_pixels)) ** 0.5) + 1)
                sample = np.asarray(array[::stride, ::stride])
            minimum, maximum = float(np.nanmin(sample)), float(np.nanmax(sample))
            result = {"path": relative, "kind": "depth" if array.ndim == 2 else "array",
                      "shape": list(array.shape), "dtype": str(array.dtype),
                      "sampled_elements": int(sample.size),
                      "sample_min": minimum, "sample_max": maximum}
            if array.ndim == 2 and maximum > minimum:
                normalized = np.nan_to_num((sample - minimum) / (maximum - minimum), nan=0.0)
                preview = cv2.applyColorMap(np.asarray(normalized * 255, dtype=np.uint8), cv2.COLORMAP_TURBO)
                ok, encoded = cv2.imencode(".png", preview)
                if ok:
                    result["image_url"] = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()
            return result
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            if size > budgets.max_video_bytes:
                raise ProtocolError("video artifact exceeds the byte limit")
            import cv2
            capture = cv2.VideoCapture(str(candidate)); frames = []
            try:
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                indices = sorted(set(int(i) for i in (frame_indices or [])))
                if not indices:
                    return {"path": relative, "kind": "video", "frame_count": total,
                            "frames": [], "requires_frame_indices": True}
                indices = [i for i in indices if 0 <= i < total][:budgets.max_video_frames]
                deadline = time.monotonic() + 10.0
                total_pixels = 0
                for index in indices:
                    if time.monotonic() >= deadline:
                        break
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read()
                    if ok:
                        pixels = int(frame.shape[0]) * int(frame.shape[1])
                        total_pixels += pixels
                        if (pixels > budgets.max_image_pixels
                                or total_pixels > budgets.max_total_image_pixels):
                            raise ProtocolError("video keyframes exceed the pixel limit")
                        encoded_ok, encoded = cv2.imencode(".jpg", frame)
                        frames.append({"frame": index, "shape": list(frame.shape),
                            "image_url": ("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode())
                                if encoded_ok else None})
            finally:
                capture.release()
            return {"path": relative, "kind": "video", "frames": frames, "frame_count": total,
                    "image_urls": [row["image_url"] for row in frames if row.get("image_url")]}
        if suffix in {".ply", ".pcd", ".las", ".laz"}:
            with candidate.open("rb") as stream:
                data = stream.read(budgets.max_point_cloud_header_bytes)
            header = data.decode("ascii", errors="ignore")
            points = None
            for pattern in (r"element\s+vertex\s+(\d+)", r"POINTS\s+(\d+)"):
                match = re.search(pattern, header, flags=re.IGNORECASE)
                if match:
                    points = int(match.group(1)); break
            return {"path": relative, "kind": "point_cloud", "format": suffix[1:],
                    "bytes": size, "points": points,
                    "header_sha256": hashlib.sha256(data).hexdigest(),
                    "header_bytes": len(data)}
        maximum = min(max(int(max_chars), 1), 20000)
        offset = max(int(offset_bytes), 0)
        if offset > size:
            raise ProtocolError("artifact text offset exceeds file size")
        with candidate.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(maximum)
        next_offset = offset + len(data)
        return {"path": relative, "kind": "text",
                "content": data.decode("utf-8", errors="replace"),
                "bytes": size, "offset_bytes": offset,
                "next_offset_bytes": next_offset if next_offset < size else None,
                "truncated": next_offset < size}

    def _bound_tool_result(self, payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        content = self.context_window.bound_tool_payload(payload)
        summary = json.loads(content)
        if summary.get("truncated") is not True:
            return content, summary
        digest = str(summary["sha256"])
        directory = self.root / "artifacts" / "tool-results"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest}.json"
        if not target.exists():
            encoded = json.dumps(dict(payload), sort_keys=True, default=str) + "\n"
            added = len(encoded.encode())
            current = sum(path.stat().st_size for path in
                          (self.root / "artifacts").rglob("*") if path.is_file())
            if current + added > self.context_window.budgets.max_artifact_bytes:
                raise ProtocolError("run artifact disk quota exceeded")
            temporary = target.with_suffix(f".tmp-{time.time_ns()}")
            try:
                with temporary.open("x") as stream:
                    stream.write(encoded)
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        summary["artifact_uri"] = f"run://artifacts/tool-results/{target.name}"
        self._artifact_handles[summary["artifact_uri"]] = target
        return json.dumps(summary, default=str), summary

    @staticmethod
    def _evidence_authenticity(evidence: Mapping[str, Any]) -> bool:
        receipt = evidence.get("verification_receipt") if isinstance(evidence, Mapping) else None
        if not isinstance(receipt, Mapping):
            return False
        identity = evidence.get("environment_identity")
        return bool(receipt.get("verified") is True
                    and receipt.get("controller_sha256") == evidence.get("controller_sha256")
                    and receipt.get("environment_identity") == identity
                    and isinstance(identity, Mapping)
                    and receipt.get("episode_id") == identity.get("episode_id")
                    and receipt.get("environment_generation")
                        == identity.get("environment_generation"))

    @classmethod
    def _task_success(cls, evidence: Mapping[str, Any]) -> bool:
        receipt = evidence.get("verification_receipt") if isinstance(evidence, Mapping) else None
        return cls._evidence_authenticity(evidence) and isinstance(receipt, Mapping) \
            and receipt.get("verified") is True

    def _execution_identity(self) -> dict[str, Any]:
        provider = getattr(self.adapter, "execution_identity", None)
        value = provider() if callable(provider) else provider
        if not isinstance(value, Mapping):
            raise ProtocolError("Adapter execution_identity must be an object")
        result = dict(value)
        if not str(result.get("episode_id") or ""):
            raise ProtocolError("Adapter execution_identity requires episode_id")
        if result.get("environment_generation") in {None, ""}:
            raise ProtocolError("Adapter execution_identity requires environment_generation")
        return result

    def _resume_protocol(self) -> dict[str, Any] | None:
        provider = getattr(self.adapter, "resume_protocol", None)
        value = provider() if callable(provider) else provider
        if not isinstance(value, Mapping) or not isinstance(value.get("supports_resume"), bool):
            raise ProtocolError("Adapter resume_protocol requires supports_resume")
        if value.get("supports_resume") is not True:
            return None
        result = dict(value)
        required = ("resume_token", "environment_generation", "replay_allowed", "actions_idempotent")
        if any(key not in result for key in required) or not str(result.get("resume_token") or ""):
            raise ProtocolError("resumable Adapter protocol is incomplete")
        identity = self._execution_identity()
        if result.get("environment_generation") != identity.get("environment_generation"):
            raise ProtocolError("resume protocol generation differs from execution identity")
        return result

    def _finish(self, summary: str):
        evidence = self.latest_evidence
        current_sha = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest() \
            if self.workspace.controller.is_file() else None
        errors = []
        if not isinstance(evidence, Mapping):
            errors.append("controller has not been executed")
        else:
            if self.state.get("restored_evidence_unverified"):
                errors.append("restored evidence is not valid for the current Adapter generation")
            if evidence.get("controller_sha256") != current_sha:
                errors.append("latest evidence belongs to an older Controller")
            if not self._evidence_authenticity(evidence):
                errors.append("latest execution has no authentic Adapter verification receipt")
            if not self._task_success(evidence):
                errors.append("latest execution has no successful Adapter verification receipt")
            identity = self._execution_identity()
            recorded = evidence.get("environment_identity") or {}
            if identity and recorded and identity != recorded:
                errors.append("environment identity changed since verification")
            protocol = self._resume_protocol()
            if protocol and evidence.get("resume_token") != protocol.get("resume_token"):
                errors.append("execution resume token is not current")
        if errors:
            raise ProtocolError("completion rejected: " + "; ".join(errors))
        self.state.update({"finished": True, "completion_valid": True,
                           "completion_summary": str(summary), "successful_cases": 1})
        return dict(self.state)

    def _run_controller(self):
        if self.runtime is None: raise RuntimeError("controller runtime is not configured")
        controller_sha = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest()
        identity = self._execution_identity()
        protocol = self._resume_protocol()
        key_material = {"controller_sha256": controller_sha,
                        "environment_identity": identity,
                        "resume_token": (protocol or {}).get("resume_token")}
        execution_key = hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode()).hexdigest()
        pending = (self.state.get("completed_execution") or self.state.get("pending_execution")) \
            if self._recovery_mode else None
        if isinstance(pending, Mapping) and pending.get("execution_key") == execution_key:
            for row in self.event_store.events():
                payload = row.get("payload", {})
                if row.get("kind") == "execution" and payload.get("execution_key") == execution_key:
                    candidate = self._load_evidence_reference(payload)
                    candidate = {"reused_committed_execution": True, **candidate}
                    self.latest_evidence = candidate
                    self._agent_latest_evidence = self._agent_evidence(candidate)
                    self.state["pending_execution"] = None
                    self.state["completed_execution"] = None
                    self._recovery_mode = False
                    self.state["restored_evidence_unverified"] = False
                    return candidate
        self.state["pending_execution"] = {"execution_key": execution_key,
                                             "controller_sha256": controller_sha,
                                             "call_id": self._active_tool_call_id}
        self._checkpoint()
        self.budget.executions += 1
        self.cumulative_executions += 1
        result = self.runtime.execute(self.workspace.controller, self.adapter)
        self._register_artifacts(result)
        report = self.adapter.sensor_report(result)
        public_provider = getattr(self.adapter, "agent_evidence", None)
        public_report = public_provider(result, report) if callable(public_provider) else {}
        public_report = self._register_artifacts(public_report)
        if not isinstance(public_report, Mapping):
            raise ProtocolError("Adapter agent_evidence must return an object")
        evidence_ref = f"evidence://execution-{self.cumulative_executions:06d}"
        agent_evidence = AgentEvidence.from_execution(
            result, public_report, evidence_ref=evidence_ref).as_dict()
        verifier = getattr(self.adapter, "verification_receipt", None)
        if not callable(verifier):
            raise ProtocolError("Adapter must implement verification_receipt(execution)")
        receipt = dict(verifier(result))
        required_receipt = {"verified", "controller_sha256", "environment_identity",
                            "episode_id", "environment_generation"}
        if not required_receipt.issubset(receipt):
            raise ProtocolError("Adapter verification receipt is incomplete")
        if (receipt.get("controller_sha256") != controller_sha
                or receipt.get("environment_identity") != identity
                or receipt.get("episode_id") != identity.get("episode_id")
                or receipt.get("environment_generation") != identity.get("environment_generation")
                or not isinstance(receipt.get("verified"), bool)):
            raise ProtocolError("Adapter verification receipt is not bound to this execution")
        evidence = {"execution": result, "sensor_report": report, "controller_sha256": controller_sha,
                    "execution_key": execution_key,
                    "environment_identity": identity,
                    "verification_receipt": receipt,
                    "agent_evidence": agent_evidence,
                    "resume_token": (protocol or {}).get("resume_token"),
                    "environment_generation": (protocol or {}).get("environment_generation")}
        evidence_dir = self.root / "evidence"
        if not evidence_dir.exists(): evidence_dir.mkdir(parents=True)
        evidence_path = evidence_dir / f"execution-{self.cumulative_executions:06d}-{execution_key[:12]}.json"
        temporary = evidence_path.with_suffix(".tmp")
        try:
            encoded = json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n"
            added = len(encoded.encode())
            if self._evidence_bytes + added > self.max_evidence_bytes:
                raise ProtocolError("execution evidence disk quota exceeded")
            with temporary.open("w") as stream:
                stream.write(encoded); stream.flush()
                import os
                os.fsync(stream.fileno())
            temporary.replace(evidence_path)
            directory = os.open(evidence_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self._evidence_bytes += added
        finally:
            temporary.unlink(missing_ok=True)
        evidence["artifact_uri"] = f"run://evidence/{evidence_path.name}"
        evidence["artifact_sha256"] = _file_sha256(evidence_path)
        self.latest_evidence = evidence
        self._agent_latest_evidence = agent_evidence
        self.state["restored_evidence_unverified"] = False
        self.event_store.commit("execution", self._evidence_reference(evidence))
        self.state["pending_execution"] = None
        self.state["completed_execution"] = {"execution_key": execution_key}
        return evidence

    def _messages(self, task: str):
        evidence = self._agent_latest_evidence
        public_state = {"research": self.research_state,
                        "session_index": self.session_index,
                        "active_tool_groups": list(self.tools.active_groups),
                        "active_shared_tools": list(
                            self.capability_manager.bound_tool_ids)}
        if hasattr(self.adapter, "active_case"):
            public_state["selected_case"] = str(self.adapter.active_case)
        context = self.context_builder.build(task=task, latest_evidence=evidence,
                                             retrieved_assets=self.retrieved_assets,
                                             state=public_state)
        context = self.context_window.bound_context(
            context, artifact_root=self.root / "artifacts")
        if not self.messages:
            self.messages = [{"role": "system", "content": context["system"]},
                             {"role": "user", "content": json.dumps(context, default=str)}]
        else:
            # State is a replaceable view.  Keep the event/tool transcript
            # bounded, while the current context is represented exactly once.
            state_message = {"role": "user", "content": json.dumps(context, default=str)}
            self.messages = [self.messages[0], state_message, *self.messages[2:]]
        self.messages = self.context_window.compact(self.messages, self.tools.schemas)
        return self.messages

    @staticmethod
    def _context_size(messages: list[Mapping[str, Any]]) -> int:
        encoded = json.dumps(messages, default=str)
        # Image bytes are billed/processed as image inputs, not as prompt text.
        encoded = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "<image>", encoded)
        return len(encoded)

    def run(self, task: str | None = None):
        task = str(task or getattr(self.adapter, "instruction", ""))
        if self.checkpoint_task is not None and self.checkpoint_task != task:
            raise ProtocolError("run directory checkpoint belongs to a different task")
        self.current_task = task
        while not self.budget.exhausted() and not self.state.get("finished"):
            self.budget.steps += 1
            self.cumulative_steps += 1
            messages = self._messages(task)
            response = self.model.decide(messages=messages, tools=self.tools.schemas)
            calls = response.get("tool_calls") if isinstance(response, Mapping) else None
            if not isinstance(calls, list):
                self.event_store.commit("protocol_error", {"error": "model response must contain tool_calls", "response": response})
                self.messages.append({"role": "user", "content": "Protocol error: return a valid function call using the provided tools."})
                self._checkpoint(); continue
            call_pairs = []
            maximum_calls = self.context_window.budgets.max_tool_calls_per_turn
            for index, call in enumerate(calls[:maximum_calls]):
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
                call_name = str(call.get("name") or function.get("name") or "")
                raw_arguments = call.get("arguments") if "arguments" in call else function.get("arguments", "{}")
                if not isinstance(raw_arguments, str):
                    raw_arguments = json.dumps(raw_arguments, default=str)
                normalized = {"id": str(call.get("id") or f"call-{self.budget.steps}-{index}"),
                    "type": "function", "function": {"name": call_name, "arguments": raw_arguments}}
                call_pairs.append((call, normalized))
            normalized_calls = [item for _call, item in call_pairs]
            assistant = {"role": "assistant", "content": response.get("content", ""),
                         "tool_calls": normalized_calls}
            self.messages.append(assistant)
            delivered_images = 0
            for call, normalized in call_pairs:
                name = ""
                arguments: dict[str, Any] = {}
                try:
                    name = str(call.get("name") or call.get("function", {}).get("name") or "")
                    raw = call.get("arguments") or call.get("function", {}).get("arguments") or "{}"
                    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    self._active_tool_call_id = normalized["id"]
                    result = self.tools.invoke(name, arguments)
                    if name == "search_assets": self.retrieved_assets = result
                    payload = {"ok": True, "result": result}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    self._active_tool_call_id = None
                multimodal = []
                if payload.get("ok") and isinstance(payload.get("result"), Mapping):
                    value = payload["result"]
                    if value.get("image_url"):
                        multimodal.append(value["image_url"])
                    multimodal.extend(value.get("image_urls") or [])
                    if multimodal:
                        value = dict(value); value.pop("image_url", None); value.pop("image_urls", None)
                        if isinstance(value.get("frames"), list):
                            value["frames"] = [{k: v for k, v in row.items() if k != "image_url"}
                                               for row in value["frames"]]
                        payload = {**payload, "result": value, "multimodal_delivered": len(multimodal)}
                remaining_images = max(0, self.context_window.budgets.max_images_per_turn
                                       - delivered_images)
                multimodal = multimodal[:remaining_images]
                delivered_images += len(multimodal)
                content, event_payload = self._bound_tool_result(payload)
                self.messages.append({"role": "tool", "tool_call_id": normalized["id"],
                                      "content": content})
                if multimodal:
                    parts = [{"type": "text", "text": "Selected sensor artifact."}]
                    parts.extend({"type": "image_url", "image_url": {"url": url}}
                                 for url in multimodal)
                    self.messages.append({"role": "user", "content": parts})
                self.event_store.commit("tool_result", {"name": name,
                                                        "payload": event_payload})
                self._record_attempt(name, arguments, bool(payload.get("ok")))
            self._checkpoint()
        exhausted = self.budget.exhausted()
        finished = bool(self.state.get("finished", False))
        result = {"steps": self.budget.steps, "executions": self.budget.executions,
                  "budget_exhausted": exhausted, "finished": finished,
                  "resumable": bool(exhausted and not finished),
                  "session": {"index": self.session_index,
                              "steps": self.budget.steps,
                              "executions": self.budget.executions,
                              "elapsed_seconds": self.budget.elapsed()},
                  "cumulative": {"steps": self.cumulative_steps,
                                 "executions": self.cumulative_executions,
                                 "elapsed_seconds": self.cumulative_elapsed
                                    + self.budget.elapsed()},
                  "completion_valid": self.state.get("completion_valid", False),
                  "latest_evidence": (self._evidence_reference(self.latest_evidence)
                                      if isinstance(self.latest_evidence, Mapping)
                                      else None)}
        return result

    def _checkpoint(self):
        snapshot = self.workspace.snapshot()
        save_checkpoint(self.root, {"protocol": "roboforge-checkpoint-v1",
            "steps": self.budget.steps, "executions": self.budget.executions,
            "task": getattr(self, "current_task", None), "elapsed_seconds": self.budget.elapsed(),
            "session": {"index": self.session_index, "steps": self.budget.steps,
                        "executions": self.budget.executions,
                        "elapsed_seconds": self.budget.elapsed()},
            "cumulative": {"steps": self.cumulative_steps,
                           "executions": self.cumulative_executions,
                           "elapsed_seconds": self.cumulative_elapsed
                              + self.budget.elapsed()},
            "latest_evidence": (self._evidence_reference(self.latest_evidence)
                                if isinstance(self.latest_evidence, Mapping) else None),
            "snapshot_id": snapshot.snapshot_id,
            "active_tool_groups": list(self.tools.active_groups),
            "active_shared_tools": list(self.capability_manager.bound_tool_ids),
            "selected_case": getattr(self.adapter, "active_case", None),
            "retrieved_assets": self.retrieved_assets, "state": self.state,
            "research_state": self._bound_research_state(self.research_state)})


__all__ = ["AgentLoop", "LoopBudget", "ProtocolError"]
