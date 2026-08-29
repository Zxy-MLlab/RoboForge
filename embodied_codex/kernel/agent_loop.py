"""Canonical model-driven agent loop with structured function calling."""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from .capability_manager import CapabilityManager
from .context import ContextBuilder
from .context_window import ContextWindowManager
from .evidence import AgentEvidence, HarnessMetadata, build_execution_digest
from .events import EventStore
from .recovery import load_checkpoint, save_checkpoint
from .tools import CONSEQUENCE_LEVELS, ToolRegistry


@dataclass
class LoopBudget:
    max_steps: int = 60; max_executions: int = 20; timeout_seconds: float = 3600
    max_trials: int | None = None
    started: float = field(default_factory=time.monotonic); steps: int = 0; executions: int = 0
    elapsed_before: float = 0.0
    def elapsed(self):
        return self.elapsed_before + time.monotonic() - self.started
    def exhausted(self):
        execution_limit = self.max_trials if self.max_trials is not None else self.max_executions
        return (self.steps >= self.max_steps or self.executions >= execution_limit
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
        self.trial_index = 0
        self.trial_control_steps = 0
        self.cumulative_control_steps = 0
        self.controller_versions: list[dict[str, Any]] = []
        self.progress_ledger: list[dict[str, Any]] = []
        self._load_learning_state()
        self.research_state: dict[str, Any] = {"summary": "", "attempts": []}
        self.state: dict[str, Any] = {"finished": False, "last_tool_call": None,
                                      "completion_valid": False, "successful_cases": 0}
        self._active_tool_call_id: str | None = None
        self._artifact_handles: dict[str, Path] = {}
        self._artifact_handle_digests: dict[str, str] = {}
        # Adapter artifact paths may be reused by later executions.  Handles
        # therefore point at immutable snapshots owned by this run.
        self._artifact_scope: str | None = None
        self._artifact_manifest_path = self.root / "artifacts" / "manifest.json"
        self._load_artifact_manifest()
        self._pending_decision_id: str | None = None
        self._decision_records: dict[str, dict[str, Any]] = {}
        # Consequential interventions are fail-closed from the first call;
        # an explicit Decision Record is never optional.
        self._decision_protocol_active = True
        self._active_operation_decision_id: str | None = None
        self._current_model_response_id: str | None = None
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
                self.trial_index = int(checkpoint.get("trial_index", self.cumulative_executions))
                self.cumulative_control_steps = int(checkpoint.get("cumulative_control_steps", 0))
                self.session_index = int((checkpoint.get("session") or {}).get("index", 1)) + 1
                self.checkpoint_task = checkpoint.get("task")
                self.latest_evidence = self._load_evidence_reference(
                    checkpoint.get("latest_evidence"))
                self.state.update(checkpoint.get("state") or {})
                self._recovery_mode = (isinstance(self.state.get("completed_execution"), Mapping)
                                       or isinstance(self.state.get("pending_execution"), Mapping))
                self.research_state = self._bound_research_state(
                    checkpoint.get("research_state") or self.research_state)
                self.controller_versions = [dict(x) for x in checkpoint.get("controller_versions", [])
                                            if isinstance(x, Mapping)] or self.controller_versions
                self.progress_ledger = [dict(x) for x in checkpoint.get("progress_ledger", [])
                                        if isinstance(x, Mapping)] or self.progress_ledger
                decision_state = checkpoint.get("decision_state") or {}
                self._pending_decision_id = decision_state.get("pending_id")
                self._decision_records = {
                    str(key): dict(value) for key, value in
                    dict(decision_state.get("records") or {}).items()
                    if isinstance(value, Mapping)
                }
                pending_record = self._decision_records.get(str(self._pending_decision_id))
                if isinstance(pending_record, Mapping) and pending_record.get("status") in {"active", "committed"}:
                    self._active_operation_decision_id = str(self._pending_decision_id)
                self._decision_protocol_active = bool(decision_state.get("protocol_active"))
                restore_transport = getattr(self.model, "restore_transport_state", None)
                if callable(restore_transport):
                    restore_transport(checkpoint.get("model_transport"))
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
                # Historical evidence validity cannot resolve an in-flight
                # physical execution. Preserve pending uncertainty so the
                # execution path can require exact durable recovery or fail.
                if not isinstance(self.state.get("pending_execution"), Mapping):
                    self.state["completed_execution"] = None
                    self._recovery_mode = False
        initial = getattr(self.adapter, "initial_observation", None)
        if callable(initial) and getattr(self.context_builder, "initial_observation", None) is None:
            self.context_builder.initial_observation = self._canonical_observation(initial())
        self.context_builder.initial_observation = self._register_artifacts(
            self._canonical_observation(self.context_builder.initial_observation))

    def _schema(self, properties=None, required=()):
        return {"type": "object", "properties": dict(properties or {}),
                "required": list(required), "additionalProperties": False}

    def _load_learning_state(self):
        versions = self.root / "controller_versions" / "index.json"
        ledger = self.root / "progress" / "ledger.json"
        try:
            value = json.loads(versions.read_text())
            if isinstance(value, list): self.controller_versions = [dict(x) for x in value if isinstance(x, Mapping)]
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        try:
            value = json.loads(ledger.read_text())
            if isinstance(value, list): self.progress_ledger = [dict(x) for x in value if isinstance(x, Mapping)]
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    def _persist_learning_state(self):
        directory = self.root / "controller_versions"; directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.json").write_text(json.dumps(self.controller_versions[-128:], indent=2, sort_keys=True) + "\n")
        directory = self.root / "progress"; directory.mkdir(parents=True, exist_ok=True)
        (directory / "ledger.json").write_text(json.dumps(self.progress_ledger[-128:], indent=2, sort_keys=True) + "\n")

    def _snapshot_controller(self, trial_index: int, controller_sha: str) -> dict[str, Any]:
        source = self.workspace.controller.read_text()
        version_id = f"controller-v{trial_index:04d}-{controller_sha[:12]}"
        path = self.root / "controller_versions" / f"{version_id}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(source)
        record = {"version_id": version_id, "controller_sha256": controller_sha,
                  "path": f"run://controller_versions/{path.name}",
                  "trial_index": trial_index, "environment_generation":
                  self._execution_identity().get("environment_generation")}
        self.controller_versions = [x for x in self.controller_versions if x.get("version_id") != version_id]
        self.controller_versions.append(record); self._persist_learning_state()
        return record

    def _canonical_observation(self, observation: Any) -> Any:
        provider = getattr(self.adapter, "canonical_observation", None)
        if not callable(provider):
            if getattr(self.adapter, "observation_protocol", None) == "non_embodied":
                return observation
            raise ProtocolError("Adapter must provide canonical_observation")
        projected = provider(observation)
        if not isinstance(projected, Mapping):
            raise ProtocolError("Adapter canonical_observation must return a mapping")
        return projected

    def _load_artifact_manifest(self) -> None:
        """Restore only checksum-validated, run-relative artifact handles."""
        path = self._artifact_manifest_path
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text())
            entries = payload.get("artifacts") if isinstance(payload, Mapping) else None
            if not isinstance(entries, list):
                raise ValueError("artifact manifest entries are missing")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                uri, relative, digest = (entry.get("uri"), entry.get("path"), entry.get("sha256"))
                if not (isinstance(uri, str) and isinstance(relative, str)
                        and isinstance(digest, str) and uri.startswith(("artifact://", "run://"))):
                    continue
                candidate = (self.root / relative).resolve()
                if self.root not in candidate.parents or not candidate.is_file():
                    continue
                if _file_sha256(candidate) != digest:
                    continue
                self._artifact_handles[uri] = candidate
                self._artifact_handle_digests[uri] = digest
        except (OSError, ValueError, json.JSONDecodeError):
            # A corrupt optional index must not grant any file access.
            return

    def _persist_artifact_manifest(self) -> None:
        entries = []
        for uri, path in sorted(self._artifact_handles.items()):
            try:
                relative = path.resolve().relative_to(self.root).as_posix()
                expected = self._artifact_handle_digests.get(uri)
                if not path.is_file() or not expected:
                    raise ProtocolError("registered artifact is missing its immutable digest")
                if _file_sha256(path) != expected:
                    raise ProtocolError("registered artifact checksum mismatch")
                entries.append({"uri": uri, "path": relative, "sha256": expected})
            except (OSError, ValueError):
                raise ProtocolError("registered artifact manifest path is invalid")
        target = getattr(self, "_artifact_manifest_path",
                         self.root / "artifacts" / "manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"protocol": "roboforge-artifact-manifest-v1",
                                         "artifacts": entries}, sort_keys=True, indent=2) + "\n")
        temporary.replace(target)

    def _immutable_artifact(self, source: Path) -> tuple[str, Path]:
        """Snapshot an authorized source file before exposing an opaque handle."""
        source = source.resolve()
        digest = _file_sha256(source)
        scope = getattr(self, "_artifact_scope", None) or "unscoped"
        scope_token = hashlib.sha256(scope.encode()).hexdigest()[:24]
        identity = hashlib.sha256(
            (scope + "\0" + str(source) + "\0" + digest).encode()).hexdigest()
        directory = self.root / "artifacts" / "immutable" / scope_token
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{digest[:24]}-{source.name}"
        if target.exists():
            if not target.is_file() or _file_sha256(target) != digest:
                raise ProtocolError("immutable artifact snapshot checksum mismatch")
        else:
            temporary = target.with_name(f".{target.name}.tmp-{time.time_ns()}")
            try:
                shutil.copyfile(source, temporary)
                if _file_sha256(temporary) != digest or _file_sha256(source) != digest:
                    raise ProtocolError("artifact changed while creating immutable snapshot")
                temporary.chmod(0o444)
                with temporary.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)
        return f"artifact://agent/{identity[:24]}/{source.name}", target

    def _register_artifacts(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): self._register_artifacts(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._register_artifacts(item) for item in value]
        if not isinstance(value, str):
            return value
        if value in self._artifact_handles:
            return value
        candidate = None
        if value.startswith("artifact://adapter/"):
            candidate = Path(getattr(self.adapter, "artifact_dir", "")) / value.removeprefix("artifact://adapter/")
        elif value.startswith("artifact://"):
            resolver = getattr(self.adapter, "resolve_controller_artifact", None)
            if callable(resolver):
                try:
                    candidate = Path(resolver(value))
                except Exception:
                    candidate = None
        elif Path(value).is_absolute():
            candidate = Path(value)
        if candidate is None:
            return value
        try:
            candidate = candidate.resolve()
            if not candidate.is_file():
                token = hashlib.sha256(str(candidate).encode()).hexdigest()[:24]
                return f"artifact://agent/unavailable/{token}"
            allowed_roots = [self.workspace.root.resolve(), self.root.resolve()]
            adapter_root = getattr(self.adapter, "artifact_dir", None)
            if adapter_root:
                allowed_roots.append(Path(adapter_root).resolve())
            if not any(candidate == root or root in candidate.parents for root in allowed_roots):
                token = hashlib.sha256(str(candidate).encode()).hexdigest()[:24]
                return f"artifact://agent/denied/{token}"
            handle, snapshot = self._immutable_artifact(candidate)
            self._artifact_handles[handle] = snapshot
            if not hasattr(self, "_artifact_handle_digests"):
                self._artifact_handle_digests = {}
            self._artifact_handle_digests[handle] = _file_sha256(snapshot)
            self._persist_artifact_manifest()
            return handle
        except OSError:
            token = hashlib.sha256(str(value).encode()).hexdigest()[:24]
            return f"artifact://agent/unavailable/{token}"

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

    def _record_decision(self, *, goal: str | None = None,
                         evidence_refs: list[str] | None = None,
                         hypothesis: str | None = None,
                         decision: str | None = None,
                         expected_effect: str | None = None,
                         uncertainty: str | None = None,
                         decision_id: str | None = None):
        """Persist externally stated decision context, never private reasoning."""
        refs = [str(ref) for ref in (evidence_refs or [])]
        if any(not ref.startswith(("evidence://", "artifact://", "run://"))
               for ref in refs):
            raise ProtocolError("decision evidence_refs must be opaque routing references")
        call_id = getattr(self, "_active_tool_call_id", None) or "unknown-call"
        identifier = str(decision_id or f"decision-{call_id}")
        existing = getattr(self, "_decision_records", {}).get(identifier)
        if existing is not None:
            if existing.get("status") == "open":
                self._pending_decision_id = identifier
            return {"recorded": False, "duplicate": True, "decision_id": identifier}
        def public_text(value):
            if value is None:
                return None
            text = str(value)
            if text.startswith(("/", "\\")) or (len(text) > 2 and text[1] == ":"
                                                 and text[2:3] in ("/", "\\")):
                return "<host path omitted>"
            return text[:2000]
        record = {"decision_id": identifier, "goal": public_text(goal), "evidence_refs": refs[:16],
                  "hypothesis": public_text(hypothesis), "decision": public_text(decision),
                  "expected_effect": public_text(expected_effect), "uncertainty": public_text(uncertainty),
                  "status": "open", "intervention_id": identifier, "linked_call_ids": [],
                  "model_response_id": getattr(self, "_current_model_response_id", None),
                  "model_call_id": call_id}
        # A newly stated decision starts a new intervention and explicitly
        # closes any incompatible stale record.
        for previous_id, previous in self._decision_records.items():
            if previous_id != identifier and previous.get("status") in {"open", "active"}:
                previous["status"] = "superseded"
        self.event_store.commit("decision_record", record)
        self._decision_records[identifier] = record
        self._decision_protocol_active = True
        self._pending_decision_id = identifier
        return {"recorded": True, "decision_id": identifier,
                "evidence_refs": list(record["evidence_refs"])}

    def _claim_decision(self, operation: str, *, consequence: str = "PHYSICAL_INTERVENTION") -> str | None:
        """Link a consequential operation to one active intervention record."""
        level = str(consequence).upper()
        if level == "CONSEQUENTIAL":
            level = "PHYSICAL_INTERVENTION"
        if level not in CONSEQUENCE_LEVELS:
            raise ProtocolError(f"unknown consequence level: {consequence}")
        if level in {"READ_ONLY", "VALIDATION"}:
            return None
        identifier = getattr(self, "_pending_decision_id", None)
        record = getattr(self, "_decision_records", {}).get(identifier) if identifier else None
        if not isinstance(record, dict) or record.get("status") not in {"open", "active"}:
            raise ProtocolError(
                f"consequential operation {operation!r} requires a current open Decision Record")
        call_id = getattr(self, "_active_tool_call_id", None)
        linked = record.setdefault("linked_call_ids", [])
        token = str(call_id or operation)
        if token in linked:
            raise ProtocolError(
                f"consequential operation {operation!r} requires a current open Decision Record")
        # A claim is durably committed to this intervention.  The runtime
        # keeps _active_operation_decision_id so related validation and
        # execution calls can continue the same intervention without a second
        # high-level Decision Record.
        record["status"] = "active" if getattr(self, "_in_model_dispatch", False) else "committed"
        linked.append(token)
        record.setdefault("operations", []).append({"name": str(operation), "level": level})
        record["last_operation"] = str(operation)
        record["intervention_id"] = record.get("intervention_id") or str(identifier)
        self._active_operation_decision_id = str(identifier)
        return str(identifier)

    def _list_decisions(self):
        links: dict[str, list[dict[str, Any]]] = {}
        for row in self.event_store.events():
            if row.get("kind") != "decision_link":
                continue
            payload = row.get("payload") or {}
            identifier = str(payload.get("decision_id") or "")
            if identifier:
                links.setdefault(identifier, []).append(dict(payload))
        records = []
        for record in list(getattr(self, "_decision_records", {}).values())[-32:]:
            item = dict(record)
            if links.get(str(item.get("decision_id"))):
                item["links"] = links[str(item["decision_id"])][-16:]
            records.append(item)
        return {"decisions": records}

    def _recent_decisions(self, limit: int = 4) -> list[dict[str, Any]]:
        """Return a small factual decision view for the next model turn."""
        records = list(getattr(self, "_decision_records", {}).values())[-max(1, int(limit)):]
        return [{key: value for key, value in record.items()
                if key in {"decision_id", "goal", "evidence_refs", "hypothesis",
                            "decision", "expected_effect", "uncertainty", "status",
                            "committed_operation"}}
                for record in records]

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
            {"path": string, "content": string}, ["path", "content"]), ws.write_file,
            consequence="WORKSPACE_MUTATION")
        registry.add("replace_file_lines", "Atomically replace an inspected line range.", self._schema(
            {"path": string, "start_line": integer, "end_line": integer, "new_content": string,
             "expected_old_sha256": string}, ["path", "start_line", "end_line", "new_content"]), ws.replace_file_lines,
            consequence="WORKSPACE_MUTATION")
        registry.add("run_command", "Run a bounded engineering/test command in the workspace.", self._schema(
            {"argv": {"type": "array", "items": string, "minItems": 1},
             "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 600}}, ["argv"]), ws.run_command,
            consequence="WORKSPACE_MUTATION")
        registry.add("run_validation", "Run a command in an isolated disposable workspace stage.", self._schema(
            {"argv": {"type": "array", "items": string, "minItems": 1},
             "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 600}}, ["argv"]),
            ws.run_validation, consequence="VALIDATION")
        registry.add("search_assets", "Search promoted shared Tool, Skill and Experience summaries.", self._schema(
            {"query": string, "limit": integer, "include_gaps": {"type": "boolean"}}, ["query"]), cap.search)
        registry.add("inspect_asset", "Load selected asset manual/contract detail.", self._schema({"asset_id": string}, ["asset_id"]), cap.inspect)
        registry.add("activate_shared_tool", "Bind one inspected promoted Tool to the current Adapter.",
                     self._schema({"tool_id": string}, ["tool_id"]), cap.activate_tool,
                     consequence="ASSET_MUTATION")
        registry.add("load_tool_source", "Explicitly load a Tool implementation after manual inspection.",
                     self._schema({"tool_id": string}, ["tool_id"]), cap.load_tool_source,
                     group="source_inspection")
        registry.add("materialize_skill", "Materialize a selected Skill controller into the workspace.",
                     self._schema({"skill_id": string}, ["skill_id"]), cap.materialize_skill,
                     group="source_inspection", consequence="ASSET_MUTATION")
        registry.add("search_web", "Search public web sources for a capability.",
                     self._schema({"query": string, "limit": integer}, ["query"]),
                     cap.web_search, group="web_acquisition")
        registry.add("fetch_web_page", "Open one HTTPS public page.",
                     self._schema({"url": string, "max_chars": integer}, ["url"]),
                     cap.fetch_page, group="web_acquisition")
        registry.add("download_public_asset", "Download one HTTPS asset into the workspace with optional SHA256.", self._schema(
            {"url": string, "filename": string, "sha256": string}, ["url", "filename"]),
                     cap.download, group="web_acquisition", consequence="ASSET_MUTATION")
        registry.add("unpack_public_asset", "Safely unpack a downloaded archive inside the workspace.", self._schema(
            {"path": string, "destination": string}, ["path", "destination"]),
                     cap.unpack, group="web_acquisition", consequence="ASSET_MUTATION")
        registry.add("build_capability", "Build or compile-check an acquired capability bundle in isolation.", self._schema(
            {"directory": string, "argv": {"type": "array", "items": string}}, ["directory"]),
            cap.build, group="web_acquisition", consequence="ASSET_MUTATION")
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
                     tool_schema, cap.register_tool, group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("register_capability_package", "Register an acquired bundle for isolated execution.",
                     package_schema, cap.register_package, group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("revise_tool_manual", "Update a Tool manual using explicit execution evidence.", self._schema(
            {"tool_id": string, "manual": manual_schema, "evidence_paths": {"type": "array", "items": string, "minItems": 1}},
            ["tool_id", "manual", "evidence_paths"]), cap.revise_manual,
            group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("test_tool", "Run JSON contract tests against a registered Tool.", self._schema(
            {"tool_id": string, "cases": {"type": "array", "items": test_case_schema, "minItems": 1}}, ["tool_id", "cases"]),
            cap.test_tool, group="asset_authoring")
        registry.add("register_skill", "Persist a successful reusable Skill.", self._schema(
            {"name": string, "task": string, "controller": string, "tool_ids": {"type": "array", "items": string},
             "evidence_paths": {"type": "array", "items": string}, "evidence": {"type": "object"}},
            ["name", "task", "controller", "tool_ids", "evidence_paths"]), cap.register_skill,
            group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("register_experience", "Persist an evidence-backed Experience.", self._schema(
            {"name": string, "summary": string, "applicability": string, "keywords": {"type": "array", "items": string},
             "evidence_paths": {"type": "array", "items": string},
             "outcome": {"type": "string", "enum": ["success", "failure", "mixed"]}},
            ["name", "summary", "applicability", "evidence_paths"]), cap.register_experience,
            group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("promote_asset", "Promote a verified asset after successful integration evidence.", self._schema(
            {"asset_id": string, "evidence_paths": {"type": "array", "items": string, "minItems": 1},
             "applicability": {"type": "object"}}, ["asset_id", "evidence_paths"]), cap.promote_asset,
            group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("record_gap", "Persist an unresolved capability Gap.", self._schema(
            {"name": string, "task": string, "failure_summary": string, "evidence_paths": {"type": "array", "items": string},
             "attempted_methods": {"type": "array", "items": string}, "missing_capability": string,
             "blocked_reason": string, "next_steps": {"type": "array", "items": string}},
            ["name", "task", "failure_summary", "evidence_paths", "attempted_methods",
             "missing_capability", "blocked_reason", "next_steps"]), cap.record_gap,
            group="asset_authoring", consequence="ASSET_MUTATION")
        registry.add("run_controller", "Execute the current controller once and return sensor evidence.",
                     self._schema(), lambda: self._agent_evidence(self._run_controller()),
                     consequence="PHYSICAL_INTERVENTION")
        registry.add("record_decision",
                     "Record the model's concise externally stated decision context.",
                     self._schema({"decision_id": {"type": ["string", "null"]},
                                   "goal": {"type": ["string", "null"]},
                                   "evidence_refs": {"type": "array", "items": string,
                                                      "maxItems": 16},
                                   "hypothesis": {"type": ["string", "null"]},
                                   "decision": {"type": ["string", "null"]},
                                   "expected_effect": {"type": ["string", "null"]},
                                   "uncertainty": {"type": ["string", "null"]}},
                                  ["goal", "evidence_refs", "hypothesis", "decision",
                                   "expected_effect", "uncertainty"]),
                     self._record_decision)
        registry.add("list_decisions", "List bounded structured decision records.",
                     self._schema(), self._list_decisions)
        registry.add("reset_case", "Create a fresh episode for the selected case when the Adapter supports it.",
                     self._schema(), self._reset_case, consequence="ENVIRONMENT_MUTATION")
        registry.add("inspect_execution", "Inspect the latest committed execution evidence summary.",
                     self._schema({"evidence_ref": string}), self._inspect_execution)
        registry.add("list_executions", "List opaque references to prior Controller experiments.",
                     self._schema(), self._list_executions)
        registry.add("list_controller_versions", "List immutable Controller versions from this run.",
                     self._schema(), lambda: {"versions": self.controller_versions[-32:]})
        registry.add("inspect_controller_version", "Inspect one immutable Controller version.",
                     self._schema({"version_id": string}, ["version_id"]), self._inspect_controller_version)
        registry.add("compare_controller_versions", "Compare bounded facts about two Controller versions.",
                     self._schema({"version_a": string, "version_b": string}, ["version_a", "version_b"]),
                     self._compare_controller_versions)
        registry.add("restore_controller_version", "Restore a selected Controller version into the workspace.",
                     self._schema({"version_id": string}, ["version_id"]), self._restore_controller_version,
                     consequence="WORKSPACE_MUTATION")
        registry.add("record_progress", "Record model-authored task progress linked to evidence.",
                     self._schema({"summary": string, "status": {"type": "string",
                         "enum": ["working", "failed", "uncertain", "superseded"]},
                         "evidence_refs": {"type": "array", "items": string, "maxItems": 16},
                         "controller_version_id": {"type": ["string", "null"]},
                         "notes": {"type": ["string", "null"]}},
                         ["summary", "status", "evidence_refs"]), self._record_progress)
        registry.add("update_progress", "Update one model-authored progress record.",
                     self._schema({"progress_id": string, "summary": string,
                         "status": {"type": "string", "enum": ["working", "failed", "uncertain", "superseded"]},
                         "evidence_refs": {"type": "array", "items": string, "maxItems": 16},
                         "notes": {"type": ["string", "null"]}},
                         ["progress_id", "summary", "status", "evidence_refs"]), self._update_progress)
        registry.add("list_progress", "List bounded model-authored task progress.",
                     self._schema(), lambda: {"progress": self.progress_ledger[-32:]})
        registry.add("compare_executions",
                     "Compare public facts from two committed Controller executions.",
                     self._schema({"evidence_ref_a": string, "evidence_ref_b": string},
                                  ["evidence_ref_a", "evidence_ref_b"]),
                     self._compare_executions)
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
                         self._schema({"case_id": string}, ["case_id"]), self._select_case,
                         consequence="ENVIRONMENT_MUTATION")
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
                try:
                    evidence = self._load_evidence_reference(payload)
                    ref = ((evidence.get("agent_evidence") or {}).get("evidence_ref")
                           or f"evidence://execution-{len(refs)+1:06d}")
                except ProtocolError:
                    continue
                refs.append({"evidence_ref": ref, "summary": payload.get("summary") or {}})
        return {"executions": refs[-64:]}

    def _version(self, version_id):
        for item in self.controller_versions:
            if item.get("version_id") == str(version_id): return item
        raise ProtocolError("unknown Controller version")

    def _inspect_controller_version(self, version_id):
        item = self._version(version_id)
        path = self.root / "controller_versions" / Path(str(item["path"])).name
        if not path.is_file() or _file_sha256(path) != item.get("controller_sha256"):
            raise ProtocolError("Controller version integrity check failed")
        return {**item, "source": path.read_text()}

    def _compare_controller_versions(self, version_a, version_b):
        a, b = self._version(version_a), self._version(version_b)
        source_a = self._inspect_controller_version(version_a)["source"]
        source_b = self._inspect_controller_version(version_b)["source"]
        return {"version_a": a["version_id"], "version_b": b["version_id"],
                "sha256_equal": a["controller_sha256"] == b["controller_sha256"],
                "source_changed": source_a != source_b,
                "source_bytes_a": len(source_a.encode()), "source_bytes_b": len(source_b.encode())}

    def _restore_controller_version(self, version_id):
        item = self._inspect_controller_version(version_id)
        self.workspace.write_file("controller.py", item["source"])
        return {"restored": True, "version_id": item["version_id"],
                "controller_sha256": item["controller_sha256"]}

    def _validate_progress_refs(self, refs):
        valid = []
        for ref in refs or []:
            ref = str(ref)
            if not (ref.startswith("evidence://execution-") or ref.startswith("run://")):
                raise ProtocolError("progress evidence_refs must be run/evidence references")
            if ref.startswith("evidence://execution-"):
                try: self._execution_by_ref(ref)
                except Exception as exc: raise ProtocolError("progress evidence reference is invalid") from exc
            valid.append(ref)
        return valid

    def _record_progress(self, summary, status, evidence_refs, controller_version_id=None, notes=None):
        if controller_version_id is not None: self._version(controller_version_id)
        refs = self._validate_progress_refs(evidence_refs)
        record = {"progress_id": f"progress-{time.time_ns()}", "summary": str(summary),
                  "status": str(status), "evidence_refs": refs,
                  "controller_version_id": controller_version_id,
                  "trial_index": self.trial_index, "notes": notes}
        self.progress_ledger.append(record); self._persist_learning_state(); return record

    def _update_progress(self, progress_id, summary, status, evidence_refs, notes=None):
        refs = self._validate_progress_refs(evidence_refs)
        for record in self.progress_ledger:
            if record.get("progress_id") == str(progress_id):
                record.update(summary=str(summary), status=str(status), evidence_refs=refs, notes=notes)
                self._persist_learning_state(); return dict(record)
        raise ProtocolError("unknown progress record")

    def _inspect_execution(self, evidence_ref: str | None = None):
        if evidence_ref:
            for row in self.event_store.events():
                payload = row.get("payload") or {}
                if payload.get("artifact_uri"):
                    evidence = self._load_evidence_reference(payload)
                    candidate = ((evidence.get("agent_evidence") or {}).get("evidence_ref"))
                    if payload.get("artifact_uri") == evidence_ref or candidate == evidence_ref:
                        return self._agent_evidence(evidence)
            raise ProtocolError("unknown evidence reference")
        return self._agent_evidence(self.latest_evidence) if isinstance(self.latest_evidence, Mapping) else {}

    def _execution_by_ref(self, evidence_ref: str) -> Mapping[str, Any]:
        for row in self.event_store.events():
            if row.get("kind") != "execution":
                continue
            payload = row.get("payload") or {}
            if not payload.get("artifact_uri"):
                continue
            evidence = self._load_evidence_reference(payload)
            agent_ref = ((evidence.get("agent_evidence") or {}).get("evidence_ref"))
            if payload.get("artifact_uri") == evidence_ref or agent_ref == evidence_ref:
                return evidence
        raise ProtocolError("unknown evidence reference")

    @staticmethod
    def _comparison_value(value: Any):
        if isinstance(value, Mapping):
            return {str(key): AgentLoop._comparison_value(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [AgentLoop._comparison_value(item) for item in value]
        if isinstance(value, float):
            return round(value, 6)
        return value

    def _compare_executions(self, evidence_ref_a: str, evidence_ref_b: str):
        a = self._execution_by_ref(evidence_ref_a)
        b = self._execution_by_ref(evidence_ref_b)
        da = (a.get("agent_evidence") or {}).get("digest") or build_execution_digest(
            a.get("execution") or {}, controller_sha256=a.get("controller_sha256"),
            diagnostics=a.get("sensor_report"))
        db = (b.get("agent_evidence") or {}).get("digest") or build_execution_digest(
            b.get("execution") or {}, controller_sha256=b.get("controller_sha256"),
            diagnostics=b.get("sensor_report"))
        actions_a, actions_b = da.get("actions") or [], db.get("actions") or []
        tools_a, tools_b = da.get("tool_calls") or [], db.get("tool_calls") or []
        verify_a, verify_b = da.get("verifications") or [], db.get("verifications") or []
        target_fields = ("target_xyz", "target_ref", "pose_ref", "offset",
                         "quaternion_xyzw", "rotation_matrix")
        def stable_target(item):
            requested = item.get("requested") if isinstance(item, Mapping) else {}
            result = item.get("result") if isinstance(item, Mapping) else {}
            if isinstance(result, Mapping) and any(key in result for key in
                                                   ("target_xyz", "target_quaternion_xyzw")):
                return {key: result.get(key) for key in
                        ("target_xyz", "target_quaternion_xyzw") if key in result}
            if not isinstance(requested, Mapping):
                return {}
            return {key: requested.get(key) for key in target_fields if key in requested}
        target_a = [stable_target(item) for item in actions_a]
        target_b = [stable_target(item) for item in actions_b]
        types_a = [item.get("type") for item in actions_a]
        types_b = [item.get("type") for item in actions_b]
        def gripper_command(item):
            requested = item.get("requested") if isinstance(item, Mapping) else {}
            if not isinstance(requested, Mapping):
                return None
            return {key: requested.get(key) for key in ("gripper", "command")
                    if key in requested}
        gripper_a = [gripper_command(item) for item in actions_a]
        gripper_b = [gripper_command(item) for item in actions_b]
        endpoint_a = [{key: item.get("result", {}).get(key)
                       for key in ("eef_before", "eef_after")}
                      for item in actions_a]
        endpoint_b = [{key: item.get("result", {}).get(key)
                       for key in ("eef_before", "eef_after")}
                      for item in actions_b]
        position_errors_a = [item.get("result", {}).get("final_position_error_m")
                             for item in actions_a]
        position_errors_b = [item.get("result", {}).get("final_position_error_m")
                             for item in actions_b]
        orientation_errors_a = [item.get("result", {}).get("final_orientation_error_rad")
                                for item in actions_a]
        orientation_errors_b = [item.get("result", {}).get("final_orientation_error_rad")
                                for item in actions_b]
        transitions_a = [item.get("transition") or {} for item in actions_a]
        transitions_b = [item.get("transition") or {} for item in actions_b]
        facts_a = [dict(item) for item in verify_a if isinstance(item, Mapping)]
        facts_b = [dict(item) for item in verify_b if isinstance(item, Mapping)]
        fact_fields = sorted({str(key) for row in facts_a + facts_b for key in row
                              if key != "verifier"})
        changed_fact_fields = [key for key in fact_fields
                               if [row.get(key) for row in facts_a] !=
                                  [row.get(key) for row in facts_b]]
        def transition_values(transitions, key):
            values = []
            for transition in transitions:
                delta = transition.get("delta") if isinstance(transition, Mapping) else {}
                values.append(delta.get(key) if isinstance(delta, Mapping) else None)
            return values
        return {
            "controller_changed": a.get("controller_sha256") != b.get("controller_sha256"),
            "tool_calls": {"changed": self._comparison_value(tools_a) != self._comparison_value(tools_b),
                           "count_changed": len(tools_a) != len(tools_b),
                           "tool_ids_changed": [x.get("tool_id") for x in tools_a] !=
                                                [x.get("tool_id") for x in tools_b]},
            "actions": {"count_changed": len(actions_a) != len(actions_b),
                        "requested_targets_changed": self._comparison_value(target_a) !=
                                                     self._comparison_value(target_b),
                        "gripper_commands_changed": self._comparison_value(gripper_a) !=
                                                     self._comparison_value(gripper_b),
                        "action_types_changed": types_a != types_b},
            "behavior": {"eef_endpoints_changed": self._comparison_value(endpoint_a) !=
                                             self._comparison_value(endpoint_b),
                         "position_errors_changed": self._comparison_value(position_errors_a) !=
                                                    self._comparison_value(position_errors_b),
                         "orientation_errors_changed": self._comparison_value(orientation_errors_a) !=
                                                        self._comparison_value(orientation_errors_b),
                         "robot_motion_changed": self._comparison_value(
                             transition_values(transitions_a, "robot_motion")) !=
                             self._comparison_value(transition_values(transitions_b, "robot_motion")),
                         "eef_displacement_changed": self._comparison_value(
                             transition_values(transitions_a, "eef_displacement")) !=
                             self._comparison_value(transition_values(transitions_b, "eef_displacement")),
                         "entity_displacement_changed": self._comparison_value(
                             transition_values(transitions_a, "entity_displacement")) !=
                             self._comparison_value(transition_values(transitions_b, "entity_displacement")),
                         "action_frame_error_changed": self._comparison_value(
                             transition_values(transitions_a, "action_frame")) !=
                             self._comparison_value(transition_values(transitions_b, "action_frame"))},
            "verification": {"changed": self._comparison_value(verify_a) !=
                                      self._comparison_value(verify_b),
                              "facts_a": self._comparison_value(facts_a),
                              "facts_b": self._comparison_value(facts_b),
                              "changed_fields": changed_fact_fields},
        }

    def _list_artifacts(self, evidence_ref: str):
        payload = None
        for row in self.event_store.events():
            if row.get("kind") != "execution":
                continue
            candidate = row.get("payload", {})
            try:
                evidence = self._load_evidence_reference(candidate)
            except ProtocolError:
                continue
            agent_ref = (evidence.get("agent_evidence") or {}).get("evidence_ref")
            if candidate.get("artifact_uri") == evidence_ref or agent_ref == evidence_ref:
                payload = candidate
                break
        if payload is None:
            raise ProtocolError("unknown evidence reference")
        evidence = self._load_evidence_reference(payload)
        self._register_artifacts(evidence)
        handles = []
        def visit(value):
            if isinstance(value, Mapping):
                for item in value.values(): visit(item)
            elif isinstance(value, list):
                for item in value: visit(item)
            elif isinstance(value, str) and value in self._artifact_handles:
                handles.append(value)
        # Only the explicit AgentEvidence projection is readable by the model;
        # Harness metadata and raw execution payload remain private.
        visit(evidence.get("agent_evidence") or {})
        return {"evidence_ref": evidence_ref, "artifacts": sorted(set(handles))}

    def _select_case(self, case_id: str):
        selector = getattr(self.adapter, "select", None)
        if not callable(selector):
            raise ProtocolError("Adapter does not provide selectable cases")
        selector(str(case_id))
        observe = getattr(self.adapter, "initial_observation", None)
        observation = observe() if callable(observe) else None
        self.context_builder.initial_observation = self._register_artifacts(
            self._canonical_observation(observation))
        # Keep full previous evidence for recovery/audit, but remove it from
        # the replaceable model view after an explicit environment switch.
        self._agent_latest_evidence = None
        return {"selected": str(getattr(self.adapter, "active_case", case_id)),
                "observation": self.context_builder._observation_summary(
                    self.context_builder.initial_observation)}

    def _reset_case(self):
        reset = getattr(self.adapter, "reset_case", None)
        if not callable(reset):
            reset = getattr(self.adapter, "restart_episode", None)
        if not callable(reset):
            raise ProtocolError("Adapter does not provide reset_case/restart_episode")
        before_identity = self._execution_identity()
        observation = reset()
        if observation is None:
            observe = getattr(self.adapter, "initial_observation", None)
            observation = observe() if callable(observe) else None
        after_identity = self._execution_identity()
        before_generation = before_identity.get("environment_generation")
        after_generation = after_identity.get("environment_generation")
        if not after_generation or after_generation == before_generation:
            raise ProtocolError("Adapter reset did not create a fresh environment generation")
        # The successful fresh generation is the only operation that can make
        # an unknown prior physical execution irrelevant without replaying it.
        self.state["pending_execution"] = None
        self.state["completed_execution"] = None
        self._recovery_mode = False
        self.latest_evidence = None
        self._agent_latest_evidence = None
        self.state.update({"completion_valid": False, "finished": False,
                           "restored_evidence_unverified": False})
        observation = self._register_artifacts(self._canonical_observation(observation))
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
            expected_digest = getattr(self, "_artifact_handle_digests", {}).get(handle)
            if expected_digest and (not candidate.is_file() or _file_sha256(candidate) != expected_digest):
                raise ProtocolError("registered artifact checksum mismatch")
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
        if not hasattr(self, "_artifact_handle_digests"):
            self._artifact_handle_digests = {}
        self._artifact_handle_digests[summary["artifact_uri"]] = _file_sha256(target)
        self._persist_artifact_manifest()
        return json.dumps(summary, default=str), summary

    @staticmethod
    def _evidence_authenticity(evidence: Mapping[str, Any]) -> bool:
        receipt = evidence.get("verification_receipt") if isinstance(evidence, Mapping) else None
        if not isinstance(receipt, Mapping):
            return False
        identity = evidence.get("environment_identity")
        return bool(receipt.get("controller_sha256") == evidence.get("controller_sha256")
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
        episodic = bool(getattr(self.adapter, "episodic_trials", False))
        if episodic and not self._recovery_mode:
            reset = getattr(self.adapter, "reset_case", None)
            if not callable(reset):
                raise ProtocolError("episodic Adapter must provide reset_case")
            reset()
        self.trial_index = self.cumulative_executions + 1
        self.trial_control_steps = 0
        controller_sha = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest()
        version = self._snapshot_controller(self.trial_index, controller_sha)
        identity = self._execution_identity()
        protocol = self._resume_protocol()
        decision_id = (getattr(self, "_active_operation_decision_id", None)
                       or (getattr(self, "_pending_decision_id", None)
                           if self._active_tool_call_id is None else None))
        # A fresh model tool call is always a new experiment.  Only recovery
        # may reuse the exact call identity persisted in the checkpoint.
        call_id = self._active_tool_call_id or f"direct-{time.time_ns()}"
        key_material = {"tool_call_id": call_id,
                        "controller_sha256": controller_sha,
                        "environment_identity": identity,
                        "resume_token": (protocol or {}).get("resume_token")}
        execution_key = hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode()).hexdigest()
        completed = self.state.get("completed_execution") if self._recovery_mode else None
        pending = completed or (self.state.get("pending_execution") if self._recovery_mode else None)
        if isinstance(completed, Mapping):
            for row in self.event_store.events():
                payload = row.get("payload") or {}
                if (row.get("kind") == "execution"
                        and payload.get("execution_key") == completed.get("execution_key")
                        and payload.get("environment_identity") == identity):
                    candidate = self._load_evidence_reference(payload)
                    candidate = {"reused_committed_execution": True, **candidate}
                    self.latest_evidence = candidate
                    self._agent_latest_evidence = self._agent_evidence(candidate)
                    self.state["completed_execution"] = None
                    self.state["pending_execution"] = None
                    self._recovery_mode = False
                    return candidate
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
        if isinstance(pending, Mapping) and pending.get("execution_key"):
            # The process may have crashed after the execution event was
            # durably committed but before the checkpoint was advanced.  Match
            # that exact persisted execution id; never infer identity from a
            # newly emitted model call.
            for row in self.event_store.events():
                payload = row.get("payload") or {}
                if (row.get("kind") == "execution"
                        and payload.get("execution_key") == pending.get("execution_key")):
                    candidate = self._load_evidence_reference(payload)
                    candidate = {"reused_committed_execution": True, **candidate}
                    self.latest_evidence = candidate
                    self._agent_latest_evidence = self._agent_evidence(candidate)
                    self.state["pending_execution"] = None
                    self.state["completed_execution"] = None
                    self._recovery_mode = False
                    return candidate
        if self._recovery_mode and isinstance(pending, Mapping):
            raise ProtocolError("pending physical execution outcome is unknown; reset or Adapter recovery is required")
        if decision_id is None and self._active_tool_call_id is not None:
            decision_id = self._claim_decision("run_controller")
        self.state["pending_execution"] = {"execution_key": execution_key,
                                             "controller_sha256": controller_sha,
                                             "call_id": self._active_tool_call_id}
        self._checkpoint()
        self._artifact_scope = execution_key
        self.budget.executions += 1
        self.cumulative_executions += 1
        begin = getattr(self.adapter, "begin_controller_execution", None)
        if callable(begin):
            begin()
        result = self.runtime.execute(self.workspace.controller, self.adapter)
        self.trial_control_steps = int(getattr(self.adapter, "step", 0) or 0)
        self.cumulative_control_steps += self.trial_control_steps
        self._register_artifacts(result)
        report = self.adapter.sensor_report(result)
        public_provider = getattr(self.adapter, "agent_evidence", None)
        public_report = public_provider(result, report) if callable(public_provider) else {}
        public_report = self._register_artifacts(public_report)
        if not isinstance(public_report, Mapping):
            raise ProtocolError("Adapter agent_evidence must return an object")
        # Preserve only opaque handles from this execution's RPC observations
        # so a model can inspect the image it just requested without receiving
        # host paths or benchmark metadata.
        visible_artifacts = []
        def collect(value):
            if isinstance(value, Mapping):
                for item in value.values(): collect(item)
            elif isinstance(value, list):
                for item in value: collect(item)
            elif isinstance(value, str) and value.startswith("artifact://agent/"):
                visible_artifacts.append(value)
        transformed_execution = self._register_artifacts(result)
        collect(transformed_execution)
        if visible_artifacts:
            public_report = {**dict(public_report), "artifacts": sorted(set(visible_artifacts))}
            if not any(key in public_report for key in ("rgb_path", "image_uri")):
                public_report["rgb_path"] = sorted(set(visible_artifacts))[0]
        evidence_ref = f"evidence://execution-{self.cumulative_executions:06d}"
        digest = build_execution_digest(transformed_execution,
                                        controller_sha256=controller_sha,
                                        diagnostics=public_report)
        agent_evidence = AgentEvidence.from_execution(
            transformed_execution, public_report, digest=digest,
            evidence_ref=evidence_ref).as_dict()
        agent_evidence["trial"] = {"trial_index": self.trial_index,
                                   "trial_control_steps": self.trial_control_steps,
                                   "cumulative_control_steps": self.cumulative_control_steps,
                                   "trial_horizon_exhausted": bool(getattr(
                                       self.adapter, "trial_horizon_exhausted", False)),
                                   "controller_version_id": version["version_id"]}
        if decision_id:
            agent_evidence["decision_id"] = decision_id
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
                    "environment_generation": (protocol or {}).get("environment_generation"),
                    "decision_id": decision_id,
                    "trial_index": self.trial_index,
                    "trial_control_steps": self.trial_control_steps,
                    "cumulative_control_steps": self.cumulative_control_steps,
                    "trial_horizon_exhausted": bool(getattr(
                        self.adapter, "trial_horizon_exhausted", False)),
                    "controller_version_id": version["version_id"]}
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
        execution_reference = self._evidence_reference(evidence)
        if evidence.get("decision_id"):
            execution_reference["decision_id"] = evidence["decision_id"]
        self.event_store.commit("execution", execution_reference)
        if evidence.get("decision_id"):
            self.event_store.commit("decision_link", {
                "decision_id": evidence["decision_id"],
                "evidence_ref": evidence_ref,
                "artifact_uri": evidence["artifact_uri"],
                "controller_sha256": controller_sha,
                "execution_key": execution_key})
            record = self._decision_records.get(str(evidence["decision_id"]))
            if isinstance(record, dict):
                record["status"] = "committed"
                record["linked_execution_id"] = execution_key
                record["linked_controller_sha"] = controller_sha
        self._artifact_scope = None
        self._pending_decision_id = None
        self._active_operation_decision_id = None
        self.state["pending_execution"] = None
        self.state["completed_execution"] = {"execution_key": execution_key,
                                              "call_id": call_id}
        return evidence

    def _messages(self, task: str):
        evidence = self._agent_latest_evidence
        public_state = {"research": self.research_state,
                        "session_index": self.session_index,
                        "trial_index": self.trial_index,
                        "controller_versions": self.controller_versions[-16:],
                        "task_progress": self.progress_ledger[-16:],
                        "recent_decisions": self._recent_decisions(),
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
            audit = response.get("audit") if isinstance(response, Mapping) else None
            if isinstance(audit, Mapping):
                # Transport metadata is persisted for provenance, but never
                # added to the model-visible transcript.
                self.event_store.commit("model_call", dict(audit))
                self._current_model_response_id = (str(audit["response_id"])
                                                   if audit.get("response_id") else None)
            calls = response.get("tool_calls") if isinstance(response, Mapping) else None
            if not isinstance(calls, list):
                self.event_store.commit("protocol_error", {"error": "model response must contain tool_calls", "response": response})
                self.messages.append({"role": "user", "content": "Protocol error: return a valid function call using the provided tools."})
                self._checkpoint(); continue
            call_pairs = []
            maximum_calls = self.context_window.budgets.max_tool_calls_per_turn
            for index, call in enumerate(calls):
                if not isinstance(call, Mapping):
                    raise ProtocolError(
                        f"model function call {index} is not a mapping")
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
            self._in_model_dispatch = True
            delivered_images = 0
            for call_index, (call, normalized) in enumerate(call_pairs):
                name = ""
                arguments: dict[str, Any] = {}
                claimed_decision_id = None
                skipped = call_index >= maximum_calls
                try:
                    name = str(call.get("name") or call.get("function", {}).get("name") or "")
                    raw = call.get("arguments") or call.get("function", {}).get("arguments") or "{}"
                    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    if skipped:
                        payload = {"ok": False, "status": "failed",
                            "error": ("Tool call was not executed because the response "
                                      f"exceeded the per-turn limit of {maximum_calls}")}
                    else:
                        self._active_tool_call_id = normalized["id"]
                        metadata = self.tools.metadata(name)
                        active_id = getattr(self, "_active_operation_decision_id", None)
                        active_record = self._decision_records.get(active_id, {}) if active_id else {}
                        pending_id = getattr(self, "_pending_decision_id", None)
                        pending_record = self._decision_records.get(pending_id, {}) if pending_id else {}
                        if (metadata.consequence not in {"READ_ONLY", "VALIDATION"}
                                and not ((pending_record.get("status") == "open")
                                         or (active_id and active_id == pending_id
                                             and active_record.get("status") in {"active", "committed"}))):
                            raise ProtocolError(
                                "A Decision Record is required before starting a consequential intervention.")
                        if (active_id and self._pending_decision_id == active_id
                                and active_record.get("status") in {"active", "committed"}
                                and metadata.consequence not in {"READ_ONLY", "VALIDATION"}):
                            claimed_decision_id = str(active_id)
                        else:
                            claimed_decision_id = self._claim_decision(
                                name, consequence=metadata.consequence)
                        if metadata.consequence in {"READ_ONLY", "VALIDATION"} and active_id:
                            # Read/validation calls remain part of the current
                            # intervention for provenance and never clear it.
                            claimed_decision_id = active_id
                            token = str(normalized["id"])
                            if token not in active_record.setdefault("linked_call_ids", []):
                                active_record["linked_call_ids"].append(token)
                                active_record.setdefault("operations", []).append(
                                    {"name": name, "level": metadata.consequence})
                        if claimed_decision_id is not None:
                            self._active_operation_decision_id = claimed_decision_id
                        result = self.tools.invoke(name, arguments)
                        if name == "search_assets": self.retrieved_assets = result
                        payload = {"ok": True, "result": result}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                finally:
                    self._active_tool_call_id = None
                    # Keep the intervention id across related calls; it is
                    # cleared when the intervention's execution is recorded.
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
                record_tool_output = getattr(self.model, "record_tool_output", None)
                if callable(record_tool_output):
                    record_tool_output(normalized["id"], content,
                                       multimodal_inputs=multimodal,
                                       failed=not bool(payload.get("ok")))
                tool_event = {"name": name, "payload": event_payload, "skipped": skipped}
                if claimed_decision_id:
                    tool_event["decision_id"] = claimed_decision_id
                if getattr(self, "_current_model_response_id", None):
                    tool_event["model_response_id"] = self._current_model_response_id
                self.event_store.commit("tool_result", tool_event)
                if not skipped:
                    self._record_attempt(name, arguments, bool(payload.get("ok")))
            self._checkpoint()
            self._in_model_dispatch = False
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
                  "physical_trials": self.cumulative_executions,
                  "cumulative_control_steps": self.cumulative_control_steps,
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
            "trial_index": self.trial_index,
            "cumulative_control_steps": self.cumulative_control_steps,
            "controller_versions": self.controller_versions[-128:],
            "progress_ledger": self.progress_ledger[-128:],
            "latest_evidence": (self._evidence_reference(self.latest_evidence)
                                if isinstance(self.latest_evidence, Mapping) else None),
            "snapshot_id": snapshot.snapshot_id,
            "active_tool_groups": list(self.tools.active_groups),
            "active_shared_tools": list(self.capability_manager.bound_tool_ids),
            "selected_case": getattr(self.adapter, "active_case", None),
            "retrieved_assets": self.retrieved_assets, "state": self.state,
            "model_transport": (self.model.transport_state()
                                 if callable(getattr(self.model, "transport_state", None)) else None),
            "research_state": self._bound_research_state(self.research_state),
            "decision_state": {"protocol_active": getattr(self, "_decision_protocol_active", False),
                               "pending_id": getattr(self, "_pending_decision_id", None),
                               "records": dict(list(getattr(self, "_decision_records", {}).items())[-32:])}})


__all__ = ["AgentLoop", "LoopBudget", "ProtocolError"]
