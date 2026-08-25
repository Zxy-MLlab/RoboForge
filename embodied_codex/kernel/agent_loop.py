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
                 web_search: Any = None, policies: list[Any] | None = None,
                 resume: bool = True):
        self.model, self.workspace, self.adapter = model, workspace, adapter
        self.context_builder, self.capability_manager = context_builder, capability_manager
        self.runtime = runtime; self.root = Path(root or workspace.root.parent).resolve()
        self.event_store = event_store or EventStore(self.root / "events", protect=True)
        self.budget = budget or LoopBudget()
        self.web_search = web_search; self.policies = list(policies or [])
        self.latest_evidence = None; self.retrieved_assets = None; self.messages: list[dict[str, Any]] = []
        self.context_window = ContextWindowManager()
        self.max_context_chars = self.context_window.max_message_chars
        self.max_tool_result_chars = self.context_window.max_tool_result_chars
        self.checkpoint_task = None
        self.state: dict[str, Any] = {"finished": False, "last_tool_call": None,
                                      "completion_valid": False, "successful_cases": 0}
        if resume:
            checkpoint = load_checkpoint(self.root)
            if checkpoint:
                self.budget.steps = int(checkpoint.get("steps", 0)); self.budget.executions = int(checkpoint.get("executions", 0))
                self.budget.elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
                self.checkpoint_task = checkpoint.get("task")
                self.latest_evidence = checkpoint.get("latest_evidence"); self.state.update(checkpoint.get("state") or {})
                if checkpoint.get("snapshot_id"):
                    self.workspace.restore(checkpoint["snapshot_id"])
                self.retrieved_assets = checkpoint.get("retrieved_assets")
        self.tools = self._build_tools()
        if self.latest_evidence is not None:
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

    def _schema(self, properties=None, required=()):
        return {"type": "object", "properties": dict(properties or {}),
                "required": list(required), "additionalProperties": False}

    def _build_tools(self):
        registry = ToolRegistry(); ws = self.workspace; cap = self.capability_manager
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
        registry.add("load_tool_source", "Explicitly load a Tool implementation after manual inspection.", self._schema({"tool_id": string}, ["tool_id"]), cap.load_tool_source)
        registry.add("search_web", "Search public web sources for a capability.", self._schema({"query": string, "limit": integer}, ["query"]), cap.web_search)
        registry.add("fetch_web_page", "Open one HTTPS public page.", self._schema({"url": string, "max_chars": integer}, ["url"]), cap.fetch_page)
        registry.add("download_public_asset", "Download one HTTPS asset into the workspace with optional SHA256.", self._schema(
            {"url": string, "filename": string, "sha256": string}, ["url", "filename"]), cap.download)
        registry.add("unpack_public_asset", "Safely unpack a downloaded archive inside the workspace.", self._schema(
            {"path": string, "destination": string}, ["path", "destination"]), cap.unpack)
        registry.add("build_capability", "Build or compile-check an acquired capability bundle in isolation.", self._schema(
            {"directory": string, "argv": {"type": "array", "items": string}}, ["directory"]), cap.build)
        tool_schema = self._schema({"name": string, "source_path": string, "description": string,
            "input_schema": schema_document, "output_schema": schema_document,
            "source_urls": {"type": "array", "items": string},
            "runtime_requirements": {"type": "array", "items": {
                "type": "string", "pattern": "^[A-Za-z0-9_.-]+==[^\\s=]+$"}},
            "manual": manual_schema}, ["name", "source_path", "description", "input_schema", "output_schema"])
        package_spec_schema = {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["algorithm", "perception", "planner", "policy", "model"]},
            "entrypoint": string, "accelerator": {"type": "string", "enum": ["cpu", "cuda"]},
            "timeout_seconds": {"type": "number", "minimum": 0.1, "maximum": 600},
            "runtime_requirements": {"type": "array", "items": string},
            "checkpoint_sha256": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-fA-F]{64}$"}}},
            "required": ["kind", "entrypoint"], "additionalProperties": False}
        package_schema = self._schema({"name": string, "bundle_path": string, "description": string,
            "input_schema": schema_document, "output_schema": schema_document,
            "package_spec": package_spec_schema, "source_urls": {"type": "array", "items": string}},
            ["name", "bundle_path", "description", "input_schema", "output_schema", "package_spec"])
        registry.add("register_tool", "Register an immutable Tool version; call test_tool before it can be bound.", tool_schema, cap.register_tool)
        registry.add("register_capability_package", "Register an acquired bundle for isolated execution.", package_schema, cap.register_package)
        registry.add("revise_tool_manual", "Update a Tool manual using explicit execution evidence.", self._schema(
            {"tool_id": string, "manual": manual_schema, "evidence_paths": {"type": "array", "items": string, "minItems": 1}},
            ["tool_id", "manual", "evidence_paths"]), cap.revise_manual)
        registry.add("test_tool", "Run JSON contract tests against a registered Tool.", self._schema(
            {"tool_id": string, "cases": {"type": "array", "items": test_case_schema, "minItems": 1}}, ["tool_id", "cases"]), cap.test_tool)
        registry.add("register_skill", "Persist a successful reusable Skill.", self._schema(
            {"name": string, "task": string, "controller": string, "tool_ids": {"type": "array", "items": string},
             "evidence_paths": {"type": "array", "items": string}, "evidence": {"type": "object"}},
            ["name", "task", "controller", "tool_ids", "evidence_paths"]), cap.register_skill)
        registry.add("register_experience", "Persist an evidence-backed Experience.", self._schema(
            {"name": string, "summary": string, "applicability": string, "keywords": {"type": "array", "items": string},
             "evidence_paths": {"type": "array", "items": string}},
            ["name", "summary", "applicability", "evidence_paths"]), cap.register_experience)
        registry.add("promote_asset", "Promote a verified asset after successful integration evidence.", self._schema(
            {"asset_id": string, "evidence_paths": {"type": "array", "items": string, "minItems": 1},
             "applicability": {"type": "object"}}, ["asset_id", "evidence_paths"]), cap.promote_asset)
        registry.add("record_gap", "Persist an unresolved capability Gap.", self._schema(
            {"name": string, "task": string, "failure_summary": string, "evidence_paths": {"type": "array", "items": string},
             "attempted_methods": {"type": "array", "items": string}, "missing_capability": string,
             "blocked_reason": string, "next_steps": {"type": "array", "items": string}},
            ["name", "task", "failure_summary", "evidence_paths", "attempted_methods",
             "missing_capability", "blocked_reason", "next_steps"]), cap.record_gap)
        registry.add("run_controller", "Execute the current controller once and return sensor evidence.", self._schema(), self._run_controller)
        registry.add("inspect_execution", "Inspect the latest committed execution evidence.", self._schema(), lambda: self.latest_evidence or {})
        registry.add("view_sensor_artifact", "Read a bounded sensor/evidence artifact path.", self._schema(
            {"path": string, "max_chars": {"type": "integer", "minimum": 1,
                "maximum": 20000}, "offset_bytes": {"type": "integer", "minimum": 0}},
            ["path"]), self._view_artifact)
        registry.add("finish", "Finish only after the task is actually verified.", self._schema({"summary": string}, ["summary"]), self._finish)
        return registry

    def _view_artifact(self, path: str, max_chars: int = 12000,
                       offset_bytes: int = 0):
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
        candidate = candidate.resolve()
        roots = [("workspace", self.workspace.root), ("run", self.root)]
        adapter_root = getattr(self.adapter, "artifact_dir", None)
        if adapter_root:
            roots.append(("adapter", Path(adapter_root).resolve()))
        owner = next(((name, root) for name, root in roots
                      if candidate == root or root in candidate.parents), None)
        if owner is None or not candidate.is_file():
            raise ProtocolError("artifact is outside registered evidence roots")
        relative = f"{owner[0]}://{candidate.relative_to(owner[1]).as_posix()}"
        suffix = candidate.suffix.casefold()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            data = candidate.read_bytes()
            if len(data) > 4 * 1024 * 1024:
                raise ProtocolError("image artifact exceeds the multimodal size limit")
            mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            return {"path": relative, "kind": "image", "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "image_url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}
        if suffix == ".npy":
            import numpy as np
            import cv2
            array = np.load(candidate, mmap_mode="r", allow_pickle=False)
            minimum, maximum = float(np.nanmin(array)), float(np.nanmax(array))
            result = {"path": relative, "kind": "depth" if array.ndim == 2 else "array",
                      "shape": list(array.shape), "dtype": str(array.dtype),
                      "min": minimum, "max": maximum}
            if array.ndim == 2 and maximum > minimum:
                normalized = np.nan_to_num((array - minimum) / (maximum - minimum), nan=0.0)
                preview = cv2.applyColorMap(np.asarray(normalized * 255, dtype=np.uint8), cv2.COLORMAP_TURBO)
                ok, encoded = cv2.imencode(".png", preview)
                if ok:
                    result["image_url"] = "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode()
            return result
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            import cv2
            capture = cv2.VideoCapture(str(candidate)); frames = []
            try:
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                for index in sorted(set([0, max(0, total // 2), max(0, total - 1)])):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, frame = capture.read()
                    if ok:
                        encoded_ok, encoded = cv2.imencode(".jpg", frame)
                        frames.append({"frame": index, "shape": list(frame.shape),
                            "image_url": ("data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode())
                                if encoded_ok else None})
            finally:
                capture.release()
            return {"path": relative, "kind": "video", "frames": frames, "frame_count": total,
                    "image_urls": [row["image_url"] for row in frames if row.get("image_url")]}
        if suffix in {".ply", ".pcd", ".las", ".laz"}:
            data = candidate.read_bytes()[:65536]
            header = data.decode("ascii", errors="ignore")
            points = None
            for pattern in (r"element\s+vertex\s+(\d+)", r"POINTS\s+(\d+)"):
                match = re.search(pattern, header, flags=re.IGNORECASE)
                if match:
                    points = int(match.group(1)); break
            return {"path": relative, "kind": "point_cloud", "format": suffix[1:],
                    "bytes": candidate.stat().st_size, "points": points,
                    "sha256": _file_sha256(candidate)}
        maximum = min(max(int(max_chars), 1), 20000)
        offset = max(int(offset_bytes), 0)
        size = candidate.stat().st_size
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
            temporary = target.with_suffix(f".tmp-{time.time_ns()}")
            try:
                with temporary.open("x") as stream:
                    json.dump(dict(payload), stream, sort_keys=True, default=str)
                    stream.write("\n")
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        summary["artifact_uri"] = f"run://artifacts/tool-results/{target.name}"
        return json.dumps(summary, default=str), summary

    @staticmethod
    def _evidence_success(evidence: Mapping[str, Any]) -> bool:
        report = evidence.get("sensor_report") if isinstance(evidence, Mapping) else None
        execution = evidence.get("execution") if isinstance(evidence, Mapping) else None
        receipt = evidence.get("verification_receipt") if isinstance(evidence, Mapping) else None
        if not isinstance(report, Mapping) or not isinstance(execution, Mapping):
            return False
        if execution.get("completed") is not True or execution.get("error"):
            return False
        if execution.get("sensor_verification_observed") is not True:
            return False
        report_keys = ("sensor_success", "sensor_success_candidate", "success", "verified",
                       "sensor_verification_passed")
        declared = [report.get(key) for key in report_keys if key in report]
        if not declared or not any(value is True for value in declared):
            return False
        return bool(isinstance(receipt, Mapping)
                    and receipt.get("verified") is True
                    and receipt.get("controller_sha256") == evidence.get("controller_sha256")
                    and receipt.get("environment_identity") == evidence.get("environment_identity"))

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
            if not self._evidence_success(evidence):
                errors.append("latest execution has no successful Adapter verification or sensor report")
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
        case_handle = getattr(getattr(self.adapter, "episode", None), "case_handle", None)
        identity = self._execution_identity()
        protocol = self._resume_protocol()
        key_material = {"controller_sha256": controller_sha, "case_handle": case_handle,
                        "environment_identity": identity,
                        "resume_token": (protocol or {}).get("resume_token")}
        execution_key = hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode()).hexdigest()
        for row in self.event_store.events():
            payload = row.get("payload", {})
            validator = getattr(self.adapter, "validate_execution_receipt", None)
            if (protocol and callable(validator) and row.get("kind") == "execution"
                    and payload.get("execution_key") == execution_key
                    and payload.get("environment_identity") == identity
                    and payload.get("resume_token") == protocol.get("resume_token")
                    and validator(payload.get("verification_receipt") or {}) is True):
                candidate = {"reused_committed_execution": True, **payload}
                self.latest_evidence = candidate
                self.state["restored_evidence_unverified"] = False
                return candidate
        self.budget.executions += 1
        result = self.runtime.execute(self.workspace.controller, self.adapter)
        report = self.adapter.sensor_report(result)
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
                    "execution_key": execution_key, "case_handle": case_handle,
                    "environment_identity": identity,
                    "verification_receipt": receipt,
                    "resume_token": (protocol or {}).get("resume_token"),
                    "environment_generation": (protocol or {}).get("environment_generation")}
        evidence_dir = self.root / "evidence"
        if not evidence_dir.exists(): evidence_dir.mkdir(parents=True)
        evidence_path = evidence_dir / f"execution-{self.budget.executions:06d}-{execution_key[:12]}.json"
        temporary = evidence_path.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n")
            temporary.replace(evidence_path)
        finally:
            temporary.unlink(missing_ok=True)
        evidence["artifact_uri"] = f"run://evidence/{evidence_path.name}"
        self.latest_evidence = evidence
        self.state["restored_evidence_unverified"] = False
        self.event_store.commit("execution", evidence)
        return evidence

    def _messages(self, task: str):
        context = self.context_builder.build(task=task, latest_evidence=self.latest_evidence,
                                             retrieved_assets=self.retrieved_assets, state=self.state)
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
            call_pairs = []
            for index, call in enumerate(calls):
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
            for call, normalized in call_pairs:
                name = ""
                try:
                    name = str(call.get("name") or call.get("function", {}).get("name") or "")
                    raw = call.get("arguments") or call.get("function", {}).get("arguments") or "{}"
                    arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    result = self.tools.invoke(name, arguments)
                    if name == "search_assets": self.retrieved_assets = result
                    payload = {"ok": True, "result": result}
                except Exception as exc:
                    payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
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
                content, event_payload = self._bound_tool_result(payload)
                self.messages.append({"role": "tool", "tool_call_id": normalized["id"],
                                      "content": content})
                if multimodal:
                    parts = [{"type": "text", "text": "Selected sensor artifact."}]
                    parts.extend({"type": "image_url", "image_url": {"url": url}} for url in multimodal[:4])
                    self.messages.append({"role": "user", "content": parts})
                self.event_store.commit("tool_result", {"name": name,
                                                        "payload": event_payload})
            self._checkpoint()
        result = {"steps": self.budget.steps, "executions": self.budget.executions,
                  "budget_exhausted": self.budget.exhausted(), "finished": self.state.get("finished", False),
                  "completion_valid": self.state.get("completion_valid", False),
                  "latest_evidence": self.latest_evidence}
        for policy in self.policies:
            after = getattr(policy, "after_run", None)
            if callable(after): after(self, result)
        return result

    def _checkpoint(self):
        snapshot = self.workspace.snapshot()
        save_checkpoint(self.root, {"protocol": "roboforge-checkpoint-v1",
            "steps": self.budget.steps, "executions": self.budget.executions,
            "task": getattr(self, "current_task", None), "elapsed_seconds": self.budget.elapsed(),
            "latest_evidence": self.latest_evidence, "snapshot_id": snapshot.snapshot_id,
            "retrieved_assets": self.retrieved_assets, "state": self.state})


__all__ = ["AgentLoop", "LoopBudget", "ProtocolError"]
