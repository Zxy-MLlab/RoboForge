from __future__ import annotations

import json
import difflib
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Protocol

from .models import AdapterResult, ArtifactHandle, ExperimentEvidence
from .store import CorruptStore, ExperimentStore


class ProtocolError(RuntimeError):
    pass


class BudgetExhausted(ProtocolError):
    pass


class IndeterminateExperiment(ProtocolError):
    pass


class EmbodiedAdapter(Protocol):
    def begin_execution(self, kind: str) -> None: ...

    def observe(self) -> AdapterResult: ...

    def reset_to_s0(self) -> str: ...

    def execute_controller(
        self,
        *,
        controller_path: Path,
        controller_sha256: str,
        environment_generation: str,
    ) -> AdapterResult: ...

    def validate_receipt(
        self,
        receipt: dict[str, Any],
        *,
        controller_sha256: str,
        environment_generation: str,
    ) -> bool: ...


CrashHook = Callable[[str], None]


class ExperimentService:
    def __init__(
        self,
        root: str | Path,
        adapter: EmbodiedAdapter,
        *,
        max_trials: int = 12,
        max_diagnostics: int = 8,
        crash_hook: CrashHook | None = None,
    ) -> None:
        self.store = ExperimentStore(
            root,
            max_trials=max_trials,
            max_diagnostics=max_diagnostics,
        )
        self.adapter = adapter
        self.crash_hook = crash_hook or (lambda _point: None)

    def status(self) -> dict[str, Any]:
        with self.store.locked():
            self._recover_committed_evidence()
            state = self.store.load_state()
            return {
                "physical_trials": state["physical_trials"],
                "max_trials": state["max_trials"],
                "diagnostics": state["diagnostics"],
                "max_diagnostics": state["max_diagnostics"],
                "latest_evidence": state.get("latest_evidence"),
                "latest_diagnostic_evidence": state.get(
                    "latest_diagnostic_evidence"
                ),
                "latest_physical_evidence": state.get("latest_physical_evidence"),
            }

    def task_info(self) -> dict[str, Any]:
        legacy = getattr(self.adapter, "legacy", None)
        instruction = getattr(legacy, "instruction", None)
        sdk_index = getattr(legacy, "sdk_index", None)
        return {"instruction": str(instruction or ""),
                "robot_interface": sdk_index if isinstance(sdk_index, dict) else {}}

    def reconcile_pending(self, request_id: str, *, disposition: str, note: str) -> dict[str, Any]:
        """External-only reconciliation; never executes or refunds an action."""
        if disposition not in {"abandoned", "confirmed_executed_without_evidence"}:
            raise ProtocolError("unsupported reconciliation disposition")
        if not note.strip(): raise ProtocolError("reconciliation note is required")
        with self.store.locked():
            state = self.store.load_state(); record = state["requests"].get(request_id)
            if not isinstance(record, dict) or record.get("kind") != "physical_trial" or record.get("status") != "pending":
                raise ProtocolError("request is not a pending physical experiment")
            state["requests"][request_id] = {**record, "status": disposition,
                "reconciliation_note": note}
            self.store.save_state(state)
            return dict(state["requests"][request_id])

    def experiment_spine(self) -> dict[str, Any]:
        """Return the minimal factual state that context condensation cannot drop."""
        with self.store.locked():
            self._recover_committed_evidence()
            state = self.store.load_state()

            def evidence_body(ref: str | None) -> dict[str, Any] | None:
                if not ref:
                    return None
                metadata = state["evidence"].get(ref)
                if not isinstance(metadata, dict):
                    raise CorruptStore("latest experiment reference is not indexed")
                body = self.store.load_evidence_file(
                    self.store.evidence_dir / metadata["file"]
                )
                if body["evidence_sha256"] != metadata["sha256"]:
                    raise CorruptStore("latest experiment digest mismatch")
                return body

            latest_physical = evidence_body(state.get("latest_physical_evidence"))
            latest_diagnostic = evidence_body(
                state.get("latest_diagnostic_evidence")
            )
            pending = sorted(
                (
                    {
                        "kind": str(record.get("kind")),
                        "index": int(record.get("index")),
                    }
                    for record in state["requests"].values()
                    if isinstance(record, dict)
                    and record.get("status") == "pending"
                    and record.get("kind") in {"diagnostic", "physical_trial"}
                    and isinstance(record.get("index"), int)
                ),
                key=lambda item: (item["kind"], item["index"]),
            )
            return {
                "schema_version": 1,
                "physical_trials": int(state["physical_trials"]),
                "max_physical_trials": int(state["max_trials"]),
                "remaining_physical_trials": max(
                    0, int(state["max_trials"]) - int(state["physical_trials"])
                ),
                "diagnostics": int(state["diagnostics"]),
                "max_diagnostics": int(state["max_diagnostics"]),
                "remaining_diagnostics": max(
                    0, int(state["max_diagnostics"]) - int(state["diagnostics"])
                ),
                "latest_evidence": state.get("latest_evidence"),
                "latest_diagnostic_evidence": state.get(
                    "latest_diagnostic_evidence"
                ),
                "latest_physical_evidence": state.get("latest_physical_evidence"),
                "latest_controller_sha256": (
                    latest_physical.get("controller_sha256")
                    if latest_physical else None
                ),
                "physical_verification": (
                    latest_physical.get("physical_verification")
                    if latest_physical else None
                ),
                "latest_physical_execution_error": (
                    latest_physical.get("execution_error")
                    if latest_physical else None
                ),
                "latest_diagnostic_execution_error": (
                    latest_diagnostic.get("execution_error")
                    if latest_diagnostic else None
                ),
                "indeterminate_attempts": pending,
            }

    def observe(self, *, request_id: str) -> ExperimentEvidence:
        with self.store.locked():
            self._recover_committed_evidence()
            existing = self._existing_or_pending(request_id, expected_kind="diagnostic")
            if isinstance(existing, ExperimentEvidence):
                return existing
            if existing == "pending":
                raise IndeterminateExperiment(
                    "diagnostic was interrupted before durable evidence commit"
                )
            state = self.store.load_state()
            if state["diagnostics"] >= state["max_diagnostics"]:
                raise BudgetExhausted("diagnostic budget exhausted")
            index = state["diagnostics"] + 1
            state["diagnostics"] = index
            state["requests"][request_id] = {
                "kind": "diagnostic",
                "status": "pending",
                "index": index,
            }
            self.store.save_state(state)

        self.crash_hook("diagnostic_reserved")
        self.adapter.begin_execution("diagnostic")
        result = self.adapter.observe()
        self.crash_hook("diagnostic_executed")
        return self._commit(
            request_id=request_id,
            kind="diagnostic",
            index=index,
            controller_sha256=None,
            environment_generation=None,
            intent=None,
            result=result,
            verified=None,
            execution_error=None,
        )

    def run_controller(
        self,
        *,
        request_id: str,
        controller_path: str | Path,
        intent: str,
        assets_used: list[str] | None = None,
    ) -> ExperimentEvidence:
        path = Path(controller_path).resolve()
        if not intent.strip():
            raise ProtocolError("physical experiment intent must be non-empty")
        try:
            source = path.read_bytes()
        except OSError as exc:
            raise ProtocolError("Controller is not readable") from exc

        with self.store.locked():
            self._recover_committed_evidence()
            existing = self._existing_or_pending(
                request_id,
                expected_kind="physical_trial",
            )
            if isinstance(existing, ExperimentEvidence):
                return existing
            if existing == "pending":
                raise IndeterminateExperiment(
                    "physical experiment was reserved but has no committed evidence; "
                    "it will not be rerun automatically"
                )
            state = self.store.load_state()
            unresolved = [
                key for key, record in state["requests"].items()
                if isinstance(record, dict)
                and record.get("kind") == "physical_trial"
                and record.get("status") == "pending"
            ]
            if unresolved:
                raise IndeterminateExperiment(
                    "a prior physical experiment is unresolved; no new physical "
                    "action may execute until external reconciliation: " + ", ".join(sorted(unresolved))
                )
            if state["physical_trials"] >= state["max_trials"]:
                raise BudgetExhausted("physical trial budget exhausted")
            controller_sha = self.store.put_controller(source)
            index = state["physical_trials"] + 1
            state["physical_trials"] = index
            state["requests"][request_id] = {
                "kind": "physical_trial",
                "status": "pending",
                "index": index,
                "controller_sha256": controller_sha,
            }
            self.store.save_state(state)

        self.crash_hook("physical_reserved")
        self.adapter.begin_execution("physical_trial")
        generation = self.adapter.reset_to_s0()
        self.crash_hook("physical_reset")
        try:
            result = self.adapter.execute_controller(
                controller_path=self.store.controller_dir / f"{controller_sha}.py",
                controller_sha256=controller_sha,
                environment_generation=generation,
            )
            receipt = result.private_receipt
            verified = bool(
                receipt is not None
                and self.adapter.validate_receipt(
                    receipt,
                    controller_sha256=controller_sha,
                    environment_generation=generation,
                )
            )
            execution_error = None
            if receipt is not None:
                self.store.put_private_receipt(request_id, receipt)
        except Exception as exc:
            result = AdapterResult(public={"execution_status": "error"})
            verified = False
            execution_error = f"{type(exc).__name__}: {exc}"
        self.crash_hook("physical_executed")
        return self._commit(
            request_id=request_id,
            kind="physical_trial",
            index=index,
            controller_sha256=controller_sha,
            environment_generation=generation,
            intent=intent,
            result=result,
            verified=verified,
            execution_error=execution_error,
            assets_used=tuple(assets_used or ()),
        )

    def inspect_trial(self, ref: str) -> ExperimentEvidence:
        with self.store.locked():
            self._recover_committed_evidence()
            state = self.store.load_state()
            metadata = state["evidence"].get(ref)
            if not isinstance(metadata, dict):
                raise ProtocolError("unknown experiment reference")
            body = self.store.load_evidence_file(
                self.store.evidence_dir / metadata["file"]
            )
            if body["evidence_sha256"] != metadata["sha256"]:
                raise CorruptStore("evidence index digest mismatch")
            return self._evidence_from_body(body)

    def list_trials(self) -> list[ExperimentEvidence]:
        with self.store.locked():
            self._recover_committed_evidence()
            state = self.store.load_state()
            refs = sorted(state["evidence"], key=lambda ref: state["evidence"][ref]["order"])
        return [self.inspect_trial(ref) for ref in refs]

    def read_artifact(self, handle: ArtifactHandle) -> bytes:
        return self.store.read_artifact(handle.uri, handle.sha256)

    def compare_trials(self, first_ref: str, second_ref: str) -> dict[str, Any]:
        first_evidence = self.inspect_trial(first_ref)
        second_evidence = self.inspect_trial(second_ref)
        first = first_evidence.public_dict()
        second = second_evidence.public_dict()
        ignored = {"ref", "request_id", "evidence_sha256"}
        result = {
            "first": first_ref,
            "second": second_ref,
            "differences": self._diff(
                {key: value for key, value in first.items() if key not in ignored},
                {key: value for key, value in second.items() if key not in ignored},
            ),
        }
        if first_evidence.controller_sha256 and second_evidence.controller_sha256:
            before = (self.store.controller_dir / f"{first_evidence.controller_sha256}.py").read_text(errors="replace")
            after = (self.store.controller_dir / f"{second_evidence.controller_sha256}.py").read_text(errors="replace")
            result["controller_diff"] = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                fromfile=first_evidence.controller_sha256, tofile=second_evidence.controller_sha256))
        return result

    def _commit(
        self,
        *,
        request_id: str,
        kind: str,
        index: int,
        controller_sha256: str | None,
        environment_generation: str | None,
        intent: str | None,
        result: AdapterResult,
        verified: bool | None,
        execution_error: str | None,
        assets_used: tuple[str, ...] = (),
    ) -> ExperimentEvidence:
        self._assert_public_projection(result.public)
        handles = tuple(
            ArtifactHandle(**self.store.put_artifact(
                name=artifact.name,
                media_type=artifact.media_type,
                data=artifact.data,
            ))
            for artifact in result.artifacts
        )
        self.crash_hook("artifacts_durable")
        ref = (
            f"experiment://diagnostic-{index:06d}"
            if kind == "diagnostic"
            else f"experiment://physical-{index:06d}"
        )
        evidence = ExperimentEvidence(
            schema_version=1,
            ref=ref,
            execution_kind=kind,  # type: ignore[arg-type]
            request_id=request_id,
            diagnostic_index=index if kind == "diagnostic" else None,
            physical_trial_index=index if kind == "physical_trial" else None,
            environment_generation=environment_generation,
            controller_sha256=controller_sha256,
            intent=intent,
            public=result.public,
            assets_used=assets_used,
            artifacts=handles,
            physical_verification=(
                {"verified": bool(verified)} if kind == "physical_trial" else None
            ),
            execution_error=execution_error,
        )
        unsigned = evidence.public_dict()
        unsigned.pop("evidence_sha256")
        _, digest = self.store.put_evidence(unsigned)
        self.crash_hook("evidence_durable")

        with self.store.locked():
            state = self.store.load_state()
            safe_name = ref.replace("://", "-").replace("/", "-") + ".json"
            order = len(state["evidence"]) + 1
            state["evidence"][ref] = {
                "sha256": digest,
                "file": safe_name,
                "kind": kind,
                "order": order,
            }
            state["requests"][request_id] = {
                **state["requests"][request_id],
                "status": "committed",
                "ref": ref,
            }
            state["latest_evidence"] = ref
            if kind == "diagnostic":
                state["latest_diagnostic_evidence"] = ref
            else:
                state["latest_physical_evidence"] = ref
            self.store.save_state(state)
        self.crash_hook("index_durable")
        return self.inspect_trial(ref)

    def _existing_or_pending(
        self,
        request_id: str,
        *,
        expected_kind: str,
    ) -> ExperimentEvidence | str | None:
        state = self.store.load_state()
        record = state["requests"].get(request_id)
        if record is None:
            return None
        if record.get("kind") != expected_kind:
            raise ProtocolError("request id was already used for another operation")
        if record.get("status") == "committed":
            ref = record.get("ref")
            metadata = state["evidence"].get(ref)
            if not isinstance(metadata, dict):
                raise CorruptStore("committed request has no evidence index")
            body = self.store.load_evidence_file(
                self.store.evidence_dir / metadata["file"]
            )
            return self._evidence_from_body(body)
        return "pending"

    def _recover_committed_evidence(self) -> None:
        state = self.store.load_state()
        changed = False
        for request_id, record in list(state["requests"].items()):
            if record.get("status") != "pending":
                continue
            body = self.store.find_evidence_by_request(request_id)
            if body is None:
                continue
            ref = body["ref"]
            safe_name = ref.replace("://", "-").replace("/", "-") + ".json"
            state["evidence"].setdefault(
                ref,
                {
                    "sha256": body["evidence_sha256"],
                    "file": safe_name,
                    "kind": body["execution_kind"],
                    "order": len(state["evidence"]) + 1,
                },
            )
            state["requests"][request_id] = {
                **record,
                "status": "committed",
                "ref": ref,
            }
            state["latest_evidence"] = ref
            if body["execution_kind"] == "diagnostic":
                state["latest_diagnostic_evidence"] = ref
            else:
                state["latest_physical_evidence"] = ref
            changed = True
        if changed:
            self.store.save_state(state)

    @staticmethod
    def _assert_public_projection(value: Any, path: str = "public") -> None:
        if isinstance(value, dict):
            forbidden_keys = {
                "host_path",
                "filesystem_path",
                "verification_receipt",
                "resume_token",
                "environment_identity",
            }
            overlap = forbidden_keys.intersection(value)
            if overlap:
                raise ProtocolError(
                    f"public projection contains forbidden fields at {path}: "
                    f"{sorted(overlap)}"
                )
            for key, item in value.items():
                ExperimentService._assert_public_projection(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                ExperimentService._assert_public_projection(item, f"{path}[{index}]")
        elif isinstance(value, str) and (
            value.startswith("/root/")
            or value.startswith("/tmp/")
            or value.startswith("file://")
        ):
            raise ProtocolError(f"public projection exposes a host path at {path}")

    @staticmethod
    def _evidence_from_body(body: dict[str, Any]) -> ExperimentEvidence:
        allowed = {item.name for item in fields(ExperimentEvidence)}
        value = {key: item for key, item in body.items() if key in allowed}
        for optional_name in (
            "diagnostic_index",
            "physical_trial_index",
            "environment_generation",
            "controller_sha256",
            "intent",
            "assets_used",
            "physical_verification",
            "execution_error",
        ):
            value.setdefault(optional_name, () if optional_name == "assets_used" else None)
        value["assets_used"] = tuple(value.get("assets_used") or ())
        value["artifacts"] = tuple(
            ArtifactHandle(**artifact) for artifact in value.get("artifacts", ())
        )
        return ExperimentEvidence(**value)

    @staticmethod
    def _diff(first: Any, second: Any, path: str = "") -> list[dict[str, Any]]:
        if isinstance(first, dict) and isinstance(second, dict):
            rows: list[dict[str, Any]] = []
            for key in sorted(set(first) | set(second)):
                child = f"{path}.{key}" if path else key
                if key not in first:
                    rows.append({"path": child, "before": None, "after": second[key]})
                elif key not in second:
                    rows.append({"path": child, "before": first[key], "after": None})
                else:
                    rows.extend(ExperimentService._diff(first[key], second[key], child))
            return rows
        if first != second:
            return [{"path": path, "before": first, "after": second}]
        return []
