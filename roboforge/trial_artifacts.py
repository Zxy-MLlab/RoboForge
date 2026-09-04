"""Materialize sanitized development evidence as ordinary Workspace files."""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import uuid
import zipfile
from pathlib import Path
from typing import Any

from .evidence import derive_status, extract_first_error
from .models import ExperimentEvidence
from .service import ExperimentService, ProtocolError


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_trajectory_fallback(path: Path, action_trace: list[Any]) -> None:
    data = json.dumps(action_trace, sort_keys=True).encode("utf-8")
    header = str({"descr": "|u1", "fortran_order": False, "shape": (len(data),)})
    padding = 16 - ((10 + len(header) + 1) % 16)
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header) + padding + 1)
    npy += header.encode("ascii") + b" " * padding + b"\n" + data
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("action_trace.npy", npy)


def materialize_trial(
    service: ExperimentService,
    evidence: ExperimentEvidence,
    workspace: str | Path,
    *,
    controller_path: str | Path | None = None,
    runner_stdout: str = "",
    runner_stderr: str = "",
) -> dict[str, Any]:
    """Write one immutable-evidence projection below ``.roboforge/trials``."""
    root = Path(workspace).resolve()
    trial_id = evidence.ref.removeprefix("experiment://")
    target = (root / ".roboforge" / "trials" / trial_id).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("trial materialization escaped workspace") from exc
    target.mkdir(parents=True, exist_ok=True)
    public = evidence.public_dict()
    public_result = dict(evidence.public)
    lifecycle = dict(public_result.get("lifecycle") or derive_status(public_result, evidence.execution_error))
    trace = public_result.get("sanitized_trace") or public_result.get("sanitized_runtime_trace") or []
    first_error = lifecycle.get("first_error") or extract_first_error(public_result)

    frozen = target / "frozen_source"
    frozen.mkdir(exist_ok=True)

    keyframes = target / "keyframes"
    keyframes.mkdir(exist_ok=True)
    rendered = []
    frozen_bundle_files = 0
    bundle_manifest_path = None
    for handle in evidence.artifacts:
        data = service.read_artifact(handle)
        if hashlib.sha256(data).hexdigest() != handle.sha256:
            raise ProtocolError("artifact content digest mismatch")
        raw_name = str(handle.name)
        name = Path(raw_name).name or handle.sha256
        if raw_name == "candidate_bundle/manifest.json":
            destination = target / "candidate_bundle.json"
            bundle_manifest_path = destination
        elif raw_name.startswith("candidate_bundle/files/"):
            relative = Path(raw_name.removeprefix("candidate_bundle/files/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ProtocolError("Candidate Bundle artifact escaped frozen_source")
            destination = (frozen / relative).resolve()
            try:
                destination.relative_to(frozen.resolve())
            except ValueError as exc:
                raise ProtocolError("Candidate Bundle artifact escaped frozen_source") from exc
            frozen_bundle_files += 1
        elif name == "rollout.mp4" or handle.media_type == "video/mp4":
            destination = target / "rollout.mp4"
        elif name == "trajectory.npz":
            destination = target / "trajectory.npz"
        elif handle.media_type.startswith("image/"):
            destination = keyframes / f"{handle.sha256[:12]}-{name}"
        else:
            destination = target / "artifacts" / f"{handle.sha256[:12]}-{name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists(): destination.write_bytes(data)
        rendered.append({**handle.__dict__, "local_path": str(destination)})

    # Legacy evidence created before Candidate Bundles still receives a
    # best-effort single-file snapshot. Canonical trials always take the
    # immutable artifact path above, so later Workspace edits cannot change
    # what is materialized as executed source.
    if frozen_bundle_files == 0 and controller_path is not None:
        source = Path(controller_path).resolve()
        if source.is_file():
            destination = frozen / source.name
            data = source.read_bytes()
            if evidence.controller_sha256 and hashlib.sha256(data).hexdigest() != evidence.controller_sha256:
                raise ProtocolError("Controller changed before trial materialization")
            destination.write_bytes(data)

    action_trace = list(public_result.get("action_trace") or [])
    trajectory_synthetic = False
    if not (target / "trajectory.npz").exists():
        _write_trajectory_fallback(target / "trajectory.npz", action_trace)
        trajectory_synthetic = True
    (target / "action_receipts.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in action_trace),
        encoding="utf-8",
    )
    _json(target / "trace.json", trace)
    _json(target / "first_error.json", first_error or {})
    stderr = runner_stderr or str(public_result.get("controller_stderr") or "")
    (target / "stderr.log").write_text(stderr, encoding="utf-8")
    result = {**lifecycle, "trial_id": trial_id, "evidence_ref": evidence.ref,
              "controller_sha256": evidence.controller_sha256,
              "candidate_bundle_digest": evidence.candidate_bundle_digest,
              "physical_verification": evidence.physical_verification,
              "trace_path": str(target / "trace.json"),
              "first_error_path": str(target / "first_error.json")}
    if first_error:
        result.update({key: first_error.get(key) for key in ("step", "api", "error_type", "message")})
    stdout = runner_stdout or json.dumps(result, indent=2, sort_keys=True) + "\n"
    (target / "stdout.log").write_text(stdout, encoding="utf-8")
    _json(target / "result.json", result)
    _json(target / "manifest.json", {"schema_version": 1, "trial_id": trial_id,
          "candidate_bundle_digest": evidence.candidate_bundle_digest,
          "candidate_bundle_manifest": str(bundle_manifest_path) if bundle_manifest_path else None,
          "frozen_bundle_file_count": frozen_bundle_files,
          "evidence": public, "artifacts": rendered, "paths": {
              "trace": str(target / "trace.json"), "first_error": str(target / "first_error.json"),
              "actions": str(target / "action_receipts.jsonl"), "result": str(target / "result.json"),
              "rollout": str(target / "rollout.mp4") if (target / "rollout.mp4").is_file() else None,
              "trajectory": str(target / "trajectory.npz") if (target / "trajectory.npz").is_file() else None},
          "artifact_availability": {
              "rollout_mp4": (target / "rollout.mp4").is_file(),
              "trajectory_npz": (target / "trajectory.npz").is_file(),
              "trajectory_synthetic": trajectory_synthetic,
              "keyframe_count": len(list(keyframes.iterdir())),
          }})
    return result


def materialize_preflight_failure(
    workspace: str | Path,
    report: dict[str, Any],
    *,
    controller_path: str | Path,
) -> dict[str, Any]:
    """Write a non-physical preflight failure without consuming trial budget."""
    root = Path(workspace).resolve()
    trial_id = f"preflight-{uuid.uuid4().hex[:12]}"
    target = root / ".roboforge" / "trials" / trial_id
    target.mkdir(parents=True, exist_ok=False)
    frozen = target / "frozen_source"; frozen.mkdir()
    source = Path(controller_path).resolve()
    if source.is_file(): shutil.copy2(source, frozen / source.name)
    error = (report.get("errors") or [report])[0]
    first = {"trial_status": "preflight_error", "step": None,
             "api": error.get("api"), "error_type": error.get("error_type", "ToolContractError"),
             "message": error.get("message", "contract preflight failed")}
    _json(target / "trace.json", [{"event": "contract_preflight", **first}])
    _json(target / "first_error.json", first)
    (target / "action_receipts.jsonl").write_text("", encoding="utf-8")
    (target / "stdout.log").write_text("", encoding="utf-8")
    (target / "stderr.log").write_text(first["message"] + "\n", encoding="utf-8")
    result = {**first, "trial_id": trial_id, "runner_exit_code": 2, "controller_status": "not_started",
              "environment_status": "not_started", "task_success": None,
              "termination_reason": "contract_preflight_failed",
              "physical_trial_consumed": False, "trace_path": str(target / "trace.json"),
              "first_error_path": str(target / "first_error.json")}
    # Mirror the CLI's machine-readable stdout so the evidence directory is
    # directly inspectable even when execution stops before a Runtime worker
    # is started.  stderr remains the concise human-readable first error.
    (target / "stdout.log").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _json(target / "result.json", result)
    _json(target / "manifest.json", {"schema_version": 1, "trial_id": trial_id,
          "preflight": report, "result": result})
    (target / "keyframes").mkdir()
    return result
