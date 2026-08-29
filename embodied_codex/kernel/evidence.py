"""Explicit evidence views at the model/Harness trust boundary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .embodied_state import build_transition, normalize_entity


_PUBLIC_RESULT_FIELDS = (
    "reached", "target_xyz", "target_quaternion_xyzw", "eef_before", "eef_after",
    "final_position_error_m", "final_orientation_error_rad",
    "gripper", "gripper_width_m", "action_axis", "action_axis_frame", "action_frame_axis", "target_source", "verified", "sensor_only",
    "verifier_error", "reason", "criterion",
)
_PUBLIC_VERIFIER_FIELDS = (
    "verified", "sensor_only", "verifier_error", "reason", "criterion",
    "target_xy_error_m", "vertical_offset_m", "source_vacated",
    "nearest_source_detection_m", "object_to_eef_distance_m", "gripper_width_m",
    "retained_width", "object_count", "object", "target", "eef_world_xyz",
    "source_anchor_world_xyz", "support_overlap_fraction", "object_coverage_fraction",
    "target_coverage_fraction", "support_overlap_normalization", "target_geometry_source",
    "minimum_support_overlap", "support_gap_m", "support_gap_range_m",
    "center_inside_target_bounds", "target_count", "target_anchor_world_xyz",
    "target_association_rank", "maximum_target_association_rank", "support_height_source",
    "support_plane_height_m", "target_surface_height_error_m",
    "maximum_target_surface_height_error_m", "selected_object_source_displacement_m",
    "geometric_source_vacated", "source_transport_verified", "source_vacancy_method",
)
_PRIVATE_KEYS = {"reward", "done", "env.check_success", "env_check_success",
                 "check_success", "hidden_evaluator", "hidden evaluator", "evaluator"}


def is_routing_reference(value: Any) -> bool:
    """Return whether a string is an opaque URI used to route evidence/tools."""
    return isinstance(value, str) and value.startswith(("artifact://", "evidence://", "run://"))


def _bounded_public(value: Any, *, depth: int = 0, max_items: int = 24):
    """Bound an already-public RPC value without interpreting its meaning."""
    if isinstance(value, str):
        # Public evidence may contain opaque artifact URIs, but never relay a
        # host filesystem path across the model boundary.
        if value.startswith(("/", "\\")) or (len(value) > 2 and value[1] == ":"
                                               and value[2:3] in ("/", "\\")):
            return "<host path omitted>"
        if is_routing_reference(value):
            return value
        return value if len(value) <= 512 else value[:512] + "..."
    if isinstance(value, Mapping):
        if depth >= 6:
            return "<nested value omitted>"
        items = [(key, item) for key, item in value.items()
                 if str(key).lower() not in _PRIVATE_KEYS][:max_items]
        result = {str(key): _bounded_public(item, depth=depth + 1,
                                             max_items=max_items)
                  for key, item in items}
        if len(value) > max_items:
            result["_truncated_fields"] = len(value) - max_items
        return result
    if isinstance(value, (list, tuple)):
        if depth >= 6:
            return "<nested value omitted>"
        values = list(value)
        if len(values) <= max_items:
            return [_bounded_public(item, depth=depth + 1, max_items=max_items)
                    for item in values]
        head_count = max(1, max_items // 4)
        tail_count = max(1, max_items - head_count)
        return {"total_count": len(values),
                "head": [_bounded_public(item, depth=depth + 1, max_items=max_items)
                          for item in values[:head_count]],
                "tail": [_bounded_public(item, depth=depth + 1, max_items=max_items)
                          for item in values[-tail_count:]],
                "omitted_count": len(values) - head_count - tail_count}
    return value


def _artifact_handles(value: Any, output: set[str]):
    if isinstance(value, Mapping):
        for item in value.values():
            _artifact_handles(item, output)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _artifact_handles(item, output)
    elif isinstance(value, str) and value.startswith("artifact://"):
        output.add(value)


def build_execution_digest(execution: Mapping[str, Any], *,
                           controller_sha256: str | None = None,
                           diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Create a compact, task-agnostic digest from the public RPC boundary."""
    digest: dict[str, Any] = {
        "execution": {"completed": execution.get("completed") is True,
                       "error": _bounded_public(execution.get("error")),
                       "controller_sha256": controller_sha256 or execution.get("program_sha256")},
        "controller_result": _bounded_public(execution.get("result")),
        "tool_calls": [], "actions": [], "verifications": [], "entities": [],
        "artifacts": {"rgb": [], "depth": [], "trace": None, "rollout": None},
    }
    handles: set[str] = set()
    for event in execution.get("rpc_events") or []:
        if not isinstance(event, Mapping):
            continue
        method = str(event.get("method") or "")
        arguments = event.get("arguments") or {}
        result = event.get("result")
        if method == "use":
            tool_id = str(arguments.get("tool_id") or "")
            output = result.get("result") if isinstance(result, Mapping) else result
            status = "failure" if event.get("error") or (
                isinstance(output, Mapping) and output.get("tool_error")) else "success"
            call = {"tool_id": tool_id, "status": status,
                "input_summary": _bounded_public(arguments.get("payload") or {}),
                "output_summary": _bounded_public(output)}
            if event.get("error"):
                call["error"] = _bounded_public(event.get("error"))
            digest["tool_calls"].append(call)
            # Entity normalization is an Adapter/capability boundary concern.
            # Core consumes only the explicit canonical projection attached to
            # the public RPC event; it never interprets a native Tool schema.
            entities = event.get("entities")
            if isinstance(entities, list):
                for candidate in entities:
                    if isinstance(candidate, Mapping):
                        try:
                            digest["entities"].append(_bounded_public(
                                normalize_entity(candidate).as_dict()))
                        except (TypeError, ValueError):
                            continue
        elif method == "act":
            public_result = result if isinstance(result, Mapping) else {}
            action = {"index": len(digest["actions"]) + 1,
                "type": public_result.get("type") or arguments.get("action", {}).get("type"),
                "requested": _bounded_public(arguments.get("action") or {}),
                "result": {key: _bounded_public(public_result.get(key))
                           for key in _PUBLIC_RESULT_FIELDS if key in public_result}}
            transition = build_transition(
                before=event.get("state_before"), requested_action=arguments.get("action") or {},
                achieved_action=public_result, after=event.get("state_after"))
            action["transition"] = _bounded_public(transition.as_dict())
            if event.get("error"):
                action["result"]["error"] = _bounded_public(event.get("error"))
            digest["actions"].append(action)
        elif method == "verify":
            public_result = result if isinstance(result, Mapping) else {}
            verification = {"verifier": arguments.get("verifier"),
                **{key: _bounded_public(public_result.get(key))
                   for key in _PUBLIC_VERIFIER_FIELDS if key in public_result}}
            if event.get("error"):
                verification["error"] = _bounded_public(event.get("error"))
            digest["verifications"].append(verification)
        _artifact_handles(arguments, handles)
        _artifact_handles(result, handles)
    _artifact_handles(diagnostics or {}, handles)
    for handle in sorted(handles):
        name = handle.rsplit("/", 1)[-1].lower()
        if "depth" in name:
            digest["artifacts"]["depth"].append(handle)
        elif "trace" in name:
            digest["artifacts"]["trace"] = handle
        elif "rollout" in name or name.endswith((".mp4", ".webm")):
            digest["artifacts"]["rollout"] = handle
        else:
            digest["artifacts"]["rgb"].append(handle)
    if isinstance(diagnostics, Mapping):
        for key in ("trace_path", "rollout_path"):
            value = diagnostics.get(key)
            if isinstance(value, str) and value.startswith("artifact://"):
                digest["artifacts"]["trace" if key == "trace_path" else "rollout"] = value
    return digest


@dataclass(frozen=True)
class AgentEvidence:
    """Bounded diagnostic evidence that may be sent to the coding model."""

    execution: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    evidence_ref: str | None = None
    digest: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = {"execution": dict(self.execution),
                  "diagnostics": dict(self.diagnostics),
                  "digest": dict(self.digest)}
        if self.evidence_ref:
            result["evidence_ref"] = self.evidence_ref
        return result

    @classmethod
    def from_execution(cls, execution: Mapping[str, Any],
                       diagnostics: Mapping[str, Any] | None = None,
                       evidence_ref: str | None = None,
                       *, digest: Mapping[str, Any] | None = None):
        return cls(execution={"completed": execution.get("completed") is True,
                              "error": execution.get("error")},
                   diagnostics=dict(diagnostics or {}),
                   digest=dict(digest if digest is not None else build_execution_digest(
                       execution, diagnostics=diagnostics)),
                   evidence_ref=evidence_ref)


@dataclass(frozen=True)
class HarnessMetadata:
    """Receipt and recovery metadata retained by the Harness, never the model."""

    artifact_uri: str | None
    artifact_sha256: str | None
    controller_sha256: str | None
    execution_key: str | None
    environment_identity: Mapping[str, Any]
    verification_receipt: Mapping[str, Any]
    resume_token: str | None

    @classmethod
    def from_evidence(cls, evidence: Mapping[str, Any]):
        return cls(artifact_uri=evidence.get("artifact_uri"),
                   artifact_sha256=evidence.get("artifact_sha256"),
                   controller_sha256=evidence.get("controller_sha256"),
                   execution_key=evidence.get("execution_key"),
                   environment_identity=dict(evidence.get("environment_identity") or {}),
                   verification_receipt=dict(evidence.get("verification_receipt") or {}),
                   resume_token=evidence.get("resume_token"))

    def as_reference(self, *, summary: Mapping[str, Any]) -> dict[str, Any]:
        return {"artifact_uri": self.artifact_uri,
                "artifact_sha256": self.artifact_sha256,
                "controller_sha256": self.controller_sha256,
                "execution_key": self.execution_key,
                "environment_identity": dict(self.environment_identity),
                "verification_receipt": dict(self.verification_receipt),
                "resume_token": self.resume_token,
                "summary": dict(summary)}


__all__ = ["AgentEvidence", "HarnessMetadata", "build_execution_digest",
           "is_routing_reference"]
