"""Public trial evidence helpers shared by Runtime, CLI and Workspace writers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_first_error(value: Any) -> dict[str, Any] | None:
    """Return the earliest actionable error from an execution/trace tree.

    The extractor deliberately knows no task or parameter names.  It walks
    mappings/lists in recorded order and recognises explicit error objects,
    RPC error strings, and tool_error payloads.
    """
    candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    order = 0

    def add(path: str, error: Mapping[str, Any], *, fallback_type: str = "ToolError") -> None:
        nonlocal order
        step = error.get("step")
        index = error.get("index")
        # Explicit execution step is the primary chronology.  Trace/index order
        # is the fallback for errors that occur before a robot step is assigned.
        has_step = isinstance(step, (int, float)) and not isinstance(step, bool)
        has_index = isinstance(index, (int, float)) and not isinstance(index, bool)
        step_rank = int(step) if has_step else 10**12
        index_rank = int(index) if has_index else 10**12
        # RPC event index is the strongest chronology when supplied.  Step is
        # next; traversal order is only a fallback for terminal errors that do
        # not carry either coordinate.
        coordinate_kind = 0 if has_index else 1 if has_step else 2
        coordinate = index_rank if has_index else step_rank if has_step else order
        candidates.append(((coordinate_kind, coordinate, step_rank, order), {
            "path": path,
            "error_type": str(error.get("type") or fallback_type),
            "message": str(error.get("message") or "tool call failed"),
            **({"api": error.get("tool_id")} if "tool_id" in error else {}),
            **({"step": step} if "step" in error else {}),
        }))
        order += 1

    def walk(node: Any, path: str = "$") -> None:
        if isinstance(node, Mapping):
            errors = node.get("tool_errors")
            if isinstance(errors, (list, tuple)):
                for index, error in enumerate(errors):
                    if isinstance(error, Mapping):
                        add(path + f".tool_errors[{index}]", error)
            if isinstance(node.get("tool_error"), Mapping):
                error = dict(node["tool_error"])
                for field in ("step", "index", "tool_id"):
                    if field not in error and field in node:
                        error[field] = node[field]
                add(path + ".tool_error", error)
            if node.get("error") not in (None, ""):
                error = node["error"]
                if isinstance(error, Mapping):
                    add(path + ".error", error, fallback_type="RuntimeError")
                else:
                    text = str(error)
                    kind, sep, message = text.partition(":")
                    add(path + ".error", {
                        "type": kind.strip() if sep else "RuntimeError",
                        "message": message.strip() if sep else text,
                    }, fallback_type="RuntimeError")
            for key, item in node.items():
                walk(item, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(value)
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


_FAILURE_CLASSES = {
    "controller_failure",
    "task_failure",
    "perception_failure",
    "planning_failure",
    "harness_failure",
    "environment_failure",
    "service_failure",
}
_NON_BUDGET_FAILURES = {
    "harness_failure",
    "environment_failure",
    "service_failure",
}


def _infer_failure_class(
    public: Mapping[str, Any],
    *,
    first_error: Mapping[str, Any] | None,
    controller_status: str,
    environment_status: str,
    task_success: bool | None,
) -> str | None:
    explicit = public.get("failure_class")
    if explicit in _FAILURE_CLASSES:
        return str(explicit)
    if environment_status == "error":
        return "environment_failure"
    if first_error:
        api = str(first_error.get("api") or "").casefold()
        error_type = str(first_error.get("error_type") or "").casefold()
        message = str(first_error.get("message") or "").casefold()
        service_markers = (
            "connection refused",
            "connectionerror",
            "service unavailable",
            "http 5",
            "model service",
            "rpc disconnected",
            "broken pipe",
        )
        if any(marker in error_type or marker in message for marker in service_markers):
            return "service_failure"
        if any(marker in api for marker in ("segment", "sam", "molmo", "perception", "mask")):
            return "perception_failure"
        if any(marker in api for marker in ("grasp", "plan", "ik", "curobo", "trajectory")):
            return "planning_failure"
    if controller_status == "error":
        return "controller_failure"
    if task_success is False:
        return "task_failure"
    return None


def derive_status(
    public: Mapping[str, Any],
    execution_error: str | None = None,
    *,
    failure_class: str | None = None,
    controller_started: bool | None = None,
) -> dict[str, Any]:
    """Separate runner, controller, environment and task outcomes."""
    accounting_requested = failure_class is not None or controller_started is not None
    first = extract_first_error(public)
    if first is None and execution_error:
        text = str(execution_error)
        kind, sep, message = text.partition(":")
        first = {
            "path": "$.execution_error",
            "error_type": kind.strip() if sep else "RuntimeError",
            "message": message.strip() if sep else text,
        }
    controller_error = public.get("controller_error") or first
    termination = str(public.get("controller_termination") or "unknown")
    controller_status = "error" if controller_error or termination not in {"completed", "unknown"} else "completed"
    if execution_error:
        controller_status = "error"
    environment_status = "error" if str(public.get("environment_status", "ok")) == "error" else "ok"
    if first:
        termination = (
            "tool_contract_error"
            if first.get("error_type") == "ToolContractError"
            else "controller_error"
        )
    task_success = public.get("task_success")
    if task_success is None:
        outcome = public.get("independent_task_outcome")
        task_success = outcome.get("verified") if isinstance(outcome, Mapping) else None
    if task_success is None:
        verification = public.get("physical_verification")
        task_success = verification.get("verified") if isinstance(verification, Mapping) else None
    failure = failure_class or _infer_failure_class(
        public,
        first_error=first,
        controller_status=controller_status,
        environment_status=environment_status,
        task_success=task_success if isinstance(task_success, bool) else None,
    )
    if failure is not None and failure not in _FAILURE_CLASSES:
        raise ValueError(f"unsupported failure class: {failure}")
    if controller_started is None:
        trace = public.get("sanitized_runtime_trace") or public.get("sanitized_trace")
        controller_started = (
            termination == "completed"
            or bool(trace)
            or bool(public.get("action_trace"))
        )
    task_budget_consumed = bool(
        controller_started and failure not in _NON_BUDGET_FAILURES
    )
    result = {
        "runner_exit_code": 1 if controller_status == "error" or environment_status == "error" else 0,
        "controller_status": controller_status,
        "environment_status": environment_status,
        "task_success": task_success if isinstance(task_success, bool) else None,
        "termination_reason": termination,
        "first_error": first,
    }
    # Keep the long-standing helper shape for callers that only ask for a
    # status summary.  Trial evidence opts into the accounting fields by
    # passing explicit lifecycle context from ExperimentService.
    if accounting_requested:
        result.update({
            "failure_class": failure,
            "controller_started": bool(controller_started),
            "task_budget_consumed": task_budget_consumed,
        })
    result["trial_status"] = (
        "environment_error" if environment_status == "error"
        else "controller_error" if controller_status == "error"
        else "success" if task_success is True
        else "task_failed" if task_success is False
        else "completed"
    )
    return result
