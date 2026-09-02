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
    def walk(node: Any, path: str = "$") -> dict[str, Any] | None:
        if isinstance(node, Mapping):
            errors = node.get("tool_errors")
            if (
                isinstance(errors, (list, tuple))
                and errors
                and isinstance(errors[0], Mapping)
            ):
                error = errors[0]
                return {
                    "path": path + ".tool_errors[0]",
                    "error_type": str(error.get("type") or "ToolError"),
                    "message": str(error.get("message") or "tool call failed"),
                    "api": error.get("tool_id"),
                    "step": error.get("step"),
                }
            if isinstance(node.get("tool_error"), Mapping):
                error = node["tool_error"]
                return {"path": path + ".tool_error", "error_type": str(error.get("type") or "ToolError"),
                        "message": str(error.get("message") or "tool call failed")}
            if node.get("error") not in (None, ""):
                error = node["error"]
                if isinstance(error, Mapping):
                    return {"path": path + ".error", "error_type": str(error.get("type") or "RuntimeError"),
                            "message": str(error.get("message") or "runtime error")}
                text = str(error)
                kind, sep, message = text.partition(":")
                return {"path": path + ".error", "error_type": kind.strip() if sep else "RuntimeError",
                        "message": message.strip() if sep else text}
            for key, item in node.items():
                found = walk(item, f"{path}.{key}")
                if found:
                    return found
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                found = walk(item, f"{path}[{index}]")
                if found:
                    return found
        return None

    return walk(value)


def derive_status(public: Mapping[str, Any], execution_error: str | None = None) -> dict[str, Any]:
    """Separate runner, controller, environment and task outcomes."""
    first = extract_first_error(public)
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
    result = {
        "runner_exit_code": 1 if controller_status == "error" or environment_status == "error" else 0,
        "controller_status": controller_status,
        "environment_status": environment_status,
        "task_success": task_success if isinstance(task_success, bool) else None,
        "termination_reason": termination,
        "first_error": first,
    }
    result["trial_status"] = (
        "environment_error" if environment_status == "error"
        else "controller_error" if controller_status == "error"
        else "success" if task_success is True
        else "task_failed" if task_success is False
        else "completed"
    )
    return result
