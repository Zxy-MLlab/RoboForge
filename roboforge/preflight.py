"""Static validation of literal Robot SDK calls before physical execution."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, ValidationError


_UNKNOWN = object()


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _UNKNOWN


def _validate_literal_object(node: ast.AST, schema: Mapping[str, Any]) -> None:
    """Validate known literal fields without guessing dynamic expressions."""
    value = _literal(node)
    if value is not _UNKNOWN:
        Draft202012Validator(dict(schema)).validate(value)
        return
    if not isinstance(node, ast.Dict):
        return
    properties = dict(schema.get("properties") or {})
    additional = schema.get("additionalProperties", True)
    for key_node, value_node in zip(node.keys, node.values):
        if key_node is None:
            continue
        key = _literal(key_node)
        if not isinstance(key, str):
            continue
        if key not in properties:
            if additional is False:
                raise ValueError(f"additional property {key!r} is not allowed")
            continue
        field_value = _literal(value_node)
        if field_value is not _UNKNOWN:
            Draft202012Validator(dict(properties[key])).validate(field_value)


def _literal_dict_keys(node: ast.AST) -> set[str] | None:
    if not isinstance(node, ast.Dict):
        return None
    return {
        key
        for key in (_literal(item) for item in node.keys if item is not None)
        if isinstance(key, str)
    }


def preflight_controller(controller_path: str | Path, *, capability_contracts: Mapping[str, Mapping[str, Any]], sdk_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Conservatively reject deterministic literal contract violations."""
    path = Path(controller_path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {"ok": False, "error_type": type(exc).__name__, "message": str(exc), "source": str(path)}
    errors: list[dict[str, Any]] = []
    checked = 0
    actions = dict((sdk_contract or {}).get("actions") or {})
    verifiers = dict((sdk_contract or {}).get("verifiers") or {})
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "robot":
            continue
        method = node.func.attr
        try:
            if method == "use" and len(node.args) >= 2:
                tool_id = _literal(node.args[0])
                if not isinstance(tool_id, str): continue
                checked += 1
                contract = capability_contracts.get(tool_id)
                if contract is None: raise ValueError(f"unregistered Tool: {tool_id}")
                _validate_literal_object(node.args[1], dict(contract.get("input_schema") or {}))
            elif method == "act" and node.args:
                action_node = node.args[0]
                action = _literal(action_node)
                keys = _literal_dict_keys(action_node)
                if not isinstance(action, dict) and keys is None: continue
                checked += 1
                kind = action.get("type") if isinstance(action, dict) else None
                if kind is None and isinstance(action_node, ast.Dict):
                    for key_node, value_node in zip(action_node.keys, action_node.values):
                        if key_node is not None and _literal(key_node) == "type":
                            kind = _literal(value_node)
                            break
                contract = actions.get(kind)
                if not isinstance(contract, Mapping): raise ValueError(f"unsupported action type {kind!r}")
                present = set(action) if isinstance(action, dict) else keys or set()
                missing = [key for key in contract.get("required", ()) if key not in present]
                if missing: raise ValueError(f"action {kind} missing required fields {missing}")
                alternatives = contract.get("any_of") or ()
                if alternatives and not any(all(key in present for key in option.get("required", ())) for option in alternatives):
                    raise ValueError(f"action {kind} does not satisfy any required field set")
            elif method == "verify" and len(node.args) >= 2:
                verifier = _literal(node.args[0])
                if not isinstance(verifier, str): continue
                checked += 1
                contract = verifiers.get(verifier)
                if not isinstance(contract, Mapping): raise ValueError(f"unknown verifier: {verifier}")
                payload_node = node.args[1]
                if isinstance(payload_node, ast.Dict):
                    keys = {_literal(key) for key in payload_node.keys if key is not None}
                    missing = [key for key in contract.get("required", ()) if key not in keys]
                    if missing: raise ValueError(f"verifier {verifier} missing required fields {missing}")
        except (ValidationError, ValueError) as exc:
            errors.append({"line": getattr(node, "lineno", None), "column": getattr(node, "col_offset", None),
                           "api": f"robot.{method}", "error_type": "ToolContractError",
                           "message": exc.message if isinstance(exc, ValidationError) else str(exc)})
    return {"ok": not errors, "checked_calls": checked, "errors": errors}
