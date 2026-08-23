"""Immutable typed Stage Nodes compiled independently of any environment."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

from .errors import NodeCompileError, NodeRuntimeError
from .runtime import NodeRuntime


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_FIELD = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_FORBIDDEN_TEXT = {
    "check_success", "bddl", "mujoco", "sim.", "task_id", "state_id",
    "object_id", "evaluator",
}
_FORBIDDEN_NAMES = {"reward", "done", "terminated", "truncated", "success"}
_ALLOWED_IMPORTS = {"math", "statistics"}


def _direct_returns(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Return]:
    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None: self.items: list[ast.Return] = []
        def visit_Return(self, node: ast.Return) -> None: self.items.append(node)  # noqa: N802
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None: return  # noqa: N802
        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None: return  # noqa: N802
        def visit_Lambda(self, node: ast.Lambda) -> None: return  # noqa: N802
    visitor = Visitor()
    for statement in function.body: visitor.visit(statement)
    return visitor.items


def compile_node_source(
    source: str, *, provides_by_outcome: Mapping[str, list[str]],
    checkpoint_outcomes: list[str],
) -> dict[str, Any]:
    violations: list[str] = []
    lowered = source.casefold()
    violations.extend(
        f"forbidden_term:{term}" for term in _FORBIDDEN_TEXT if term in lowered
    )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"eligible": False, "violations": [f"syntax:{exc.msg}"]}
    functions = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "run_stage"]
    if len(functions) != 1 or [arg.arg for arg in functions[0].args.args] != ["adapter", "context"]:
        violations.append("run_stage_signature_must_be_adapter_context")
        return {"eligible": False, "violations": sorted(set(violations))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                f"forbidden_import:{alias.name}" for alias in node.names
                if alias.name not in _ALLOWED_IMPORTS
            )
        elif isinstance(node, ast.ImportFrom) and str(node.module or "") not in _ALLOWED_IMPORTS:
            violations.append(f"forbidden_import:{node.module}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            violations.append(f"private_attribute:{node.attr}")
        elif isinstance(node, ast.Name) and node.id.casefold() in _FORBIDDEN_NAMES:
            violations.append(f"forbidden_name:{node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.casefold() in _FORBIDDEN_NAMES:
            violations.append(f"forbidden_attribute:{node.attr}")
    adapter_calls = [
        node for node in ast.walk(functions[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "adapter"
    ]
    allowed_methods = {"instruction", "sense", "act", "use", "verify", "record"}
    violations.extend(
        f"unknown_adapter_method:{node.func.attr}" for node in adapter_calls
        if node.func.attr not in allowed_methods
    )
    verifier_calls = [node for node in adapter_calls if node.func.attr == "verify"]
    action_calls = [node for node in adapter_calls if node.func.attr == "act"]
    if checkpoint_outcomes and (not verifier_calls or action_calls):
        violations.append("checkpoint_node_must_verify_and_must_not_act")

    implemented: set[str] = set()
    for index, returned in enumerate(_direct_returns(functions[0]), 1):
        value = returned.value
        if not isinstance(value, ast.Dict):
            violations.append(f"return_{index}:result_must_be_object_literal"); continue
        if any(not isinstance(key, ast.Constant) or not isinstance(key.value, str)
               for key in value.keys):
            violations.append(f"return_{index}:keys_must_be_literal"); continue
        fields = {str(key.value): item for key, item in zip(value.keys, value.values)}
        if set(fields) != {"outcome", "updates"}:
            violations.append(f"return_{index}:requires_outcome_updates"); continue
        outcome_node = fields["outcome"]
        if not isinstance(outcome_node, ast.Constant) or not isinstance(outcome_node.value, str):
            violations.append(f"return_{index}:outcome_must_be_literal"); continue
        outcome = str(outcome_node.value)
        if outcome not in provides_by_outcome:
            violations.append(f"return_{index}:undeclared_outcome:{outcome}"); continue
        updates = fields["updates"]
        if not isinstance(updates, ast.Dict) or any(
            not isinstance(key, ast.Constant) or not isinstance(key.value, str)
            for key in getattr(updates, "keys", [])
        ):
            violations.append(f"return_{index}:updates_must_be_object_literal"); continue
        actual = {str(key.value) for key in updates.keys}
        expected = set(provides_by_outcome[outcome])
        if actual != expected:
            violations.append(
                f"return_{index}:updates_mismatch:{outcome}:"
                f"expected={sorted(expected)}:actual={sorted(actual)}"
            ); continue
        implemented.add(outcome)
    missing = set(provides_by_outcome) - implemented
    if missing: violations.append(f"outcomes_not_implemented:{sorted(missing)}")
    return {
        "eligible": not violations, "violations": sorted(set(violations)),
        "implemented_outcomes": sorted(implemented),
        "adapter_methods": sorted({node.func.attr for node in adapter_calls}),
    }


class NodeStore:
    def __init__(
        self, root: str | Path, *, python: str | Path | None = None,
        timeout_seconds: float = 300,
    ) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.runtime = NodeRuntime(python=python, timeout_seconds=timeout_seconds)

    def create(
        self, *, name: str, kind: str, description: str, source: str,
        requires: list[str], provides_by_outcome: Mapping[str, list[str]],
        checkpoint_outcomes: list[str] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(name) or not _FIELD.fullmatch(kind):
            raise NodeCompileError("invalid node name or kind")
        required = list(dict.fromkeys(map(str, requires)))
        provides = {str(outcome): list(dict.fromkeys(map(str, fields)))
                    for outcome, fields in provides_by_outcome.items()}
        checkpoints = list(dict.fromkeys(map(str, checkpoint_outcomes or [])))
        if not provides or any(not _FIELD.fullmatch(x) for x in required):
            raise NodeCompileError("invalid node context contract")
        if any(not _FIELD.fullmatch(outcome) for outcome in provides):
            raise NodeCompileError("invalid outcome")
        if any(not _FIELD.fullmatch(field) for fields in provides.values() for field in fields):
            raise NodeCompileError("invalid provided field")
        if set(checkpoints) - set(provides):
            raise NodeCompileError("checkpoint outcome is undeclared")
        compiled = compile_node_source(
            source, provides_by_outcome=provides,
            checkpoint_outcomes=checkpoints,
        )
        if not compiled["eligible"]:
            raise NodeCompileError(f"node compile failed: {compiled['violations']}")
        family = self.root / name
        versions = [int(path.name[1:]) for path in family.glob("v[0-9]*")
                    if path.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"; destination.mkdir(parents=True)
        source_path = destination / "node.py"; source_path.write_text(source)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest = {
            "protocol": "standalone-embodied-stage-node-v1",
            "node_id": f"{name}:v{version:03d}", "name": name,
            "version": version, "kind": kind, "description": description,
            "requires": required, "provides_by_outcome": provides,
            "outcomes": sorted(provides), "checkpoint_outcomes": checkpoints,
            "source_sha256": digest, "compile_report": compiled,
            "created_unix": time.time(), "privileged_state_used": False,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {"node_id": manifest["node_id"], "source_sha256": digest,
                "compile_report": compiled}

    def resolve(self, node_id: str) -> Path:
        name, sep, version = node_id.partition(":")
        if not sep or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise NodeCompileError("invalid node_id")
        path = (self.root / name / version).resolve()
        if self.root not in path.parents or not (path / "manifest.json").is_file():
            raise FileNotFoundError(node_id)
        return path

    def inspect(self, node_id: str) -> dict[str, Any]:
        path = self.resolve(node_id); manifest = json.loads((path / "manifest.json").read_text())
        source = (path / "node.py").read_text()
        if hashlib.sha256(source.encode()).hexdigest() != manifest["source_sha256"]:
            raise NodeCompileError("immutable node hash mismatch")
        compiled = compile_node_source(
            source, provides_by_outcome=manifest["provides_by_outcome"],
            checkpoint_outcomes=manifest["checkpoint_outcomes"],
        )
        if not compiled["eligible"]:
            raise NodeCompileError(f"stored node no longer compiles: {compiled['violations']}")
        return {"manifest": manifest, "source": source, "path": str(path)}

    def execute(self, node_id: str, context: Mapping[str, Any], dispatch) -> dict[str, Any]:
        inspected = self.inspect(node_id); manifest = inspected["manifest"]
        missing = set(manifest["requires"]) - set(context)
        if missing: raise NodeCompileError(f"missing node inputs: {sorted(missing)}")
        report = self.runtime.execute(
            Path(inspected["path"]) / "node.py",
            expected_sha256=manifest["source_sha256"], context=context,
            dispatch=dispatch,
        )
        if not report["completed"]: return report
        result = report["result"]
        if not isinstance(result, Mapping): raise NodeRuntimeError("node result is not object")
        outcome = str(result.get("outcome") or ""); updates = result.get("updates")
        if outcome not in manifest["outcomes"] or not isinstance(updates, Mapping):
            raise NodeRuntimeError("node result violates outcome/update contract")
        expected = set(manifest["provides_by_outcome"][outcome])
        if set(updates) != expected: raise NodeRuntimeError("node runtime update fields differ")
        return {**report, "outcome": outcome, "updates": dict(updates)}


__all__ = ["NodeStore", "compile_node_source"]
