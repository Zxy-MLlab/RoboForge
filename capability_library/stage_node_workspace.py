"""Immutable typed Stage Nodes for structured embodied controller graphs."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

from jsonschema import Draft202012Validator, ValidationError

from controller_program_runtime import ControllerProgramRuntime
from controller_program_workspace import (
    _ALLOWED_IMPORTS,
    _FORBIDDEN_NAMES,
    _FORBIDDEN_TERMS,
    _NAME,
    _audit_capability_use,
    _is_numeric_literal,
    ControllerProgramValidationError,
    ControllerProgramWorkspace,
)


_FIELD = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_OUTCOME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class StageNodeValidationError(ValueError):
    pass


def audit_stage_return_contract(
    source: str,
    provides_by_outcome: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Compile direct run_stage returns against the declared typed contract.

    Outcomes and update keys are intentionally literal at the node boundary.
    Branching remains ordinary Python, but each branch must terminate in an
    unambiguous protocol object so mistakes are rejected before a robot episode.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"eligible": False, "violations": [f"syntax_error:{exc.msg}"]}
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_stage"
    ]
    if len(functions) != 1:
        return {"eligible": False, "violations": ["missing_run_stage"]}

    class DirectReturnVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.returns: list[ast.Return] = []

        def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
            self.returns.append(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            # Do not mistake returns in a nested helper for Stage results.
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

    visitor = DirectReturnVisitor()
    for statement in functions[0].body:
        visitor.visit(statement)
    violations: list[str] = []
    implemented: set[str] = set()
    for index, returned in enumerate(visitor.returns, 1):
        value = returned.value
        if not isinstance(value, ast.Dict):
            violations.append(f"return_{index}:stage_result_must_be_object_literal")
            continue
        if any(not isinstance(key, ast.Constant) or not isinstance(key.value, str)
               for key in value.keys):
            violations.append(f"return_{index}:stage_result_keys_must_be_literal")
            continue
        fields = {str(key.value): item for key, item in zip(value.keys, value.values)}
        if set(fields) != {"outcome", "updates"}:
            violations.append(f"return_{index}:stage_result_requires_outcome_and_updates")
            continue
        outcome_node = fields["outcome"]
        if not isinstance(outcome_node, ast.Constant) or not isinstance(
            outcome_node.value, str
        ):
            violations.append(f"return_{index}:outcome_must_be_literal")
            continue
        outcome = str(outcome_node.value)
        if outcome not in provides_by_outcome:
            violations.append(f"return_{index}:undeclared_outcome:{outcome}")
            continue
        updates_node = fields["updates"]
        if not isinstance(updates_node, ast.Dict) or any(
            not isinstance(key, ast.Constant) or not isinstance(key.value, str)
            for key in getattr(updates_node, "keys", [])
        ):
            violations.append(f"return_{index}:updates_must_be_object_literal")
            continue
        actual = {str(key.value) for key in updates_node.keys}
        expected = set(provides_by_outcome[outcome])
        if actual != expected:
            violations.append(
                f"return_{index}:updates_mismatch:{outcome}:"
                f"expected={sorted(expected)}:actual={sorted(actual)}"
            )
            continue
        implemented.add(outcome)
    if not visitor.returns:
        violations.append("run_stage_has_no_direct_return")
    missing = set(provides_by_outcome) - implemented
    if missing:
        violations.append(f"declared_outcomes_not_implemented:{sorted(missing)}")
    return {
        "eligible": not violations,
        "violations": sorted(set(violations)),
        "implemented_outcomes": sorted(implemented),
    }


def audit_stage_node(
    source: str,
    *,
    capability_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit one ``run_stage(robot, context)`` implementation."""
    violations = [term for term in _FORBIDDEN_TERMS if term in source.casefold()]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"eligible": False, "violations": [f"syntax_error:{exc.msg}"]}
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_stage"
    ]
    if len(functions) != 1 or len(functions[0].args.args) != 2:
        violations.append("run_stage_must_accept_exactly_robot_and_context")
    else:
        robot_name = functions[0].args.args[0].arg
        context_name = functions[0].args.args[1].arg
        if robot_name == context_name:
            violations.append("robot_and_context_names_must_differ")
        if capability_contracts is not None:
            violations.extend(
                _audit_capability_use(functions[0], robot_name, capability_contracts)
            )
        for returned in (
            node for node in ast.walk(functions[0]) if isinstance(node, ast.Return)
        ):
            # The runtime protocol requires a JSON object. Catch the common
            # Python ``return outcome, updates`` mistake before a costly robot
            # episode is launched; less-direct expressions remain runtime-
            # validated by StageNodeWorkspace.execute.
            if isinstance(returned.value, (ast.Tuple, ast.List, ast.Constant)):
                violations.append("run_stage_return_must_be_object")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    violations.append(f"forbidden_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "") not in _ALLOWED_IMPORTS:
                violations.append(f"forbidden_import:{node.module}")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.append(f"forbidden_name:{node.id}")
        elif isinstance(node, ast.Attribute) and str(node.attr).startswith("_"):
            violations.append(f"private_attribute:{node.attr}")
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in {
                        "target_eef_xyz", "target_xyz", "translation_world",
                        "contact_xyz", "pregrasp_xyz", "place_xyz",
                    }
                    and isinstance(value, (ast.List, ast.Tuple))
                    and len(value.elts) == 3
                    and all(_is_numeric_literal(item) for item in value.elts)
                ):
                    violations.append("literal_absolute_geometry_target")
    return {"eligible": not violations, "violations": sorted(set(violations))}


class StageNodeWorkspace:
    """Versioned node code with explicit context and outcome contracts."""

    def __init__(
        self,
        root: str | Path,
        *,
        python: str | Path = "/data/zxy/envs/vla-report/bin/python",
        timeout_sec: float = 900,
        max_rpc_calls: int = 5000,
        capability_workspace: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.capability_workspace = (
            None if capability_workspace is None
            else Path(capability_workspace).resolve()
        )
        self.runtime = ControllerProgramRuntime(
            python=python, timeout_sec=timeout_sec, max_rpc_calls=max_rpc_calls
        )

    def capability_contracts(self) -> dict[str, dict[str, Any]] | None:
        if self.capability_workspace is None:
            return None
        # Reuse the single authoritative Tool-contract loader. The temporary
        # program root is never written by this call.
        return ControllerProgramWorkspace(
            self.root / ".contract_reader",
            python=self.runtime.python,
            capability_workspace=self.capability_workspace,
        ).capability_contracts()

    def create(
        self,
        *,
        name: str,
        stage_kind: str,
        source: str,
        description: str,
        requires: list[str],
        provides_by_outcome: Mapping[str, list[str]],
        checkpoint_outcomes: list[str] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(str(name)):
            raise StageNodeValidationError("name must match the node-name contract")
        if not _FIELD.fullmatch(str(stage_kind)):
            raise StageNodeValidationError("stage_kind must be a stable snake_case name")
        required = list(dict.fromkeys(str(item) for item in requires))
        if any(not _FIELD.fullmatch(item) for item in required):
            raise StageNodeValidationError("requires contains an invalid context field")
        outcomes = {
            str(outcome): list(dict.fromkeys(str(field) for field in fields))
            for outcome, fields in provides_by_outcome.items()
        }
        if not outcomes or any(not _OUTCOME.fullmatch(item) for item in outcomes):
            raise StageNodeValidationError("at least one valid outcome is required")
        if any(
            not _FIELD.fullmatch(field)
            for fields in outcomes.values() for field in fields
        ):
            raise StageNodeValidationError("provides contains an invalid context field")
        checkpoints = list(dict.fromkeys(str(item) for item in (checkpoint_outcomes or [])))
        if set(checkpoints) - set(outcomes):
            raise StageNodeValidationError("checkpoint_outcomes must be declared outcomes")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])
        verifier_calls = [
                str(node.args[0].value)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "call_tool"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and str(node.args[0].value).startswith("verify_")
        ]
        action_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "act"
        ]
        if verifier_calls and "verification" not in str(stage_kind):
            raise StageNodeValidationError(
                "adapter verify_* calls belong in an independent verification Stage Node"
            )
        if "verification" in str(stage_kind) and action_calls:
            raise StageNodeValidationError(
                "verification Stage Nodes cannot execute robot actions"
            )
        if checkpoints:
            if "verification" not in str(stage_kind) or not verifier_calls:
                raise StageNodeValidationError(
                    "checkpoint outcomes require a verification Stage Node "
                    "that calls an adapter-owned verify_* Tool"
                )
        audit = audit_stage_node(
            source, capability_contracts=self.capability_contracts()
        )
        if not audit["eligible"]:
            raise StageNodeValidationError(f"stage audit failed: {audit['violations']}")
        return_contract = audit_stage_return_contract(source, outcomes)
        if not return_contract["eligible"]:
            raise StageNodeValidationError(
                "stage return contract failed: "
                f"{return_contract['violations']}"
            )
        family = self.root / name
        versions = [
            int(path.name[1:]) for path in family.glob("v[0-9]*")
            if path.name[1:].isdigit()
        ]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        module = destination / "stage.py"
        module.write_text(source)
        digest = hashlib.sha256(module.read_bytes()).hexdigest()
        manifest = {
            "protocol": "embodied-stage-node-v1",
            "name": name,
            "version": version,
            "node_id": f"{name}:v{version:03d}",
            "stage_kind": stage_kind,
            "description": str(description),
            "requires": required,
            "provides_by_outcome": outcomes,
            "allowed_outcomes": sorted(outcomes),
            "checkpoint_outcomes": checkpoints,
            "sha256": digest,
            "audit": audit,
            "return_contract": return_contract,
            "status": "audited",
            "created_unix": time.time(),
            "privileged_state_used": False,
            "literal_absolute_action_targets": False,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "success": True, "node_id": manifest["node_id"],
            "sha256": digest, "audit": audit,
        }

    def resolve(self, node_id: str) -> Path:
        name, separator, version = str(node_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise StageNodeValidationError("node_id must look like name:v001")
        destination = (self.root / name / version).resolve()
        if self.root not in destination.parents or not (destination / "manifest.json").is_file():
            raise FileNotFoundError(node_id)
        return destination

    def inspect(self, node_id: str) -> dict[str, Any]:
        destination = self.resolve(node_id)
        manifest = json.loads((destination / "manifest.json").read_text())
        digest = hashlib.sha256((destination / "stage.py").read_bytes()).hexdigest()
        if digest != manifest.get("sha256"):
            raise StageNodeValidationError("immutable Stage Node hash mismatch")
        return {
            "success": True, "node_id": node_id, "manifest": manifest,
            "source": (destination / "stage.py").read_text(),
        }

    def execute(
        self,
        node_id: str,
        context: Mapping[str, Any],
        dispatch: Callable[[str, Mapping[str, Any]], Any],
    ) -> dict[str, Any]:
        inspected = self.inspect(node_id)
        manifest = inspected["manifest"]
        missing = [field for field in manifest["requires"] if field not in context]
        if missing:
            raise StageNodeValidationError(
                f"Stage Node missing context fields: {sorted(missing)}"
            )
        report = self.runtime.run(
            self.resolve(node_id) / "stage.py",
            expected_sha256=manifest["sha256"], dispatch=dispatch,
            entrypoint="run_stage", arguments=dict(context),
        )
        if not report.get("execution_completed"):
            return {**report, "node_id": node_id}
        result = report.get("result")
        if not isinstance(result, Mapping):
            raise StageNodeValidationError("Stage Node must return an object")
        outcome = str(result.get("outcome") or "")
        updates = result.get("updates")
        if outcome not in manifest["allowed_outcomes"]:
            raise StageNodeValidationError(f"undeclared Stage Node outcome: {outcome}")
        if not isinstance(updates, Mapping):
            raise StageNodeValidationError("Stage Node updates must be an object")
        promised = set(manifest["provides_by_outcome"].get(outcome) or ())
        missing_outputs = sorted(promised - set(updates))
        if missing_outputs:
            raise StageNodeValidationError(
                f"Stage Node omitted promised outputs: {missing_outputs}"
            )
        undeclared_outputs = sorted(set(updates) - promised)
        if undeclared_outputs:
            raise StageNodeValidationError(
                f"Stage Node returned undeclared outputs: {undeclared_outputs}"
            )
        return {
            **report, "node_id": node_id, "outcome": outcome,
            "updates": dict(updates),
        }


def register_stage_node_tools(registry: Any, workspace: StageNodeWorkspace) -> None:
    @registry.tool(
        name="create_stage_node",
        description=(
            "Create one immutable typed embodied Stage Node implementing "
            "run_stage(robot, context). Geometry must come from context or live sensors."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3, "maxLength": 64},
                "stage_kind": {"type": "string", "minLength": 1, "maxLength": 64},
                "source": {"type": "string", "minLength": 30, "maxLength": 30000},
                "description": {"type": "string", "maxLength": 2000},
                "requires": {
                    "type": "array", "items": {"type": "string"},
                },
                "provides_by_outcome": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "checkpoint_outcomes": {
                    "type": "array", "items": {"type": "string"},
                },
            },
            "required": [
                "name", "stage_kind", "source", "description", "requires",
                "provides_by_outcome",
            ],
            "additionalProperties": False,
        },
    )
    def create_stage_node(
        name: str, stage_kind: str, source: str, description: str,
        requires: list[str], provides_by_outcome: Mapping[str, list[str]],
        checkpoint_outcomes: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            return workspace.create(
                name=name, stage_kind=stage_kind, source=source,
                description=description, requires=requires,
                provides_by_outcome=provides_by_outcome,
                checkpoint_outcomes=checkpoint_outcomes,
            )
        except StageNodeValidationError as exc:
            return {"success": False, "node_created": False, "reason": str(exc)}

    @registry.tool(
        name="inspect_stage_node",
        description="Inspect one immutable Stage Node source, typed contract, and hash.",
        input_schema={
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"], "additionalProperties": False,
        },
    )
    def inspect_stage_node(node_id: str) -> dict[str, Any]:
        try:
            return workspace.inspect(node_id)
        except (StageNodeValidationError, FileNotFoundError) as exc:
            return {"success": False, "reason": str(exc)}


__all__ = [
    "StageNodeValidationError", "StageNodeWorkspace", "audit_stage_node",
    "audit_stage_return_contract",
    "register_stage_node_tools",
]
