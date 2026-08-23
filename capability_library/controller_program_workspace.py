"""Versioned and audited workspace for complete controller programs."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping

from controller_program_runtime import ControllerProgramRuntime
from runtime_capabilities import HOOK_CONTRACTS


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_ALLOWED_IMPORTS = {"math", "statistics"}
_FORBIDDEN_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "input", "print",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
}
_FORBIDDEN_TERMS = {
    "reward", "check_success", "bddl", "mujoco", "sim.data", "body_xpos",
    "segmentation_id", "evaluator", "task_id", "state_id", "subprocess",
    "socket", "pathlib", "os.", "sys.",
}

_STAGE_CHECKPOINT_TOOLS = {
    "articulation": {"verify_landmark_displacement"},
    "attachment": {"verify_attachment"},
    "placement": {"verify_support_relation"},
}


class ControllerProgramValidationError(ValueError):
    pass


def verified_stage_from_evidence(evidence: Mapping[str, Any]) -> str | None:
    """Return the deepest sensor-proven stage that may be frozen as code."""
    if bool(evidence.get("placement_verified")):
        return "placement"
    if bool(evidence.get("attachment_verified")):
        return "attachment"
    if any(
        isinstance(item, Mapping)
        and item.get("kind") == "articulation"
        and bool(item.get("verified"))
        for item in evidence.get("verifications") or ()
    ):
        return "articulation"
    return None


def protected_prefix_for_stage(
    source: str, stage: str, *, verifier_ordinal: int | None = None,
) -> dict[str, Any]:
    """Fingerprint executable AST through a sensor verifier guard.

    Comments and formatting are intentionally ignored.  The imports, run
    signature, helper definitions, executable statements, verifier call, and
    following guard must remain structurally identical.  This preserves a
    proven skill prefix without imposing brittle byte-for-byte copying.
    """
    tools = _STAGE_CHECKPOINT_TOOLS.get(str(stage))
    if not tools:
        raise ControllerProgramValidationError(
            f"no structural checkpoint verifier is defined for stage: {stage}"
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ControllerProgramValidationError(
            f"cannot derive protected prefix: {exc.msg}"
        ) from exc
    lines = source.splitlines(keepends=True)
    run_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run"
    ]
    if len(run_functions) != 1:
        raise ControllerProgramValidationError(
            "cannot derive protected prefix without exactly one run function"
        )
    run_function = run_functions[0]
    verifier_calls = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "call_tool"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in tools
        ):
            continue
        verifier_calls.append(node)
    verifier_calls.sort(key=lambda node: (int(node.lineno), int(node.col_offset)))
    if not verifier_calls:
        raise ControllerProgramValidationError(
            f"base program has no verifier call for protected stage: {stage}"
        )
    ordinal = len(verifier_calls) if verifier_ordinal is None else int(verifier_ordinal)
    if ordinal < 1 or ordinal > len(verifier_calls):
        raise ControllerProgramValidationError(
            f"base program has fewer than {ordinal} verifier calls for stage: {stage}"
        )
    verifier = verifier_calls[ordinal - 1]
    containing_index = next((
        index for index, statement in enumerate(run_function.body)
        if int(statement.lineno) <= int(verifier.lineno)
        <= int(getattr(statement, "end_lineno", statement.lineno))
    ), None)
    if containing_index is None:
        raise ControllerProgramValidationError(
            f"stage verifier is not part of the run control flow: {stage}"
        )
    guard_index = min(containing_index + 1, len(run_function.body) - 1)
    protected_body = run_function.body[:guard_index + 1]
    through_line = int(getattr(
        protected_body[-1], "end_lineno", protected_body[-1].lineno
    ))
    module_prefix = [
        ast.dump(node, include_attributes=False)
        for node in tree.body if node is not run_function
        and int(getattr(node, "lineno", 0)) < int(run_function.lineno)
    ]
    structure = {
        "module_prefix": module_prefix,
        "run_arguments": ast.dump(run_function.args, include_attributes=False),
        "run_decorators": [
            ast.dump(node, include_attributes=False)
            for node in run_function.decorator_list
        ],
        "run_body": [
            ast.dump(node, include_attributes=False) for node in protected_body
        ],
    }
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
    prefix = "".join(lines[:through_line])
    return {
        "stage": str(stage),
        "through_line": through_line,
        "verifier_ordinal": ordinal,
        "protected_statement_count": len(protected_body),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "structure": structure,
        "source": prefix,
    }


def _literal_mapping_keys(node: ast.AST | None) -> set[str] | None:
    if not isinstance(node, ast.Dict):
        return None
    keys: set[str] = set()
    for key in node.keys:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            return None
        keys.add(key.value)
    return keys


def _is_numeric_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    )


def _capability_call(node: ast.AST, robot_name: str) -> tuple[str, ast.AST | None] | None:
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "call_tool"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == robot_name
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and ":v" in node.args[0].value
    ):
        return None
    payload = node.args[1] if len(node.args) >= 2 else next(
        (item.value for item in node.keywords if item.arg in {"arguments", "payload"}),
        None,
    )
    return str(node.args[0].value), payload


def _audit_capability_use(
    run_function: ast.AST,
    robot_name: str,
    contracts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Require tested Tools to be wired as typed control interfaces.

    The runtime validates the JSON contract.  This authoring-time check covers
    the complementary failure mode where a program successfully invokes a Tool
    but silently ignores part of its output and therefore does not implement the
    acquired behavior.
    """
    violations: list[str] = []
    literal_dicts: dict[str, ast.Dict] = {}
    aliases: list[tuple[str, str]] = []
    assignments: dict[ast.Call, str] = {}
    for node in ast.walk(run_function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
            literal_dicts[target.id] = node.value
        elif isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
            aliases.append((target.id, node.value.id))
        elif isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
            assignments[node.value] = target.id

    for node in ast.walk(run_function):
        call = _capability_call(node, robot_name)
        if call is None:
            continue
        tool_id, payload = call
        contract = contracts.get(tool_id)
        if contract is None:
            violations.append(f"unknown_or_untested_capability_tool:{tool_id}")
            continue
        if isinstance(payload, ast.Name):
            payload = literal_dicts.get(payload.id)
        payload_keys = _literal_mapping_keys(payload)
        required_inputs = set(contract.get("input_fields") or ())
        if payload_keys is None:
            violations.append(f"capability_payload_not_statically_auditable:{tool_id}")
        else:
            for field in sorted(required_inputs - payload_keys):
                violations.append(f"capability_input_not_supplied:{tool_id}:{field}")

        result_name = assignments.get(node)
        required_outputs = set(contract.get("output_fields") or ())
        if required_outputs and result_name is None:
            violations.append(f"capability_result_not_assigned:{tool_id}")
            continue
        names = {result_name} if result_name is not None else set()
        changed = True
        while changed:
            changed = False
            for target, source in aliases:
                if source in names and target not in names:
                    names.add(target)
                    changed = True
        consumed: set[str] = set()
        for use in ast.walk(run_function):
            if (
                isinstance(use, ast.Subscript)
                and isinstance(use.value, ast.Name)
                and use.value.id in names
                and isinstance(use.slice, ast.Constant)
                and isinstance(use.slice.value, str)
            ):
                consumed.add(use.slice.value)
            elif (
                isinstance(use, ast.Call)
                and isinstance(use.func, ast.Attribute)
                and use.func.attr == "get"
                and isinstance(use.func.value, ast.Name)
                and use.func.value.id in names
                and use.args
                and isinstance(use.args[0], ast.Constant)
                and isinstance(use.args[0].value, str)
            ):
                consumed.add(use.args[0].value)
        for field in sorted(required_outputs - consumed):
            violations.append(f"capability_output_not_consumed:{tool_id}:{field}")
    return violations


def audit_controller_program(
    source: str,
    *,
    capability_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    violations = [term for term in _FORBIDDEN_TERMS if term in source.casefold()]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"eligible": False, "violations": [f"syntax_error:{exc.msg}"]}
    run_functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    if len(run_functions) != 1 or len(run_functions[0].args.args) != 1:
        violations.append("run_must_accept_exactly_one_robot_argument")
    else:
        robot_name = run_functions[0].args.args[0].arg
        uses_live_instruction = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "instruction"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == robot_name
            and not node.args and not node.keywords
            for node in ast.walk(run_functions[0])
        )
        if not uses_live_instruction:
            violations.append("missing_robot_instruction_call")
        if capability_contracts is not None:
            violations.extend(
                _audit_capability_use(
                    run_functions[0], robot_name, capability_contracts
                )
            )
        generates_grasps = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "call_tool"
            and len(node.args) >= 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "generate_grasps"
            for node in ast.walk(run_functions[0])
        )
        consumes_model_approach = any(
            (
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "approach_world"
            )
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "approach_world"
            )
            for node in ast.walk(run_functions[0])
        )
        if generates_grasps and not consumes_model_approach:
            violations.append("missing_grasp_approach_world_consumption")
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
                if not (
                    isinstance(key, ast.Constant)
                    and key.value == "target_eef_xyz"
                    and isinstance(value, (ast.List, ast.Tuple))
                    and len(value.elts) == 3
                    and all(_is_numeric_literal(item) for item in value.elts)
                ):
                    continue
                violations.append("literal_absolute_action_target")
    return {"eligible": not violations, "violations": sorted(set(violations))}


class ControllerProgramWorkspace:
    def __init__(
        self,
        root: str | Path,
        *,
        python: str | Path,
        timeout_sec: float = 900,
        max_rpc_calls: int = 5000,
        capability_workspace: str | Path | None = None,
        required_revision: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.capability_workspace = (
            None if capability_workspace is None
            else Path(capability_workspace).resolve()
        )
        self.runtime = ControllerProgramRuntime(
            python=python, timeout_sec=timeout_sec, max_rpc_calls=max_rpc_calls
        )
        self.required_revision = dict(required_revision or {})
        self.root.mkdir(parents=True, exist_ok=True)

    def revision_constraint(self) -> dict[str, Any] | None:
        if not self.required_revision:
            return None
        base_program_id = str(self.required_revision.get("base_program_id") or "")
        stage = str(self.required_revision.get("stage") or "")
        base = self.resolve(base_program_id)
        checkpoint = protected_prefix_for_stage(
            (base / "program.py").read_text(), stage
        )
        return {
            "base_program_id": base_program_id,
            "stage": stage,
            "protected_through_line": checkpoint["through_line"],
            "protected_prefix_sha256": checkpoint["sha256"],
            "protected_prefix_comparison": "python_ast",
            "verifier_ordinal": checkpoint["verifier_ordinal"],
            "protected_statement_count": checkpoint["protected_statement_count"],
            "protected_structure": checkpoint["structure"],
            "protected_source": checkpoint["source"],
        }

    def capability_contracts(self) -> dict[str, dict[str, Any]] | None:
        if self.capability_workspace is None:
            return None
        contracts: dict[str, dict[str, Any]] = {}
        for path in self.capability_workspace.glob("*/v[0-9][0-9][0-9]/manifest.json"):
            manifest = json.loads(path.read_text())
            hooks = [
                str(hook) for hook in manifest.get("compatible_hooks") or ()
                if hook in HOOK_CONTRACTS
            ]
            if manifest.get("status") != "unit_tested" or len(hooks) != 1:
                generic = manifest.get("generic_contract") or {}
                input_schema = generic.get("input_schema") or {}
                output_schema = generic.get("output_schema") or {}
                if (
                    manifest.get("status") == "unit_tested"
                    and input_schema.get("type") == "object"
                    and output_schema.get("type") == "object"
                ):
                    contracts[str(manifest.get("tool_id"))] = {
                        "hook": "generic_capability",
                        "input_fields": {
                            str(field): "required by generic JSON schema"
                            for field in input_schema.get("required") or ()
                        },
                        "output_fields": {
                            str(field): "required by generic JSON schema"
                            for field in output_schema.get("required") or ()
                        },
                    }
                continue
            contract = HOOK_CONTRACTS[hooks[0]]
            contracts[str(manifest.get("tool_id"))] = {
                "hook": hooks[0],
                "input_fields": dict(contract.get("input_fields") or {}),
                "output_fields": dict(contract.get("output_fields") or {}),
            }
        return contracts

    def create(self, name: str, source: str, rationale: str = "") -> dict[str, Any]:
        if not _NAME.fullmatch(str(name)):
            raise ControllerProgramValidationError("name must match ^[a-z][a-z0-9_]{2,63}$")
        audit = audit_controller_program(
            source, capability_contracts=self.capability_contracts()
        )
        if not audit["eligible"]:
            raise ControllerProgramValidationError(f"program audit failed: {audit['violations']}")
        revision = self.revision_constraint()
        if revision is not None:
            candidate_checkpoint = protected_prefix_for_stage(
                source, str(revision["stage"]),
                verifier_ordinal=int(revision["verifier_ordinal"]),
            )
            if (
                candidate_checkpoint["structure"] != revision["protected_structure"]
                or candidate_checkpoint["sha256"]
                    != revision["protected_prefix_sha256"]
            ):
                raise ControllerProgramValidationError(
                    "verified_stage_prefix_changed: preserve executable AST from "
                    f"{revision['base_program_id']} through line "
                    f"{revision['protected_through_line']} "
                    f"({revision['stage']})"
                )
        family = self.root / name
        versions = [int(path.name[1:]) for path in family.glob("v[0-9]*") if path.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        module = destination / "program.py"
        module.write_text(source)
        digest = hashlib.sha256(module.read_bytes()).hexdigest()
        manifest = {
            "protocol": "embodied-controller-program-v1",
            "name": name,
            "version": version,
            "program_id": f"{name}:v{version:03d}",
            "sha256": digest,
            "rationale": str(rationale),
            "audit": audit,
            "created_unix": time.time(),
            "privileged_state_used": False,
            "current_task_data_used_for_parameters": False,
            "status": "candidate",
            "revision_constraint": (
                None if revision is None else {
                    key: value for key, value in revision.items()
                    if key not in {"protected_source", "protected_structure"}
                }
            ),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "success": True,
            "program_id": manifest["program_id"],
            "sha256": digest,
            "audit": audit,
        }

    def resolve(self, program_id: str) -> Path:
        name, separator, version = str(program_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise ControllerProgramValidationError("program_id must look like name:v001")
        destination = (self.root / name / version).resolve()
        if self.root not in destination.parents or not (destination / "manifest.json").is_file():
            raise FileNotFoundError(program_id)
        return destination

    def inspect(
        self, program_id: str, *, include_run_details: bool = True,
    ) -> dict[str, Any]:
        destination = self.resolve(program_id)
        runs = []
        for path in sorted((destination / "runs").glob("run_*.json")) if (destination / "runs").is_dir() else ():
            report = json.loads(path.read_text())
            runs.append(
                report if include_run_details else {
                    "execution_completed": bool(report.get("execution_completed")),
                    "error": report.get("error"),
                    "rpc_calls": len(report.get("rpc_events") or ()),
                    "trace_path": str(path),
                }
            )
        return {
            "success": True,
            "program_id": program_id,
            "manifest": json.loads((destination / "manifest.json").read_text()),
            "source": (destination / "program.py").read_text(),
            "runs": runs,
        }

    def execute(
        self,
        program_id: str,
        dispatch: Callable[[str, Mapping[str, Any]], Any],
    ) -> dict[str, Any]:
        destination = self.resolve(program_id)
        manifest = json.loads((destination / "manifest.json").read_text())
        report = self.runtime.run(
            destination / "program.py",
            expected_sha256=manifest["sha256"],
            dispatch=dispatch,
        )
        run_dir = destination / "runs"
        run_dir.mkdir(exist_ok=True)
        run_path = run_dir / f"run_{len(list(run_dir.glob('run_*.json'))) + 1:03d}.json"
        run_path.write_text(json.dumps(report, indent=2) + "\n")
        return {**report, "program_id": program_id, "trace_path": str(run_path)}


def register_controller_program_tools(
    registry: Any,
    workspace: ControllerProgramWorkspace,
    *,
    dispatch_factory: Callable[[], Callable[[str, Mapping[str, Any]], Any]] | None = None,
    executor: Callable[[str], Mapping[str, Any]] | None = None,
) -> None:
    @registry.tool(
        name="create_controller_program",
        description=(
            "Create an immutable audited Python controller defining run(robot). "
            "The program owns loops and branching and may call robot.instruction(), "
            "robot.observe(), robot.call_tool(name,args), robot.act(action), and "
            "robot.record(event). It cannot access files, simulator internals, or evaluator data."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 3, "maxLength": 64},
                "source": {"type": "string", "minLength": 20, "maxLength": 50000},
                "rationale": {"type": "string", "maxLength": 4000},
            },
            "required": ["name", "source"],
            "additionalProperties": False,
        },
    )
    def create_controller_program(name: str, source: str, rationale: str = ""):
        try:
            return workspace.create(name, source, rationale)
        except ControllerProgramValidationError as exc:
            # Validation feedback is legal engineering evidence. Returning the
            # exact audit reason lets the Coding Agent repair its own program;
            # raising here causes the Harness transport to erase the detail.
            return {
                "success": False,
                "reason": str(exc),
                "controller_created": False,
            }

    @registry.tool(
        name="inspect_controller_program",
        description="Read one immutable controller program source and manifest.",
        input_schema={
            "type": "object",
            "properties": {
                "program_id": {"type": "string", "minLength": 8, "maxLength": 69},
            },
            "required": ["program_id"],
            "additionalProperties": False,
        },
    )
    def inspect_controller_program(program_id: str):
        return workspace.inspect(program_id, include_run_details=False)

    if executor is not None or dispatch_factory is not None:
        @registry.tool(
            name="execute_controller_program",
            description=(
                "Execute one audited complete controller program through the "
                "deployment-owned sensor/action Robot SDK. Evaluator fields are never returned."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "program_id": {"type": "string", "minLength": 8, "maxLength": 69},
                },
                "required": ["program_id"],
                "additionalProperties": False,
            },
        )
        def execute_controller_program(program_id: str):
            if executor is not None:
                result = dict(executor(program_id))
                # Tool success means the Harness call completed; robot outcome
                # remains exclusively in sensor_evidence.
                result.setdefault("success", True)
                return result
            assert dispatch_factory is not None
            return workspace.execute(program_id, dispatch_factory())


__all__ = [
    "ControllerProgramValidationError",
    "ControllerProgramWorkspace",
    "audit_controller_program",
    "protected_prefix_for_stage",
    "register_controller_program_tools",
    "verified_stage_from_evidence",
]
