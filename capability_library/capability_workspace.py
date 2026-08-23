"""Audited workspace for agent-authored, reusable pure-Python capabilities."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from jsonschema import Draft202012Validator, ValidationError

from asset_registry import register_asset
from runtime_capabilities import (
    HOOK_CONTRACTS,
    RuntimeCapabilityError,
    SUPPORTED_HOOKS,
    validate_hook_output,
)


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_ALLOWED_IMPORTS = {"json", "math", "re", "statistics"}
_FORBIDDEN_TERMS = {
    "reward", "check_success", "bddl", "mujoco", "sim.data",
    "body_xpos", "segmentation_id", "evaluator", "task_id", "state_id",
}
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}


class CapabilityValidationError(ValueError):
    pass


def audit_capability_source(source: str) -> dict[str, Any]:
    violations = [term for term in _FORBIDDEN_TERMS if term in source.casefold()]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"eligible": False, "violations": [f"syntax_error:{exc.msg}"]}
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not any(node.name == "run" for node in functions):
        violations.append("missing_run_function")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    violations.append(f"forbidden_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if str(node.module or "") not in _ALLOWED_IMPORTS:
                violations.append(f"forbidden_import:{node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            violations.append(f"forbidden_call:{node.func.id}")
    return {"eligible": not violations, "violations": sorted(set(violations))}


class CapabilityWorkspace:
    """Create, test, freeze, and invoke JSON-in/JSON-out capability tools."""

    def __init__(self, root: str | Path, *, python: str | Path, timeout_sec: int = 10, library_path: str | Path | None = None):
        self.root = Path(root).resolve()
        self.python = Path(python)
        self.timeout_sec = int(timeout_sec)
        self.library_path = None if library_path is None else Path(library_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        source: str,
        description: str,
        *,
        source_urls: list[str] | None = None,
        input_schema: Mapping[str, Any] | None = None,
        output_schema: Mapping[str, Any] | None = None,
        stage: str = "generic",
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(str(name)):
            raise CapabilityValidationError("name must match ^[a-z][a-z0-9_]{2,63}$")
        audit = audit_capability_source(source)
        if not audit["eligible"]:
            raise CapabilityValidationError(f"source audit failed: {audit['violations']}")
        if (input_schema is None) != (output_schema is None):
            raise CapabilityValidationError(
                "input_schema and output_schema must be supplied together"
            )
        generic_contract = None
        if input_schema is not None and output_schema is not None:
            try:
                Draft202012Validator.check_schema(dict(input_schema))
                Draft202012Validator.check_schema(dict(output_schema))
            except Exception as exc:
                raise CapabilityValidationError(
                    f"invalid generic JSON schema: {exc}"
                ) from exc
            if input_schema.get("type") != "object" or output_schema.get("type") != "object":
                raise CapabilityValidationError(
                    "generic controller Tool schemas must have root type object"
                )
            generic_contract = {
                "input_schema": dict(input_schema),
                "output_schema": dict(output_schema),
                "stage": str(stage or "generic")[:80],
            }
        family = self.root / name
        versions = [int(path.name[1:]) for path in family.glob("v[0-9]*") if path.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        module = destination / "tool.py"
        module.write_text(source)
        manifest = {
            "protocol": "agent-authored-capability-v1", "name": name, "version": version,
            "tool_id": f"{name}:v{version:03d}", "description": str(description),
            "source_urls": list(source_urls or []), "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "audit": audit, "status": "created", "current_task_data_used": False,
            "privileged_state_used": False, "compatible_hooks": [],
        }
        if generic_contract is not None:
            manifest["generic_contract"] = generic_contract
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {"success": True, "tool_id": manifest["tool_id"], "audit": audit}

    def resolve(self, tool_id: str) -> Path:
        name, separator, version = str(tool_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise CapabilityValidationError("tool_id must look like name:v001")
        destination = (self.root / name / version).resolve()
        if self.root not in destination.parents or not (destination / "manifest.json").is_file():
            raise FileNotFoundError(tool_id)
        return destination

    def invoke(self, tool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        destination = self.resolve(tool_id)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        module = destination / "tool.py"
        if hashlib.sha256(module.read_bytes()).hexdigest() != manifest["sha256"]:
            raise CapabilityValidationError("capability source hash changed")
        runner = (
            "import importlib.util,json,sys;"
            "s=importlib.util.spec_from_file_location('cap',sys.argv[1]);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "v=m.run(json.loads(sys.stdin.read()));"
            "print(json.dumps(v))"
        )
        completed = subprocess.run(
            [str(self.python), "-I", "-c", runner, str(module)],
            input=json.dumps(dict(payload)), text=True, capture_output=True,
            timeout=self.timeout_sec,
        )
        if completed.returncode != 0:
            return {"success": False, "reason": completed.stderr[-1000:] or "capability process failed"}
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return {"success": False, "reason": f"capability returned invalid JSON: {exc}"}
        return {"success": True, "tool_id": tool_id, "result": result}

    def test(self, tool_id: str, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not 1 <= len(cases) <= 20:
            raise CapabilityValidationError("cases must contain 1 to 20 items")
        outcomes = []
        destination = self.resolve(tool_id)
        manifest = json.loads((destination / "manifest.json").read_text())
        generic_contract = manifest.get("generic_contract") or {}
        for case in cases:
            payload = case.get("input") or {}
            schema_error = None
            if generic_contract:
                try:
                    Draft202012Validator(
                        generic_contract["input_schema"]
                    ).validate(payload)
                except ValidationError as exc:
                    schema_error = f"input schema: {exc.message}"
            actual = (
                self.invoke(tool_id, payload)
                if schema_error is None else
                {"success": False, "reason": schema_error}
            )
            if actual.get("success") and generic_contract:
                try:
                    Draft202012Validator(
                        generic_contract["output_schema"]
                    ).validate(actual.get("result"))
                except ValidationError as exc:
                    schema_error = f"output schema: {exc.message}"
            passed = bool(
                schema_error is None and actual.get("success")
                and actual.get("result") == case.get("expected")
            )
            outcomes.append({
                "passed": passed,
                "input": payload,
                "actual": actual.get("result"),
                "expected": case.get("expected"),
                "error": schema_error or actual.get("reason"),
            })
        passed = all(item["passed"] for item in outcomes)
        manifest["status"] = "unit_tested" if passed else "test_failed"
        manifest["test_cases"] = outcomes
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        registration = self._register_tested(destination, manifest) if passed else None
        if registration is not None and not registration.get("success"):
            return {"success": False, "reason": f"asset registration failed: {registration}", "tool_id": tool_id, "outcomes": outcomes}
        return {"success": passed, "reason": "all cases passed" if passed else "one or more cases failed", "tool_id": tool_id, "outcomes": outcomes}

    def test_hook(self, tool_id: str, hook: str) -> dict[str, Any]:
        """Validate a Tool against the real runtime hook payload and output gate."""
        if hook not in SUPPORTED_HOOKS:
            raise CapabilityValidationError(f"unsupported capability hook: {hook}")
        outcomes = []
        for payload in HOOK_CONTRACTS[hook]["test_inputs"]:
            actual = self.invoke(tool_id, payload)
            error = None
            normalized = None
            if actual.get("success"):
                try:
                    normalized = validate_hook_output(hook, actual.get("result"), payload)
                except RuntimeCapabilityError as exc:
                    error = str(exc)
            else:
                error = str(actual.get("reason") or "capability invocation failed")
            outcomes.append({
                "passed": error is None,
                "input": payload,
                "actual": actual.get("result"),
                "normalized": normalized,
                "error": error,
            })
        passed = all(item["passed"] for item in outcomes)
        duplicate_of = self._behavioral_duplicate(tool_id, hook, outcomes) if passed else None
        if duplicate_of is not None:
            passed = False
        destination = self.resolve(tool_id)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        hook_tests = dict(manifest.get("hook_tests") or {})
        hook_tests[hook] = outcomes
        manifest["hook_tests"] = hook_tests
        compatible = set(manifest.get("compatible_hooks") or ())
        if passed:
            compatible.add(hook)
            manifest["status"] = "unit_tested"
            manifest.pop("behavioral_duplicate_of", None)
        elif duplicate_of is not None:
            compatible.discard(hook)
            manifest["status"] = "behavior_duplicate"
            manifest["behavioral_duplicate_of"] = duplicate_of
        elif manifest.get("status") != "unit_tested":
            manifest["status"] = "hook_test_failed"
        manifest["compatible_hooks"] = sorted(compatible)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        registration = self._register_tested(destination, manifest) if passed else None
        if registration is not None and not registration.get("success"):
            return {"success": False, "reason": f"asset registration failed: {registration}", "tool_id": tool_id, "hook": hook, "outcomes": outcomes}
        return {
            "success": passed,
            "reason": (
                "runtime hook contract passed" if passed else
                f"behavior duplicates existing Tool {duplicate_of}" if duplicate_of else
                "runtime hook contract failed"
            ),
            "tool_id": tool_id,
            "hook": hook,
            "behavioral_duplicate_of": duplicate_of,
            "outcomes": outcomes,
        }

    def _behavioral_duplicate(
        self,
        tool_id: str,
        hook: str,
        outcomes: list[Mapping[str, Any]],
    ) -> str | None:
        """Reject renamed Tools that are identical on Harness-owned probes."""
        expected = [item.get("normalized") for item in outcomes]
        for path in sorted(self.root.glob("*/v[0-9][0-9][0-9]/manifest.json")):
            manifest = json.loads(path.read_text())
            other_id = str(manifest.get("tool_id") or "")
            if other_id == tool_id or hook not in set(manifest.get("compatible_hooks") or ()):
                continue
            actual_outputs = []
            valid = True
            for payload in HOOK_CONTRACTS[hook]["test_inputs"]:
                actual = self.invoke(other_id, payload)
                if not actual.get("success"):
                    valid = False
                    break
                try:
                    actual_outputs.append(
                        validate_hook_output(hook, actual.get("result"), payload)
                    )
                except RuntimeCapabilityError:
                    valid = False
                    break
            if valid and actual_outputs == expected:
                return other_id
        return None

    def _register_tested(
        self, destination: Path, manifest: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        if self.library_path is None:
            return None
        return register_asset({
                "asset_id": f"tool.agent-authored-{manifest['name']}.v{manifest['version']}",
                "kind": "tool", "name": manifest["name"], "version": str(manifest["version"]),
                "status": "unit_tested", "source_urls": manifest.get("source_urls") or [],
                "implementation": str(destination / "tool.py"), "sha256": manifest["sha256"],
                "tested_tasks": [], "reused_tasks": [],
                "current_task_data_used": False, "privileged_state_used": False,
                "compatible_hooks": list(manifest.get("compatible_hooks") or []),
            }, library_path=str(self.library_path), event="agent_authored_tool_registered")

    def tested_tools(self) -> list[dict[str, Any]]:
        tools = []
        for path in self.root.glob("*/v[0-9][0-9][0-9]/manifest.json"):
            manifest = json.loads(path.read_text())
            if manifest.get("status") == "unit_tested":
                tools.append(manifest)
        return tools


def register_capability_workspace_tools(registry: Any, workspace: CapabilityWorkspace) -> None:
    bindable_tools = workspace.tested_tools()

    @registry.tool(
        name="create_capability_tool",
        description=(
            "Create an audited immutable pure-Python JSON-in/JSON-out Tool. "
            "Source must define run(payload), may import only json/math/re/statistics, "
            "and cannot access benchmark or evaluator internals. For a capability "
            "outside the four predefined hooks, supply explicit object input_schema "
            "and output_schema plus a generic stage, then validate it with "
            "test_capability_tool."
        ),
    )
    def create_capability_tool(
        name: str,
        source: str,
        description: str,
        source_urls: list[str] | None = None,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        stage: str = "generic",
    ):
        return workspace.create(
            name, source, description, source_urls=source_urls,
            input_schema=input_schema, output_schema=output_schema, stage=stage,
        )

    @registry.tool(
        name="test_capability_tool",
        description="Run deterministic input/expected-output tests for an agent-authored Tool in an isolated subprocess. Only passing Tools become available in the next evolution round.",
    )
    def test_capability_tool(tool_id: str, cases: list[dict[str, Any]]):
        return workspace.test(tool_id, cases)

    @registry.tool(
        name="describe_capability_hook_contracts",
        description=(
            "Return the exact machine-readable runtime payload fields, output "
            "bounds, and canonical test inputs for controller capability hooks. "
            "Call this before writing a Tool intended for a runtime hook."
        ),
    )
    def describe_capability_hook_contracts():
        return {"success": True, "contracts": HOOK_CONTRACTS}

    @registry.tool(
        name="test_capability_hook",
        description=(
            "Run an agent-authored Tool against the Harness-owned canonical "
            "runtime hook payload and safety validator. A controller can bind "
            "the Tool in the next round only after this test passes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tool_id": {"type": "string"},
                "hook": {"type": "string", "enum": list(SUPPORTED_HOOKS)},
            },
            "required": ["tool_id", "hook"],
            "additionalProperties": False,
        },
    )
    def test_capability_hook(tool_id: str, hook: str):
        return workspace.test_hook(tool_id, hook)

    @registry.tool(
        name="list_tested_capability_tools",
        description=(
            "List immutable agent-authored Tools that were unit-tested before "
            "this evolution round and can therefore be bound to a controller "
            "capability hook in this round, together with the exact deduplicated "
            "hook payload and output contracts required for runtime invocation."
        ),
    )
    def list_tested_capability_tools():
        used_hooks = sorted({
            str(hook)
            for item in bindable_tools
            for hook in item.get("compatible_hooks") or ()
            if hook in HOOK_CONTRACTS
        })
        return {
            "success": True,
            "tools": [
                {
                    "tool_id": item["tool_id"],
                    "description": item.get("description", ""),
                    "sha256": item.get("sha256"),
                    "compatible_hooks": item.get("compatible_hooks") or [],
                    "generic_contract": item.get("generic_contract"),
                }
                for item in bindable_tools
                if item.get("compatible_hooks") or item.get("generic_contract")
            ],
            "contracts": {hook: HOOK_CONTRACTS[hook] for hook in used_hooks},
        }

    for manifest in bindable_tools:
        tool_id = manifest["tool_id"]
        tool_name = "capability_" + manifest["name"]

        def invoke(payload: dict[str, Any], _tool_id: str = tool_id):
            return workspace.invoke(_tool_id, payload)

        registry.tool(
            name=tool_name,
            description=f"Agent-authored audited capability {tool_id}: {manifest.get('description', '')}",
            input_schema={
                "type": "object", "properties": {"payload": {"type": "object"}},
                "required": ["payload"], "additionalProperties": False,
            },
        )(invoke)


__all__ = ["CapabilityValidationError", "CapabilityWorkspace", "audit_capability_source", "register_capability_workspace_tools"]
