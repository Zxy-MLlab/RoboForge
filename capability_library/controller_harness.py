"""Controller-authoring tools for the embodied Coding-Agent Harness.

The LLM authors a bounded controller specification.  This module compiles it
to an immutable, standalone Python program, audits the generated source, and
executes it in a separate process.  Evaluator fields are never returned to the
authoring agent; they remain in the run directory for the final scorer only.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Mapping

from runtime_capabilities import SUPPORTED_HOOKS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "generated_controllers"
DEFAULT_PYTHON = Path("/data/zxy/envs/vla-report/bin/python")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ALLOWED_STAGES = {
    "observe_rgbd",
    "detect_open_vocabulary",
    "select_physical_regions",
    "detect_articulated_handle",
    "open_drawer",
    "verify_articulation",
    "reobserve_after_articulation",
    "segment_source",
    "generate_ranked_grasps",
    "execute_guarded_grasp",
    "verify_attachment",
    "transport",
    "place",
    "verify_placement",
    "correct_or_regrasp",
    "finish",
}
_FORBIDDEN_SOURCE_TERMS = {
    "check_success",
    "reward",
    "bddl",
    "body_xpos",
    "body_pose",
    "get_body_xpos",
    "segmentation_id",
    "mujoco",
    "sim.data",
    "task_id ==",
    "state_id ==",
}
_AGENT_HIDDEN_KEYS = {
    "success",
    "reward",
    "done",
    "terminated",
    "truncated",
    "check_success",
    "evaluator_result",
}
_RUNTIME_DEPENDENCIES = (
    ROOT / "evaluation" / "libero_generated_controller.py",
    Path("/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py"),
    ROOT / "capability_library" / "tools" / "closed_loop_recovery.py",
    ROOT / "capability_library" / "tools" / "groundingdino_detector.py",
    ROOT / "capability_library" / "tools" / "sam_box_segment.py",
    ROOT / "capability_library" / "tools" / "graspnet_rgbd_grasp.py",
    ROOT / "capability_library" / "tools" / "instance_grounding.py",
    ROOT / "capability_library" / "runtime_capabilities.py",
)


class ControllerValidationError(ValueError):
    """Raised before an unsafe or malformed controller can execute."""


def normalize_controller_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the model-authored controller plan."""
    stages = [str(item).strip() for item in spec.get("stages", ())]
    if not stages:
        raise ControllerValidationError("controller stages must not be empty")
    if len(stages) != len(set(stages)):
        raise ControllerValidationError(
            "controller stages are declarative capabilities and must not repeat; "
            "use a runtime Tool hook for repeated closed-loop behavior"
        )
    unknown = sorted(set(stages) - _ALLOWED_STAGES)
    if unknown:
        raise ControllerValidationError(f"unknown controller stages: {unknown}")
    required = {"observe_rgbd", "execute_guarded_grasp", "finish"}
    missing = sorted(required - set(stages))
    if missing:
        raise ControllerValidationError(f"missing required stages: {missing}")
    if stages[0] != "observe_rgbd" or stages[-1] != "finish":
        raise ControllerValidationError("controller must start with observe_rgbd and end with finish")

    max_grasps = int(spec.get("max_grasp_attempts", 1))
    max_corrections = int(spec.get("max_place_corrections", 0))
    if not 1 <= max_grasps <= 5:
        raise ControllerValidationError("max_grasp_attempts must be within [1, 5]")
    if not 0 <= max_corrections <= 3:
        raise ControllerValidationError("max_place_corrections must be within [0, 3]")

    queries = []
    for item in spec.get("detector_queries", ("black bowl", "plate", "ramekin")):
        query = str(item).strip().lower()
        if query and query not in queries:
            queries.append(query)
    if not queries or len(queries) > 12:
        raise ControllerValidationError("detector_queries must contain 1 to 12 entries")

    raw_hooks = spec.get("capability_hooks") or {}
    if not isinstance(raw_hooks, Mapping):
        raise ControllerValidationError("capability_hooks must be an object")
    unknown_hooks = sorted(set(raw_hooks) - set(SUPPORTED_HOOKS))
    if unknown_hooks:
        raise ControllerValidationError(f"unknown capability hooks: {unknown_hooks}")
    capability_hooks: dict[str, str] = {}
    for hook, tool_id in raw_hooks.items():
        name, separator, version = str(tool_id).partition(":")
        if (
            not separator
            or not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", name)
            or not re.fullmatch(r"v[0-9]{3}", version)
        ):
            raise ControllerValidationError(
                f"capability hook {hook} must reference a tool_id like name:v001"
            )
        capability_hooks[str(hook)] = str(tool_id)

    return {
        "schema_version": 1,
        "stages": stages,
        "detector_queries": queries,
        "max_grasp_attempts": max_grasps,
        "max_place_corrections": max_corrections,
        "attachment_verification": bool(spec.get("attachment_verification", True)),
        "placement_verification": bool(spec.get("placement_verification", True)),
        "grasp_orientation": str(spec.get("grasp_orientation", "robot-topdown")),
        "preferred_downward_score": float(spec.get("preferred_downward_score", 0.75)),
        "fallback_downward_score": float(spec.get("fallback_downward_score", 0.55)),
        "capability_hooks": capability_hooks,
    }


def render_controller_source(spec: Mapping[str, Any]) -> str:
    """Compile a safe specification into a standalone executable program."""
    canonical = normalize_controller_spec(spec)
    return _render_controller_source(canonical)


def _render_controller_source(canonical: Mapping[str, Any]) -> str:
    payload = repr(canonical)
    return f'''"""Generated by the Embodied Coding-Agent Harness.  Do not edit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path({str(ROOT)!r})
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.libero_generated_controller import execute_controller

CONTROLLER_SPEC = {payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--state", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = execute_controller(
        CONTROLLER_SPEC,
        suite=args.suite,
        task=args.task,
        state=args.state,
        seed=args.seed,
        output=args.output,
        controller_root=Path(__file__).resolve().parent,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def audit_controller_source(source: str) -> dict[str, Any]:
    """Reject privileged-state access and arbitrary generated-code imports."""
    lowered = source.lower()
    violations = sorted(term for term in _FORBIDDEN_SOURCE_TERMS if term in lowered)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"eligible": False, "violations": [f"syntax_error:{exc.msg}"]}
    allowed_imports = {"argparse", "json", "sys", "pathlib", "__future__", "evaluation.libero_generated_controller"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_imports:
                    violations.append(f"forbidden_import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module not in allowed_imports:
                violations.append(f"forbidden_import:{module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "compile", "open"}:
            violations.append(f"forbidden_call:{node.func.id}")
    return {"eligible": not violations, "violations": sorted(set(violations))}


class ControllerWorkspace:
    """Version, audit, execute, and promote generated controllers."""

    def __init__(self, root: str | Path = DEFAULT_WORKSPACE, *, python: str | Path = DEFAULT_PYTHON, timeout_sec: int = 1800, max_executions: int | None = None, capability_workspace: Any | None = None):
        self.root = Path(root).resolve()
        self.python = Path(python)
        if not 30 <= int(timeout_sec) <= 3600:
            raise ValueError("timeout_sec must be within [30, 3600]")
        self.timeout_sec = int(timeout_sec)
        if max_executions is not None and not 1 <= int(max_executions) <= 100:
            raise ValueError("max_executions must be within [1, 100]")
        self.max_executions = None if max_executions is None else int(max_executions)
        self.capability_workspace = capability_workspace
        tested_capabilities = (
            capability_workspace.tested_tools()
            if capability_workspace is not None
            else ()
        )
        self._allowed_capabilities = {
            str(item["tool_id"]): item for item in tested_capabilities
        }
        self._execution_count = 0
        self._executed_keys: set[tuple[str, int, int, int]] = set()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, spec: Mapping[str, Any], rationale: str = "") -> dict[str, Any]:
        if not _NAME.fullmatch(name):
            raise ControllerValidationError("name must match ^[a-z][a-z0-9_]{0,63}$")
        canonical = normalize_controller_spec(spec)
        frozen_capabilities: dict[str, dict[str, Any]] = {}
        frozen_sources: dict[str, bytes] = {}
        for hook, tool_id in canonical["capability_hooks"].items():
            if self.capability_workspace is None:
                raise ControllerValidationError(
                    "controller requested a capability hook but no capability workspace is configured"
                )
            if tool_id not in self._allowed_capabilities:
                raise ControllerValidationError(
                    f"capability {tool_id} was not unit-tested before this authoring round"
                )
            if hook not in set(
                self._allowed_capabilities[tool_id].get("compatible_hooks") or ()
            ):
                raise ControllerValidationError(
                    f"capability {tool_id} did not pass the {hook} runtime hook contract"
                )
            capability = self.capability_workspace.resolve(tool_id)
            capability_manifest = json.loads((capability / "manifest.json").read_text())
            if capability_manifest.get("status") != "unit_tested":
                raise ControllerValidationError(f"capability {tool_id} is not unit-tested")
            module_bytes = (capability / "tool.py").read_bytes()
            digest = hashlib.sha256(module_bytes).hexdigest()
            if digest != capability_manifest.get("sha256"):
                raise ControllerValidationError(f"capability {tool_id} source hash changed")
            relative_module = f"capabilities/{hook}.py"
            frozen_sources[relative_module] = module_bytes
            frozen_capabilities[hook] = {
                "tool_id": tool_id,
                "module": relative_module,
                "sha256": digest,
                "description": capability_manifest.get("description", ""),
            }
        canonical["runtime_capability_hooks"] = frozen_capabilities
        source = _render_controller_source(canonical)
        audit = audit_controller_source(source)
        if not audit["eligible"]:
            raise ControllerValidationError(f"generated source failed audit: {audit['violations']}")
        digest = hashlib.sha256(source.encode()).hexdigest()
        family = self.root / name
        versions = [int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        for relative_module, module_bytes in frozen_sources.items():
            module = destination / relative_module
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_bytes(module_bytes)
        (destination / "controller.py").write_text(source)
        manifest = {
            "name": name,
            "version": version,
            "controller_sha256": digest,
            "created_unix": time.time(),
            "rationale": str(rationale),
            "spec": canonical,
            "audit": audit,
            "runtime_dependencies": _dependency_hashes(
                destination / relative for relative in frozen_sources
            ),
            "status": "candidate",
            "current_task_data_used_for_parameters": False,
            "privileged_state_used": False,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {"success": True, "controller_id": f"{name}:v{version:03d}", "path": str(destination), "sha256": digest, "audit": audit}

    def resolve(self, controller_id: str) -> Path:
        name, separator, version = str(controller_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise ControllerValidationError("controller_id must look like name:v001")
        path = (self.root / name / version).resolve()
        if self.root not in path.parents or not (path / "controller.py").is_file():
            raise FileNotFoundError(controller_id)
        return path

    def execute(self, controller_id: str, *, suite: str, task: int, state: int, seed: int = 7, output_root: str | Path | None = None, timeout_sec: int = 1800) -> dict[str, Any]:
        controller = self.resolve(controller_id)
        manifest = json.loads((controller / "manifest.json").read_text())
        expected_dependencies = manifest.get("runtime_dependencies")
        frozen_modules = [
            controller / str(binding["module"])
            for binding in (manifest.get("spec", {}).get("runtime_capability_hooks") or {}).values()
        ]
        current_dependencies = _dependency_hashes(frozen_modules)
        if not isinstance(expected_dependencies, dict):
            raise ControllerValidationError(
                "controller predates dependency freezing; create a new version"
            )
        if expected_dependencies != current_dependencies:
            changed = sorted(
                key
                for key in set(expected_dependencies) | set(current_dependencies)
                if expected_dependencies.get(key) != current_dependencies.get(key)
            )
            raise ControllerValidationError(
                f"controller runtime dependencies changed: {changed}; create a new version"
            )
        if suite != "libero_spatial":
            raise ControllerValidationError("the current adapter only permits libero_spatial")
        if self.max_executions is not None and self._execution_count >= self.max_executions:
            raise ControllerValidationError(
                "this authoring round already executed its controller; persist the sensor evidence and start a new evolution round"
            )
        execution_key = (str(controller_id), int(task), int(state), int(seed))
        if execution_key in self._executed_keys:
            raise ControllerValidationError(
                "this controller already ran for the task/state/seed; create a new version before retrying"
            )
        self._executed_keys.add(execution_key)
        self._execution_count += 1
        run_root = Path(output_root or (controller / "runs")).resolve()
        run_dir = run_root / f"task{int(task)}_state{int(state)}_seed{int(seed)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [str(self.python), str(controller / "controller.py"), "--suite", suite, "--task", str(int(task)), "--state", str(int(state)), "--seed", str(int(seed)), "--output", str(run_dir)]
        environment = dict(os.environ)
        # LIBERO otherwise resolves whichever stale user-level config happens
        # to be visible to the subprocess.  This file contains only documented
        # asset/init-state paths and is not exposed to controller logic.
        environment["LIBERO_CONFIG_PATH"] = str(ROOT / "runtime_home" / ".libero")
        environment.setdefault("MUJOCO_GL", "egl")
        environment.setdefault("PYOPENGL_PLATFORM", "egl")
        effective_timeout = min(int(timeout_sec), self.timeout_sec)
        completed = subprocess.run(command, text=True, capture_output=True, timeout=effective_timeout, env=environment)
        (run_dir / "process.json").write_text(json.dumps({"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}, indent=2) + "\n")
        sensor_report_path = run_dir / "agent_observation.json"
        sensor_report = json.loads(sensor_report_path.read_text()) if sensor_report_path.is_file() else {"execution_completed": False, "failure_kind": "missing_agent_observation"}
        sanitized = _strip_evaluator_fields(sensor_report)
        return {"success": completed.returncode == 0, "controller_id": controller_id, "run_dir": str(run_dir), "process_returncode": completed.returncode, "sensor_evidence": sanitized, "evaluator_hidden": True}

    def inspect_run(self, controller_id: str, *, task: int, state: int,
                    seed: int = 7) -> dict[str, Any]:
        """Reanalyze one run from sensor artifacts without reading its scorer."""
        controller = self.resolve(controller_id)
        run_dir = controller / "runs" / f"task{int(task)}_state{int(state)}_seed{int(seed)}"
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"sensor result not found: {run_dir}")
        manifest = json.loads((controller / "manifest.json").read_text())
        sanitized_result = _strip_evaluator_fields(json.loads(result_path.read_text()))
        # Import lazily so controller authoring remains usable without the
        # simulation stack.  _sensor_summary reads RGB/trace/result only and
        # never opens the sibling _evaluator_only directory.
        from evaluation.libero_generated_controller import _sensor_summary
        evidence = _sensor_summary(run_dir, sanitized_result, manifest["spec"])
        evidence = _strip_evaluator_fields(evidence)
        reanalysis_path = run_dir / "agent_reanalysis.json"
        reanalysis_path.write_text(json.dumps(evidence, indent=2) + "\n")
        return {"success": True, "controller_id": controller_id,
                "run": run_dir.name, "sensor_evidence": evidence,
                "evaluator_hidden": True,
                "reanalysis": str(reanalysis_path)}


def _strip_evaluator_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_evaluator_fields(item) for key, item in value.items() if str(key).lower() not in _AGENT_HIDDEN_KEYS}
    if isinstance(value, list):
        return [_strip_evaluator_fields(item) for item in value]
    return value


def _dependency_hashes(extra_paths: Any = ()) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in (*_RUNTIME_DEPENDENCIES, *tuple(extra_paths)):
        if not path.is_file():
            raise FileNotFoundError(f"controller runtime dependency missing: {path}")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def register_controller_authoring_tools(registry: Any, workspace: ControllerWorkspace) -> None:
    """Expose the controller Coding-Agent surface to Thea."""

    @registry.tool(
        name="create_controller_script",
        description="Compile and audit an immutable standalone LIBERO controller from a reusable sensor-only tool pipeline. Never encode task IDs, state IDs, benchmark answers, reward, evaluator results, BDDL, or simulator object state in the specification. Use exactly the documented specification fields and stage names.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 64},
                "rationale": {"type": "string"},
                "spec": {
                    "type": "object",
                    "properties": {
                        "stages": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(_ALLOWED_STAGES)},
                            "minItems": 1,
                        },
                        "detector_queries": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 12},
                        "max_grasp_attempts": {"type": "integer", "minimum": 1, "maximum": 5},
                        "max_place_corrections": {"type": "integer", "minimum": 0, "maximum": 3},
                        "attachment_verification": {"type": "boolean"},
                        "placement_verification": {"type": "boolean"},
                        "grasp_orientation": {"type": "string", "enum": ["robot-topdown", "model"]},
                        "preferred_downward_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "fallback_downward_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "capability_hooks": {
                            "type": "object",
                            "properties": {
                                "grasp_retry_ranking": {
                                    "type": "string",
                                    "description": "A previously unit-tested tool_id. Input contains public GraspNet candidate geometry and the bounded attempt budget; output must be {candidate_indices: [int, ...]}.",
                                },
                                "grasp_execution_profile": {
                                    "type": "string",
                                    "description": "A previously unit-tested tool_id. Input contains one live GraspNet candidate, tracked RGB-D source, EEF pose, and prior visual failure; output controls bounded approach, contact, close, lift, and reobservation behavior.",
                                },
                                "transport_profile": {
                                    "type": "string",
                                    "description": "A previously unit-tested tool_id. Input contains live EEF/goal/carried-offset/gripper features; output must contain lift_margin_m, horizontal_segments, position_gain, and max_translation_action within documented bounds.",
                                },
                                "support_relation_profile": {
                                    "type": "string",
                                    "description": "A previously unit-tested tool_id. Input contains two legal RGB-D/SAM support observations; output selects conservative bounded containment, clearance, motion, and centering thresholds.",
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                    "required": ["stages", "detector_queries", "max_grasp_attempts", "max_place_corrections"],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "spec"],
            "additionalProperties": False,
        },
    )
    def create_controller_script(name: str, spec: dict[str, Any], rationale: str = "") -> dict[str, Any]:
        return workspace.create(name, spec, rationale)

    @registry.tool(
        name="execute_controller_script",
        description="Execute one audited controller candidate in LIBERO-Spatial. The returned result contains sensor evidence only; evaluator success is hidden from the authoring agent.",
        input_schema={
            "type": "object",
            "properties": {
                "controller_id": {"type": "string", "minLength": 6, "maxLength": 69},
                "task": {"type": "integer", "minimum": 0, "maximum": 9},
                "state": {"type": "integer", "minimum": 0},
                "seed": {"type": "integer", "default": 7},
            },
            "required": ["controller_id", "task", "state"],
            "additionalProperties": False,
        },
    )
    def execute_controller_script(controller_id: str, task: int, state: int, seed: int = 7) -> dict[str, Any]:
        try:
            return workspace.execute(controller_id, suite="libero_spatial", task=task, state=state, seed=seed)
        except ControllerValidationError as exc:
            return {"success": False, "reason": str(exc), "kind": "controller_execution_rejected"}

    @registry.tool(
        name="inspect_controller_script",
        description="Read the immutable manifest and prior sensor-only run evidence for one generated controller.",
        input_schema={
            "type": "object",
            "properties": {"controller_id": {"type": "string", "minLength": 6, "maxLength": 69}},
            "required": ["controller_id"],
            "additionalProperties": False,
        },
    )
    def inspect_controller_script(controller_id: str) -> dict[str, Any]:
        path = workspace.resolve(controller_id)
        manifest = json.loads((path / "manifest.json").read_text())
        runs = []
        for report in sorted((path / "runs").glob("*/agent_observation.json")) if (path / "runs").is_dir() else ():
            runs.append({"run": report.parent.name, "sensor_evidence": _strip_evaluator_fields(json.loads(report.read_text()))})
        return {"success": True, "controller_id": controller_id, "manifest": manifest, "runs": runs}

    @registry.tool(
        name="inspect_controller_run",
        description="Reanalyze one prior controller run from RGB, trace, proprioception, and sanitized result artifacts only. The evaluator directory is never read or returned.",
        input_schema={
            "type": "object",
            "properties": {
                "controller_id": {"type": "string", "minLength": 6, "maxLength": 69},
                "task": {"type": "integer", "minimum": 0, "maximum": 9},
                "state": {"type": "integer", "minimum": 0},
                "seed": {"type": "integer", "default": 7},
            },
            "required": ["controller_id", "task", "state"],
            "additionalProperties": False,
        },
    )
    def inspect_controller_run(controller_id: str, task: int, state: int,
                               seed: int = 7) -> dict[str, Any]:
        return workspace.inspect_run(controller_id, task=task, state=state, seed=seed)


__all__ = ["ControllerValidationError", "ControllerWorkspace", "audit_controller_source", "normalize_controller_spec", "register_controller_authoring_tools", "render_controller_source"]
