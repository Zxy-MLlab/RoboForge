"""Freeze sensor-grounded controller programs as reusable embodied Task Skills."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping

from asset_registry import register_asset
from controller_program_workspace import (
    ControllerProgramWorkspace,
    audit_controller_program,
)
from controller_program_runtime import ControllerProgramRuntime


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,79}$")


class TaskSkillValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def referenced_capability_tools(source: str) -> list[str]:
    tree = ast.parse(source)
    return sorted({
        str(node.args[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "call_tool"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and ":v" in node.args[0].value
    })


class TaskSkillWorkspace:
    """Immutable skill candidates whose geometry remains live-sensor derived."""

    def __init__(
        self,
        root: str | Path,
        *,
        controller_workspace: ControllerProgramWorkspace | None = None,
        capability_workspace: str | Path | None = None,
        library_path: str | Path | None = None,
        python: str | Path = "/data/zxy/envs/vla-report/bin/python",
        timeout_sec: float = 1800,
        max_rpc_calls: int = 10000,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.controllers = controller_workspace
        self.capabilities = (
            None if capability_workspace is None
            else Path(capability_workspace).resolve()
        )
        self.library_path = None if library_path is None else Path(library_path)
        self.runtime = ControllerProgramRuntime(
            python=python, timeout_sec=timeout_sec, max_rpc_calls=max_rpc_calls
        )

    def resolve(self, skill_id: str) -> Path:
        name, separator, version = str(skill_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise TaskSkillValidationError("invalid Task Skill ID")
        destination = (self.root / name / version).resolve()
        if self.root not in destination.parents or not (destination / "manifest.json").is_file():
            raise FileNotFoundError(skill_id)
        return destination

    def _verify_frozen_integrity(self, destination: Path) -> dict[str, Any]:
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest.get("protocol") != "embodied-task-skill-v1":
            raise TaskSkillValidationError("unsupported Task Skill protocol")
        if _sha256(destination / "program.py") != manifest.get("program_sha256"):
            raise TaskSkillValidationError("frozen Task Skill program hash mismatch")
        for dependency in manifest.get("dependencies") or ():
            tool_id = str(dependency.get("tool_id") or "")
            tool_name, separator, tool_version = tool_id.partition(":")
            if not separator:
                raise TaskSkillValidationError(f"invalid frozen capability dependency: {tool_id}")
            frozen = destination / "tools" / tool_name / tool_version
            if _sha256(frozen / "tool.py") != dependency.get("sha256"):
                raise TaskSkillValidationError(f"frozen capability hash mismatch: {tool_id}")
            expected_manifest_hash = dependency.get("manifest_sha256")
            if expected_manifest_hash and _sha256(frozen / "manifest.json") != expected_manifest_hash:
                raise TaskSkillValidationError(
                    f"frozen capability manifest hash mismatch: {tool_id}"
                )
        return manifest

    def inspect(self, skill_id: str) -> dict[str, Any]:
        destination = self.resolve(skill_id)
        manifest = self._verify_frozen_integrity(destination)
        return {
            "success": True,
            "skill_id": skill_id,
            "path": str(destination),
            "manifest": manifest,
            "source": (destination / "program.py").read_text(),
            "capability_workspace": str(destination / "tools"),
        }

    def execute(
        self, skill_id: str, dispatch: Any,
    ) -> dict[str, Any]:
        """Execute only the hash-frozen program; validation metadata cannot alter it."""
        destination = self.resolve(skill_id)
        manifest = self._verify_frozen_integrity(destination)
        report = self.runtime.run(
            destination / "program.py",
            expected_sha256=str(manifest["program_sha256"]),
            dispatch=dispatch,
        )
        return {**report, "program_id": skill_id, "task_skill_id": skill_id}

    def create_candidate(
        self,
        *,
        name: str,
        description: str,
        semantic_task: str,
        program_id: str,
        development_evidence: Mapping[str, Any],
        development_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(str(name)):
            raise TaskSkillValidationError(
                "Task Skill name must match ^[a-z][a-z0-9_]{2,79}$"
            )
        if self.controllers is None or self.capabilities is None:
            raise TaskSkillValidationError(
                "candidate creation requires controller and capability workspaces"
            )
        program_dir = self.controllers.resolve(program_id)
        program_manifest = json.loads((program_dir / "manifest.json").read_text())
        source = (program_dir / "program.py").read_text()
        audit = audit_controller_program(
            source, capability_contracts=self.controllers.capability_contracts()
        )
        if not audit.get("eligible"):
            raise TaskSkillValidationError(
                f"controller is not eligible for a Task Skill: {audit['violations']}"
            )
        if development_evidence.get("sensor_only_conclusion") != "sensor_verification_passed":
            raise TaskSkillValidationError(
                "Task Skill candidates require sensor_verification_passed evidence"
            )
        family = self.root / name
        versions = [
            int(path.name[1:]) for path in family.glob("v[0-9]*")
            if path.name[1:].isdigit()
        ]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        shutil.copy2(program_dir / "program.py", destination / "program.py")
        dependencies = []
        dependency_root = destination / "tools"
        for tool_id in referenced_capability_tools(source):
            tool_name, _, tool_version = tool_id.partition(":")
            tool_dir = (self.capabilities / tool_name / tool_version).resolve()
            if self.capabilities not in tool_dir.parents:
                raise TaskSkillValidationError(f"capability escapes workspace: {tool_id}")
            manifest_path = tool_dir / "manifest.json"
            module_path = tool_dir / "tool.py"
            if not manifest_path.is_file() or not module_path.is_file():
                raise TaskSkillValidationError(f"missing capability dependency: {tool_id}")
            tool_manifest = json.loads(manifest_path.read_text())
            if tool_manifest.get("status") != "unit_tested":
                raise TaskSkillValidationError(f"capability is not unit-tested: {tool_id}")
            frozen = dependency_root / tool_name / tool_version
            frozen.mkdir(parents=True, exist_ok=False)
            shutil.copy2(module_path, frozen / "tool.py")
            shutil.copy2(manifest_path, frozen / "manifest.json")
            dependencies.append({
                "tool_id": tool_id,
                "sha256": _sha256(frozen / "tool.py"),
                "manifest_sha256": _sha256(frozen / "manifest.json"),
                "generic_contract": tool_manifest.get("generic_contract"),
                "compatible_hooks": tool_manifest.get("compatible_hooks") or [],
            })
        evidence = {
            "sensor_only_conclusion": development_evidence.get("sensor_only_conclusion"),
            "attachment_verified": bool(development_evidence.get("attachment_verified")),
            "placement_verified": bool(development_evidence.get("placement_verified")),
            "verifications": development_evidence.get("verifications") or [],
            "evaluator_used": False,
        }
        manifest = {
            "protocol": "embodied-task-skill-v1",
            "name": name,
            "version": version,
            "skill_id": f"{name}:v{version:03d}",
            "description": str(description),
            "semantic_task": str(semantic_task),
            "program_id": program_id,
            "program_sha256": _sha256(destination / "program.py"),
            "controller_audit": audit,
            "live_instruction_required": True,
            "live_sensor_grounding_required": True,
            "literal_absolute_action_targets": False,
            "dependencies": dependencies,
            "development_evidence": evidence,
            "development_context": dict(development_context or {}),
            "validation_runs": [],
            "minimum_unseen_sensor_validations": 3,
            "status": "development_candidate",
            "current_task_data_used": True,
            "privileged_state_used": False,
            "created_unix": time.time(),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if self.library_path is not None:
            register_asset({
                "asset_id": f"skill.agent-learned-{name}.v{version}",
                "kind": "skill", "name": name, "version": str(version),
                "status": "development_candidate", "source_urls": [],
                "implementation": str(destination),
                "sha256": manifest["program_sha256"],
                "tested_tasks": [str(semantic_task)], "reused_tasks": [],
                "current_task_data_used": True, "privileged_state_used": False,
                "dependencies": [item["tool_id"] for item in dependencies],
            }, library_path=str(self.library_path), event="task_skill_candidate_frozen")
        return {
            "success": True, "skill_id": manifest["skill_id"],
            "status": manifest["status"], "path": str(destination),
            "dependencies": dependencies,
        }

    def record_unseen_validation(
        self,
        skill_id: str,
        *,
        environment: str,
        state_key: str,
        sensor_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        destination = self.resolve(skill_id)
        manifest_path = destination / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        development = manifest.get("development_context") or {}
        if (
            str(development.get("environment") or "") == str(environment)
            and str(development.get("state_key") or "") == str(state_key)
        ):
            raise TaskSkillValidationError(
                "development state cannot count as an unseen validation"
            )
        passed = sensor_evidence.get("sensor_only_conclusion") == "sensor_verification_passed"
        validations = list(manifest.get("validation_runs") or [])
        validations.append({
            "environment": str(environment), "state_key": str(state_key),
            "sensor_passed": bool(passed),
            "sensor_only_conclusion": sensor_evidence.get("sensor_only_conclusion"),
            "diagnostic_failure_class": sensor_evidence.get("diagnostic_failure_class"),
            "artifacts": sensor_evidence.get("artifacts") or {},
            "recorded_unix": time.time(),
            "evaluator_used": False,
        })
        manifest["validation_runs"] = validations
        unique_passes = {
            (item["environment"], item["state_key"])
            for item in validations if item.get("sensor_passed")
        }
        required = int(manifest.get("minimum_unseen_sensor_validations") or 3)
        if len(unique_passes) >= required:
            manifest["status"] = "sensor_validated"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "success": True, "skill_id": skill_id, "passed": bool(passed),
            "unique_sensor_passes": len(unique_passes), "required": required,
            "status": manifest["status"],
        }


__all__ = [
    "TaskSkillValidationError", "TaskSkillWorkspace",
    "referenced_capability_tools",
]
