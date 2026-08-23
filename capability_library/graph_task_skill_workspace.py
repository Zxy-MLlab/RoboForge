"""Freeze a Controller Graph and its complete node/Tool closure as a Task Skill."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable, Mapping

from asset_registry import register_asset
from controller_graph_workspace import ControllerGraphWorkspace
from stage_node_workspace import StageNodeWorkspace
from task_skill_workspace import referenced_capability_tools


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,79}$")


class GraphTaskSkillValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GraphTaskSkillWorkspace:
    def __init__(
        self,
        root: str | Path,
        *,
        graph_workspace: ControllerGraphWorkspace | None = None,
        capability_workspace: str | Path | None = None,
        library_path: str | Path | None = None,
        python: str | Path = "/data/zxy/envs/vla-report/bin/python",
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.graphs = graph_workspace
        self.capabilities = (
            None if capability_workspace is None
            else Path(capability_workspace).resolve()
        )
        self.library_path = None if library_path is None else Path(library_path)
        self.python = str(python)

    def resolve(self, skill_id: str) -> Path:
        name, separator, version = str(skill_id).partition(":")
        if not separator or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise GraphTaskSkillValidationError("invalid graph Task Skill ID")
        destination = (self.root / name / version).resolve()
        if self.root not in destination.parents or not (destination / "manifest.json").is_file():
            raise FileNotFoundError(skill_id)
        return destination

    def create_candidate(
        self,
        *,
        name: str,
        description: str,
        semantic_task: str,
        graph_id: str,
        development_evidence: Mapping[str, Any],
        development_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(str(name)):
            raise GraphTaskSkillValidationError("invalid Task Skill name")
        if self.graphs is None or self.capabilities is None:
            raise GraphTaskSkillValidationError(
                "candidate creation requires graph and capability workspaces"
            )
        if development_evidence.get("sensor_only_conclusion") != "sensor_verification_passed":
            raise GraphTaskSkillValidationError(
                "Task Skill requires sensor_verification_passed evidence"
            )
        graph_manifest = self.graphs.inspect(graph_id)["manifest"]
        verified_aliases = set(
            ((development_evidence.get("controller_graph") or {})
             .get("verified_prefix_aliases") or [])
        )
        executed_aliases = {
            str(item.get("alias"))
            for item in ((development_evidence.get("controller_graph") or {})
                         .get("node_trace") or [])
            if isinstance(item, Mapping) and item.get("alias")
        }
        if not executed_aliases or executed_aliases - verified_aliases:
            raise GraphTaskSkillValidationError(
                "every executed success-path node must belong to the final "
                "sensor-verified prefix"
            )
        family = self.root / name
        versions = [
            int(path.name[1:]) for path in family.glob("v[0-9]*")
            if path.name[1:].isdigit()
        ]
        version = max(versions, default=0) + 1
        destination = family / f"v{version:03d}"
        destination.mkdir(parents=True, exist_ok=False)

        graph_name, _, graph_version = graph_id.partition(":")
        frozen_graph = destination / "controller_graphs" / graph_name / graph_version
        frozen_graph.mkdir(parents=True)
        shutil.copy2(self.graphs.resolve(graph_id) / "manifest.json", frozen_graph / "manifest.json")
        frozen_nodes = []
        tool_ids = set()
        for alias, node_id in graph_manifest["bindings"].items():
            node = self.graphs.nodes.inspect(node_id)
            node_name, _, node_version = node_id.partition(":")
            frozen_node = destination / "stage_nodes" / node_name / node_version
            frozen_node.mkdir(parents=True)
            source_path = self.graphs.nodes.resolve(node_id) / "stage.py"
            manifest_path = self.graphs.nodes.resolve(node_id) / "manifest.json"
            shutil.copy2(source_path, frozen_node / "stage.py")
            shutil.copy2(manifest_path, frozen_node / "manifest.json")
            source = source_path.read_text()
            tool_ids.update(referenced_capability_tools(source))
            frozen_nodes.append({
                "alias": alias, "node_id": node_id,
                "source_sha256": _sha256(frozen_node / "stage.py"),
                "manifest_sha256": _sha256(frozen_node / "manifest.json"),
            })
        frozen_tools = []
        for tool_id in sorted(tool_ids):
            tool_name, _, tool_version = tool_id.partition(":")
            source_dir = self.capabilities / tool_name / tool_version
            if not (source_dir / "tool.py").is_file():
                raise GraphTaskSkillValidationError(f"missing Tool dependency: {tool_id}")
            tool_manifest = json.loads((source_dir / "manifest.json").read_text())
            if tool_manifest.get("status") != "unit_tested":
                raise GraphTaskSkillValidationError(f"Tool is not unit-tested: {tool_id}")
            frozen_tool = destination / "tools" / tool_name / tool_version
            frozen_tool.mkdir(parents=True)
            shutil.copy2(source_dir / "tool.py", frozen_tool / "tool.py")
            shutil.copy2(source_dir / "manifest.json", frozen_tool / "manifest.json")
            frozen_tools.append({
                "tool_id": tool_id,
                "source_sha256": _sha256(frozen_tool / "tool.py"),
                "manifest_sha256": _sha256(frozen_tool / "manifest.json"),
            })
        manifest = {
            "protocol": "embodied-graph-task-skill-v1",
            "name": name, "version": version,
            "skill_id": f"{name}:v{version:03d}",
            "description": str(description), "semantic_task": str(semantic_task),
            "graph_id": graph_id,
            "graph_manifest_sha256": _sha256(frozen_graph / "manifest.json"),
            "nodes": frozen_nodes, "tools": frozen_tools,
            "development_verified_aliases": sorted(verified_aliases),
            "development_unexecuted_aliases": sorted(
                set(graph_manifest["bindings"]) - executed_aliases
            ),
            "development_context": dict(development_context or {}),
            "development_evidence": {
                "sensor_only_conclusion": "sensor_verification_passed",
                "attachment_verified": bool(development_evidence.get("attachment_verified")),
                "placement_verified": bool(development_evidence.get("placement_verified")),
                "evaluator_used": False,
            },
            "validation_runs": [],
            "minimum_unseen_sensor_validations": 3,
            "status": "development_candidate",
            "live_sensor_grounding_required": True,
            "privileged_state_used": False,
            "created_unix": time.time(),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if self.library_path is not None:
            register_asset({
                "asset_id": f"skill.agent-learned-graph-{name}.v{version}",
                "kind": "skill", "name": name, "version": str(version),
                "status": "development_candidate", "source_urls": [],
                "implementation": str(destination),
                "sha256": manifest["graph_manifest_sha256"],
                "tested_tasks": [str(semantic_task)], "reused_tasks": [],
                "current_task_data_used": True, "privileged_state_used": False,
                "dependencies": sorted(tool_ids),
            }, library_path=str(self.library_path), event="graph_task_skill_candidate_frozen")
        return {
            "success": True, "skill_id": manifest["skill_id"],
            "status": manifest["status"], "path": str(destination),
        }

    def inspect(self, skill_id: str) -> dict[str, Any]:
        destination = self.resolve(skill_id)
        manifest = json.loads((destination / "manifest.json").read_text())
        graph_name, _, graph_version = manifest["graph_id"].partition(":")
        graph_manifest = destination / "controller_graphs" / graph_name / graph_version / "manifest.json"
        if _sha256(graph_manifest) != manifest["graph_manifest_sha256"]:
            raise GraphTaskSkillValidationError("frozen graph hash mismatch")
        for node in manifest["nodes"]:
            node_name, _, node_version = node["node_id"].partition(":")
            root = destination / "stage_nodes" / node_name / node_version
            if _sha256(root / "stage.py") != node["source_sha256"]:
                raise GraphTaskSkillValidationError(f"frozen node hash mismatch: {node['node_id']}")
            if _sha256(root / "manifest.json") != node["manifest_sha256"]:
                raise GraphTaskSkillValidationError(
                    f"frozen node manifest hash mismatch: {node['node_id']}"
                )
        for tool in manifest["tools"]:
            tool_name, _, tool_version = tool["tool_id"].partition(":")
            root = destination / "tools" / tool_name / tool_version
            if _sha256(root / "tool.py") != tool["source_sha256"]:
                raise GraphTaskSkillValidationError(
                    f"frozen Tool hash mismatch: {tool['tool_id']}"
                )
            if _sha256(root / "manifest.json") != tool["manifest_sha256"]:
                raise GraphTaskSkillValidationError(
                    f"frozen Tool manifest hash mismatch: {tool['tool_id']}"
                )
        return {
            "success": True, "skill_id": skill_id, "manifest": manifest,
            "path": str(destination), "capability_workspace": str(destination / "tools"),
        }

    def execute(
        self, skill_id: str,
        dispatch: Callable[[str, Mapping[str, Any]], Any],
    ) -> dict[str, Any]:
        inspected = self.inspect(skill_id)
        destination = Path(inspected["path"])
        nodes = StageNodeWorkspace(
            destination / "stage_nodes", python=self.python,
            capability_workspace=destination / "tools",
        )
        graphs = ControllerGraphWorkspace(destination / "controller_graphs", nodes=nodes)
        graph_id = inspected["manifest"]["graph_id"]
        graph_manifest = graphs.inspect(graph_id)["manifest"]
        initial_context = {}
        if "task_instruction" in graph_manifest.get("initial_context_fields", []):
            initial_context["task_instruction"] = dispatch("instruction", {})
        return graphs.execute(graph_id, dispatch, initial_context=initial_context)

    def record_unseen_validation(
        self, skill_id: str, *, environment: str, state_key: str,
        sensor_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        destination = self.resolve(skill_id)
        path = destination / "manifest.json"
        manifest = json.loads(path.read_text())
        development = manifest.get("development_context") or {}
        if (
            development.get("environment") == environment
            and development.get("state_key") == state_key
        ):
            raise GraphTaskSkillValidationError(
                "development state cannot count as unseen validation"
            )
        passed = sensor_evidence.get("sensor_only_conclusion") == "sensor_verification_passed"
        manifest["validation_runs"].append({
            "environment": environment, "state_key": state_key,
            "sensor_passed": bool(passed),
            "sensor_only_conclusion": sensor_evidence.get("sensor_only_conclusion"),
            "evaluator_used": False, "recorded_unix": time.time(),
        })
        unique = {
            (item["environment"], item["state_key"])
            for item in manifest["validation_runs"] if item["sensor_passed"]
        }
        required = int(manifest["minimum_unseen_sensor_validations"])
        if len(unique) >= required:
            manifest["status"] = "sensor_validated"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return {
            "success": True, "skill_id": skill_id, "passed": bool(passed),
            "unique_sensor_passes": len(unique), "required": required,
            "status": manifest["status"],
        }


__all__ = ["GraphTaskSkillValidationError", "GraphTaskSkillWorkspace"]
