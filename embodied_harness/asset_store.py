"""Persistent research leads, experiences, and frozen Graph Skills."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

from .errors import AssetError
from .graph_store import GraphStore
from .tool_store import ToolStore


class AssetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"

    def record(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        event = {"unix": time.time(), "kind": str(kind), "payload": dict(payload)}
        with self.events.open("a") as stream: stream.write(json.dumps(event)+"\n")
        return event

    def leads(self) -> list[dict[str, Any]]:
        if not self.events.is_file(): return []
        return [json.loads(line) for line in self.events.read_text().splitlines() if line]

    def freeze_skill(
        self, *, name: str, task: str, graph_id: str, graphs: GraphStore,
        tools: ToolStore, development_evidence: Mapping[str, Any],
        tool_ids: list[str],
    ) -> dict[str, Any]:
        if development_evidence.get("sensor_only_conclusion") != "sensor_verification_passed":
            raise AssetError("Skill freeze requires sensor verification")
        compiled = graphs.compile(graph_id); graph = graphs.inspect(graph_id)["manifest"]
        family = self.root / "skills" / name
        versions = [int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        version = max(versions, default=0)+1; destination = family/f"v{version:03d}"
        destination.mkdir(parents=True)
        graph_dst = destination/"graph"; shutil.copytree(graphs.resolve(graph_id), graph_dst)
        nodes = []
        for alias, node_id in graph["bindings"].items():
            node_dst = destination/"nodes"/alias; node_dst.parent.mkdir(exist_ok=True)
            shutil.copytree(graphs.nodes.resolve(node_id), node_dst)
            nodes.append({"alias": alias, "node_id": node_id,
                          "source_sha256": graphs.nodes.inspect(node_id)["manifest"]["source_sha256"]})
        frozen_tools = []
        for tool_id in sorted(set(tool_ids)):
            inspected = tools.inspect(tool_id)
            if inspected["manifest"].get("status") != "tested":
                raise AssetError(f"Skill dependency is not tested: {tool_id}")
            tool_dst = destination / "tools" / tool_id.replace(":", "_")
            tool_dst.parent.mkdir(exist_ok=True)
            shutil.copytree(tools.resolve(tool_id), tool_dst)
            frozen_tools.append({
                "tool_id": tool_id,
                "source_sha256": inspected["manifest"]["source_sha256"],
            })
        manifest = {
            "protocol": "standalone-embodied-graph-skill-v1",
            "skill_id": f"{name}:v{version:03d}", "task": task,
            "graph_id": graph_id, "graph_sha256": compiled["graph_sha256"],
            "nodes": nodes, "tools": frozen_tools,
            "development_evidence": dict(development_evidence),
            "validation_runs": [], "status": "development_candidate",
            "minimum_unseen_validations": 3, "created_unix": time.time(),
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        (destination/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        self.record("skill_frozen", {"skill_id": manifest["skill_id"], "task": task})
        return {"skill_id": manifest["skill_id"], "status": manifest["status"],
                "path": str(destination)}

    def invalidate_skill(self, skill_path: str | Path, *, reason: str,
                         evidence: Mapping[str, Any]) -> dict[str, Any]:
        path = Path(skill_path).resolve()
        if self.root not in path.parents or not (path / "manifest.json").is_file():
            raise AssetError("Skill is outside this AssetStore")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "rejected_sensor_false_positive"
        manifest["invalidation"] = {
            "unix": time.time(), "reason": reason, "evidence": dict(evidence),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        self.record("skill_invalidated", {
            "skill_id": manifest["skill_id"], "reason": reason,
        })
        return {"skill_id": manifest["skill_id"], "status": manifest["status"]}


__all__ = ["AssetStore"]
