"""Immutable reusable capability Tools with executable tests and provenance."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
import shutil

from .errors import AssetError


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class ToolStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self, *, name: str, description: str, source: str,
        input_schema: Mapping[str, Any], output_schema: Mapping[str, Any],
        source_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        if not _NAME.fullmatch(name): raise AssetError("invalid tool name")
        try: tree = ast.parse(source)
        except SyntaxError as exc: raise AssetError(f"tool syntax: {exc.msg}") from exc
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "run"]
        if len(functions) != 1 or len(functions[0].args.args) != 1:
            raise AssetError("Tool must define run(payload)")
        forbidden = (ast.Import, ast.ImportFrom)
        if any(isinstance(node, forbidden) for node in ast.walk(tree)):
            raise AssetError("standalone deterministic Tool cannot import modules")
        family = self.root / name
        versions = [int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        path = family / f"v{version:03d}"; path.mkdir(parents=True)
        source_path = path / "tool.py"; source_path.write_text(source)
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest = {
            "protocol": "standalone-embodied-tool-v1",
            "tool_id": f"{name}:v{version:03d}", "name": name,
            "version": version, "description": description,
            "input_schema": dict(input_schema), "output_schema": dict(output_schema),
            "source_urls": list(source_urls or []), "source_sha256": digest,
            "status": "created", "tests": [], "created_unix": time.time(),
            "current_task_training_used": False, "privileged_state_used": False,
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        return {"tool_id": manifest["tool_id"], "status": "created"}

    def register_deployment_tool(
        self, *, name: str, description: str, implementation_path: str | Path,
        input_schema: Mapping[str, Any], output_schema: Mapping[str, Any],
        provenance: Mapping[str, Any], validation_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Freeze a tested task-disjoint Tool supplied by a deployment.

        Heavy model Tools execute in the persistent Adapter process, not through
        the deterministic Python helper runtime used for small model-authored Tools.
        """
        if not _NAME.fullmatch(name): raise AssetError("invalid tool name")
        source = Path(implementation_path).resolve()
        if not source.is_file(): raise FileNotFoundError(source)
        if provenance.get("trained_on_current_task") is not False:
            raise AssetError("deployment Tool must declare task-disjoint training")
        if provenance.get("privileged_state_used") is not False:
            raise AssetError("deployment Tool must reject privileged state")
        family = self.root / name
        existing = sorted(family.glob("v*/manifest.json"))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for manifest_path in existing:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("source_sha256") == digest:
                return {"tool_id": manifest["tool_id"], "status": manifest["status"]}
        versions = [int(p.parent.name[1:]) for p in existing
                    if p.parent.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        path = family / f"v{version:03d}"; path.mkdir(parents=True)
        shutil.copy2(source, path / "tool.py")
        manifest = {
            "protocol": "standalone-embodied-deployment-tool-v1",
            "tool_id": f"{name}:v{version:03d}", "name": name,
            "version": version, "description": description,
            "input_schema": dict(input_schema), "output_schema": dict(output_schema),
            "source_urls": list(provenance.get("source_urls") or []),
            "source_sha256": digest, "status": "tested",
            "tests": [dict(validation_evidence)], "created_unix": time.time(),
            "current_task_training_used": False, "privileged_state_used": False,
            "execution_owned_by_adapter": True, "provenance": dict(provenance),
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return {"tool_id": manifest["tool_id"], "status": "tested"}

    def resolve(self, tool_id: str) -> Path:
        name, sep, version = tool_id.partition(":")
        if not sep or not _NAME.fullmatch(name) or not re.fullmatch(r"v[0-9]{3}", version):
            raise AssetError("invalid tool_id")
        path = (self.root/name/version).resolve()
        if self.root not in path.parents or not (path/"manifest.json").is_file():
            raise FileNotFoundError(tool_id)
        return path

    def inspect(self, tool_id: str) -> dict[str, Any]:
        path = self.resolve(tool_id); manifest = json.loads((path/"manifest.json").read_text())
        if hashlib.sha256((path/"tool.py").read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise AssetError("immutable Tool hash mismatch")
        return {"manifest": manifest, "source": (path/"tool.py").read_text()}

    def run(self, tool_id: str, payload: Mapping[str, Any]) -> Any:
        inspected = self.inspect(tool_id); path = self.resolve(tool_id)/"tool.py"
        if inspected["manifest"].get("execution_owned_by_adapter"):
            raise AssetError("deployment Tool can only run through RobotAdapter.use")
        spec = importlib.util.spec_from_file_location("standalone_capability_tool", path)
        if spec is None or spec.loader is None: raise AssetError("cannot load Tool")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.run(dict(payload))

    def test(self, tool_id: str, cases: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not cases: raise AssetError("at least one Tool test is required")
        results = []
        for case in cases:
            actual = self.run(tool_id, case.get("input") or {})
            expected = case.get("expected")
            passed = actual == expected
            results.append({"passed": passed, "actual": actual, "expected": expected})
        path = self.resolve(tool_id); manifest_path = path/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tests"].append({"unix": time.time(), "results": results})
        manifest["status"] = "tested" if all(r["passed"] for r in results) else "test_failed"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        return {"tool_id": tool_id, "passed": all(r["passed"] for r in results),
                "status": manifest["status"], "results": results}

    def tested(self) -> list[dict[str, Any]]:
        assets = []
        for path in self.root.glob("*/v*/manifest.json"):
            manifest = json.loads(path.read_text())
            if manifest.get("status") == "tested": assets.append(manifest)
        return sorted(assets, key=lambda item: item["tool_id"])

    def runtime_capabilities(
        self,
    ) -> dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
        """Expose only tested model-authored Tools for Adapter installation."""
        functions: dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
        for manifest in self.tested():
            if manifest.get("execution_owned_by_adapter"):
                continue
            tool_id = str(manifest["tool_id"])

            def execute(payload: Mapping[str, Any], *, _tool_id: str = tool_id) -> Any:
                return self.run(_tool_id, payload)

            functions[tool_id] = execute
        return functions

    def install_runtime_capabilities(self, adapter: Any) -> list[str]:
        functions = self.runtime_capabilities()
        if not functions:
            return []
        register = getattr(adapter, "register_capability", None)
        if not callable(register):
            raise AssetError(
                "RobotAdapter must implement register_capability to execute "
                "model-authored tested Tools"
            )
        for tool_id, function in functions.items():
            register(tool_id, function)
        return sorted(functions)


__all__ = ["ToolStore"]
