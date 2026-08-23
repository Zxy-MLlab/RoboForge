"""Immutable Tool and Skill assets produced by the coding agent."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any, Mapping


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class AssetError(RuntimeError): pass


def _test_values_match(actual: Any, expected: Any) -> bool:
    """Structural equality with a small tolerance for numeric computation."""
    if (isinstance(actual,(int,float)) and not isinstance(actual,bool) and
            isinstance(expected,(int,float)) and not isinstance(expected,bool)):
        return math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-12)
    if isinstance(actual,Mapping) and isinstance(expected,Mapping):
        return (set(actual)==set(expected) and
                all(_test_values_match(actual[key],expected[key]) for key in actual))
    if (isinstance(actual,(list,tuple)) and isinstance(expected,(list,tuple))):
        return (len(actual)==len(expected) and
                all(_test_values_match(a,b) for a,b in zip(actual,expected)))
    return actual==expected


class CapabilityLibrary:
    def __init__(self, root: str | Path, workspace_root: str | Path):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_root).resolve()

    def _workspace_file(self, relative: str) -> Path:
        path = (self.workspace / relative).resolve()
        if self.workspace not in path.parents or not path.is_file():
            raise AssetError("Tool source must be a workspace file")
        return path

    def register_tool(self, *, name: str, source_path: str, description: str,
                      input_schema: Mapping[str, Any], output_schema: Mapping[str, Any],
                      source_urls: list[str], trained_on_current_task: bool) -> dict[str, Any]:
        if not _NAME.fullmatch(name): raise AssetError("invalid Tool name")
        if trained_on_current_task is not False:
            raise AssetError("evaluated-task-trained Tool is forbidden")
        source = self._workspace_file(source_path); text = source.read_text()
        compile(text, str(source), "exec")
        versions = [int(p.name[1:]) for p in (self.root/name).glob("v[0-9]*")
                    if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        path = self.root/name/f"v{version:03d}"; path.mkdir(parents=True)
        shutil.copy2(source, path/"tool.py")
        digest = hashlib.sha256((path/"tool.py").read_bytes()).hexdigest()
        manifest = {"protocol": "embodied-codex-tool-v1",
                    "tool_id": f"{name}:v{version:03d}", "name": name,
                    "version": version, "description": description,
                    "input_schema": dict(input_schema), "output_schema": dict(output_schema),
                    "source_urls": list(source_urls), "trained_on_current_task": False,
                    "privileged_state_used": False, "source_sha256": digest,
                    "status": "registered", "tests": [], "created_unix": time.time()}
        (path/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        return {"tool_id": manifest["tool_id"], "status": manifest["status"]}

    def register_deployment_tool(self, *, name: str, implementation_path: str,
                                 description: str, input_schema: Mapping[str,Any],
                                 output_schema: Mapping[str,Any], provenance: Mapping[str,Any]):
        if not _NAME.fullmatch(name): raise AssetError("invalid Tool name")
        if provenance.get("trained_on_current_task") is not False:
            raise AssetError("deployment Tool must be task-disjoint")
        if provenance.get("privileged_state_used") is not False:
            raise AssetError("deployment Tool used privileged state")
        source=Path(implementation_path).resolve();digest=hashlib.sha256(source.read_bytes()).hexdigest()
        for manifest_path in (self.root/name).glob("v*/manifest.json"):
            manifest=json.loads(manifest_path.read_text())
            if manifest.get("source_sha256")==digest:
                return {"tool_id":manifest["tool_id"],"status":manifest["status"]}
        versions=[int(p.name[1:]) for p in (self.root/name).glob("v[0-9]*") if p.name[1:].isdigit()]
        version=max(versions,default=0)+1;path=self.root/name/f"v{version:03d}";path.mkdir(parents=True)
        shutil.copy2(source,path/"tool.py")
        manifest={"protocol":"embodied-codex-deployment-tool-v1",
                  "tool_id":f"{name}:v{version:03d}","name":name,"version":version,
                  "description":description,"input_schema":dict(input_schema),
                  "output_schema":dict(output_schema),"source_urls":list(provenance.get("source_urls") or []),
                  "trained_on_current_task":False,"privileged_state_used":False,
                  "source_sha256":digest,"status":"tested","tests":[{"deployment_validation":True}],
                  "execution_owned_by_deployment":True,"provenance":dict(provenance),
                  "created_unix":time.time()}
        (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        return {"tool_id":manifest["tool_id"],"status":"tested"}

    def _path(self, tool_id: str) -> Path:
        name, sep, version = tool_id.partition(":")
        if not sep: raise AssetError("invalid Tool id")
        path = (self.root/name/version).resolve()
        if self.root not in path.parents or not (path/"manifest.json").is_file():
            raise FileNotFoundError(tool_id)
        return path

    def inspect(self, tool_id: str):
        path = self._path(tool_id); manifest = json.loads((path/"manifest.json").read_text())
        if hashlib.sha256((path/"tool.py").read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise AssetError("Tool hash mismatch")
        return {"manifest": manifest, "source": (path/"tool.py").read_text()}

    def run(self, tool_id: str, payload: Mapping[str, Any]):
        path = self._path(tool_id); self.inspect(tool_id)
        spec = importlib.util.spec_from_file_location("embodied_codex_capability", path/"tool.py")
        if spec is None or spec.loader is None: raise AssetError("Tool load failed")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        if not hasattr(module, "run"): raise AssetError("Tool must define run(payload)")
        return module.run(dict(payload))

    def test_tool(self, tool_id: str, cases: list[Mapping[str, Any]]):
        if not cases: raise AssetError("Tool tests are required")
        results = []
        for case in cases:
            actual = self.run(tool_id, case.get("input") or {})
            expected = case.get("expected")
            results.append({"passed": _test_values_match(actual,expected), "actual": actual,
                            "expected": expected})
        path = self._path(tool_id); manifest_path = path/"manifest.json"
        manifest = json.loads(manifest_path.read_text()); manifest["tests"].append(results)
        # Retesting a convenient subset may not erase a genuine prior failure.
        # Re-evaluate stored actual/expected pairs using the current comparator
        # so benign floating-point representation differences remain valid.
        historical=[row for batch in manifest["tests"] if isinstance(batch,list)
                    for row in batch if isinstance(row,Mapping)]
        passed=bool(historical) and all(_test_values_match(row.get("actual"),row.get("expected"))
                                        for row in historical)
        manifest["status"] = "tested" if passed else "test_failed"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        return {"tool_id": tool_id, "status": manifest["status"], "results": results}

    def tested(self):
        manifests = [json.loads(p.read_text()) for p in self.root.glob("*/v*/manifest.json")]
        return sorted([m for m in manifests if m.get("status") == "tested"],
                      key=lambda m: m["tool_id"])

    def list_all(self):
        manifests=[json.loads(p.read_text()) for p in self.root.glob("*/v*/manifest.json")]
        return sorted(manifests,key=lambda manifest:manifest["tool_id"])

    def list_summaries(self):
        """Return the compositional contract without replaying test history.

        Full manifests can contain large numeric test fixtures and provenance
        blobs.  Sending all of that through the coding-agent `list_tools` call
        duplicates the contracts already present in its instruction and can
        crowd out the controller-engineering context.  `inspect_tool` remains
        the explicit path for source, provenance, and individual test details.
        """
        fields=("protocol","tool_id","name","version","description",
                "input_schema","output_schema","status",
                "trained_on_current_task","privileged_state_used",
                "execution_owned_by_deployment")
        return [{key:manifest[key] for key in fields if key in manifest}
                for manifest in self.list_all()]

    def runtime_functions(self):
        result = {}
        for manifest in self.tested():
            if manifest.get("execution_owned_by_deployment"): continue
            tool_id = manifest["tool_id"]
            result[tool_id] = lambda payload, _id=tool_id: self.run(_id, payload)
        return result

    def import_skill_tools(self, skill_dir: str | Path) -> dict[str, Any]:
        """Import immutable analytic Tools from a previously frozen Skill.

        Deployment-owned Tools are intentionally not rebound here: a new
        Adapter must register its current implementation as a new version, and
        the coding agent receives an explicit old-to-current dependency hint.
        """
        skill = Path(skill_dir).resolve()
        skill_manifest = json.loads((skill/"manifest.json").read_text())
        imported=[]; deployment=[]
        for tool_id in skill_manifest.get("tool_ids") or []:
            source = skill/"tools"/str(tool_id).replace(":","_")
            manifest = json.loads((source/"manifest.json").read_text())
            if manifest.get("tool_id") != tool_id or manifest.get("status") != "tested":
                raise AssetError(f"invalid frozen Tool: {tool_id}")
            if (manifest.get("trained_on_current_task") is not False
                    or manifest.get("privileged_state_used") is not False):
                raise AssetError(f"forbidden frozen Tool provenance: {tool_id}")
            if hashlib.sha256((source/"tool.py").read_bytes()).hexdigest() != manifest.get("source_sha256"):
                raise AssetError(f"frozen Tool hash mismatch: {tool_id}")
            if manifest.get("execution_owned_by_deployment"):
                deployment.append({"tool_id":tool_id,"name":manifest.get("name")})
                continue
            destination=self.root/manifest["name"]/f"v{int(manifest['version']):03d}"
            if destination.exists():
                existing=json.loads((destination/"manifest.json").read_text())
                if existing.get("source_sha256") != manifest.get("source_sha256"):
                    raise AssetError(f"Tool version collision: {tool_id}")
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(source,destination)
            imported.append(tool_id)
        return {"imported_tool_ids":sorted(imported),
                "deployment_dependencies":deployment}


class SkillLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    def freeze(self, *, name: str, task: str, controller: str | Path,
               evidence: Mapping[str, Any], tool_ids: list[str], tools: CapabilityLibrary,
               experience: Mapping[str,Any]|None=None,
               task_model: Mapping[str,Any]|None=None):
        if not _NAME.fullmatch(name): raise AssetError("invalid Skill name")
        family = self.root/name
        versions = [int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        version = max(versions, default=0)+1; path=family/f"v{version:03d}"; path.mkdir(parents=True)
        shutil.copy2(controller, path/"controller.py")
        for tool_id in sorted(set(tool_ids)):
            source = tools._path(tool_id); destination=path/"tools"/tool_id.replace(":","_")
            shutil.copytree(source, destination)
        manifest={"protocol":"embodied-codex-skill-v1","skill_id":f"{name}:v{version:03d}",
                  "task":task,"controller_sha256":hashlib.sha256((path/"controller.py").read_bytes()).hexdigest(),
                  "tool_ids":sorted(set(tool_ids)),"development_evidence":dict(evidence),
                  "status":"sensor_success","created_unix":time.time()}
        if experience is not None:
            experience_path=path/"experience.json"
            experience_path.write_text(json.dumps(dict(experience),indent=2,default=str)+"\n")
            manifest["experience_sha256"]=hashlib.sha256(experience_path.read_bytes()).hexdigest()
        if task_model is not None:
            task_model_path=path/"task_model.json"
            task_model_path.write_text(json.dumps(dict(task_model),indent=2,default=str)+"\n")
            manifest["task_model_sha256"]=hashlib.sha256(task_model_path.read_bytes()).hexdigest()
        (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        return {"skill_id":manifest["skill_id"],"path":str(path)}

__all__ = ["CapabilityLibrary", "SkillLibrary", "AssetError"]
