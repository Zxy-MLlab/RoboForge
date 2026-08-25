"""Immutable Tool and Skill assets produced by the coding agent."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Mapping

from jsonschema import Draft202012Validator, ValidationError

from .tool_runtime import ToolRuntime
from .retrieval import rank_records


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


class AssetError(RuntimeError): pass


def _file_sha256(path: Path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def _normalize_name(value: str):
    """Turn a model-authored human label into a stable asset slug."""
    requested=str(value).strip();slug=re.sub(r"[^a-z0-9]+","_",requested.casefold()).strip("_")
    if slug and slug[0].isdigit():slug="asset_"+slug
    slug=slug[:63].rstrip("_")
    if len(slug)<3 or not _NAME.fullmatch(slug):raise AssetError("asset name has no valid identifier")
    return slug,requested


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


def _contract_schema(value: Mapping[str,Any],label: str):
    schema=dict(value)
    try:Draft202012Validator.check_schema(schema)
    except Exception as exc:raise AssetError(f"invalid {label} JSON Schema: {exc}") from exc
    standard={"$ref","type","anyOf","oneOf","allOf","not","if","enum","const"}
    if schema and not standard.intersection(schema):
        raise AssetError(f"{label} must be a JSON Schema, not a descriptive mapping")
    return schema


def _validate_instance(value: Any,schema: Mapping[str,Any],label: str):
    try:Draft202012Validator(dict(schema)).validate(value)
    except ValidationError as exc:raise AssetError(f"{label} violates Tool contract: {exc.message}") from exc


def execution_evidence_assessment(paths: list[str|Path]):
    """Derive an outcome from copied execution records rather than model prose."""
    records=[]
    for value in paths:
        path=Path(value).resolve()
        if path.suffix.lower()!=".json":continue
        try:document=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        if not isinstance(document,dict):continue
        report=document.get("sensor_report") or {}
        outcome=report.get("independent_task_outcome") or {}
        independent=outcome.get("verified")
        sensor=report.get("sensor_verification_passed")
        candidate=document.get("sensor_success_candidate")
        if independent is False or sensor is False or candidate is False:
            classification="failure"
        elif candidate is True and sensor is True and independent is not False:
            classification="success"
        else:classification="unknown"
        if any(key in document for key in ("sensor_report","sensor_success_candidate")):
            records.append({
                "evidence_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "classification":classification,
                "sensor_success_candidate":candidate,
                "sensor_verification_passed":sensor,
                "independent_verified":independent})
    known={item["classification"] for item in records
           if item["classification"]!="unknown"}
    overall=("mixed" if len(known)>1 else next(iter(known))) if known else "unknown"
    return {"derived_by":"embodied_codex_execution_evidence_gate_v1",
            "outcome":overall,"records":records}


def bind_authoritative_validation(value: Mapping[str,Any]|None, assessment: Mapping[str,Any]):
    """Attach a non-model task outcome and flag claims contradicted by it."""
    validation=dict(value or {})
    outcome=assessment.get("outcome")
    if outcome=="unknown":return validation
    declared=json.dumps(validation,sort_keys=True,default=str).lower()
    conflict=(outcome in {"failure","mixed"} and any(token in declared for token in
        ('"sensor_success"','"verified=true"','"verified": true','"status": "success"')))
    validation["authoritative_evidence_assessment"]=dict(assessment)
    validation["authoritative_outcome"]=outcome
    if conflict:validation["model_claim_conflicts_with_evidence"]=True
    return validation


def _provenance_record(*,source_urls,trained_on_current_task,privileged_state_used,
                       details: Mapping[str,Any]|None):
    """Build a fail-closed, immutable anti-contamination audit record."""
    if trained_on_current_task is not False:raise AssetError("evaluated-task-trained capability is forbidden")
    if privileged_state_used is not False:raise AssetError("privileged-state capability is forbidden")
    urls=[str(value) for value in source_urls]
    if not urls or any(not value.startswith("https://") for value in urls):
        raise AssetError("provenance requires public HTTPS source URLs")
    value=dict(details or {})
    declaration=str(value.get("training_data_declaration") or "").strip()
    check=dict(value.get("contamination_check") or {})
    if not declaration:raise AssetError("provenance requires a training-data declaration")
    if (not str(check.get("evaluated_benchmark") or "").strip()
            or not str(check.get("method") or "").strip()
            or check.get("result") not in {"no_declared_overlap","not_applicable_source_code"}):
        raise AssetError("provenance requires a conclusive contamination check")
    hashes=dict(value.get("checkpoint_sha256") or {})
    if any(not re.fullmatch(r"[0-9a-fA-F]{64}",str(digest)) for digest in hashes.values()):
        raise AssetError("checkpoint provenance contains an invalid sha256")
    models=list(value.get("models") or ([] if not value.get("model") else [value["model"]]))
    cards=[str(item) for item in (value.get("model_card_urls") or [])]
    if models and (not cards or any(not item.startswith("https://") for item in cards)):
        raise AssetError("model capability provenance requires model-card URLs")
    record={**value,"source_urls":urls,"trained_on_current_task":False,
            "privileged_state_used":False,"training_data_declaration":declaration,
            "contamination_check":check,"checkpoint_sha256":hashes,
            "model_card_urls":cards,"audit_status":"complete"}
    unsigned=json.dumps(record,sort_keys=True,separators=(",",":"),default=str).encode()
    record["audit_sha256"]=hashlib.sha256(unsigned).hexdigest()
    return record


class CapabilityLibrary:
    def __init__(self, root: str | Path, workspace_root: str | Path, *,
                 python: str|Path|None=None,
                 allowed_input_roots: list[str|Path]|None=None,
                 scope_id: str|None=None):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_root).resolve()
        self.scope_id=(str(scope_id).strip() if scope_id else hashlib.sha256(
            str(self.workspace).encode()).hexdigest()[:24])
        if not re.fullmatch(r"[A-Za-z0-9_.-]{8,128}",self.scope_id):
            raise AssetError("invalid Capability Library scope_id")
        self.runtime=ToolRuntime(python=python,allowed_input_roots=allowed_input_roots)

    def _accessible(self,manifest: Mapping[str,Any]):
        visibility=str(manifest.get("visibility") or "shared")
        return visibility=="shared" or (visibility=="task_local"
            and manifest.get("task_scope_id")==self.scope_id)

    def _require_accessible(self,manifest: Mapping[str,Any]):
        if not self._accessible(manifest):
            raise AssetError("Tool belongs to another task-local development scope")

    def _workspace_file(self, relative: str) -> Path:
        path = (self.workspace / relative).resolve()
        if self.workspace not in path.parents or not path.is_file():
            raise AssetError("Tool source must be a workspace file")
        return path

    def _workspace_dir(self, relative: str) -> Path:
        path=(self.workspace/relative).resolve()
        if self.workspace not in path.parents or not path.is_dir():
            raise AssetError("Tool dependency bundle must be a workspace directory")
        return path

    @staticmethod
    def _tree_sha256(root: Path):
        digest=hashlib.sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(root)).encode()+b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    @staticmethod
    def _validate_run_entrypoint(source_text: str):
        tree=ast.parse(source_text)
        entries=[node for node in tree.body
                 if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
                 and node.name=="run"]
        if len(entries)!=1 or isinstance(entries[0],ast.AsyncFunctionDef):
            raise AssetError("Tool entrypoint must define exactly one top-level def run(payload)")
        args=entries[0].args
        positional=[*args.posonlyargs,*args.args]
        if (len(positional)!=1 or args.vararg is not None or args.kwarg is not None
                or args.kwonlyargs or args.defaults or args.kw_defaults):
            raise AssetError("Tool entrypoint must define exactly one top-level def run(payload)")

    def _freeze_dependencies(self, *, source_text: str, dependency_spec, destination: Path):
        spec=dict(dependency_spec or {"mode":"stdlib"});mode=spec.get("mode")
        imports=set()
        for node in ast.walk(ast.parse(source_text)):
            if isinstance(node,ast.Import):imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        external=sorted(name for name in imports
                        if name not in sys.stdlib_module_names and name!="__future__")
        if mode=="stdlib":
            if external:raise AssetError(f"non-stdlib imports require a vendored dependency lock: {external}")
            return {"mode":"stdlib","requirements":[],"runtime":"isolated-python"}
        if mode!="vendored":raise AssetError("dependency_spec.mode must be stdlib or vendored")
        lock=self._workspace_file(str(spec.get("requirements_lock_path") or ""))
        vendor=self._workspace_dir(str(spec.get("vendor_path") or ""))
        lines=[line.strip() for line in lock.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
        pattern=re.compile(r"^[A-Za-z0-9_.-]+==[^\s]+(?:\s+--hash=sha256:[0-9a-fA-F]{64})+$")
        if not lines or any(not pattern.fullmatch(line) for line in lines):
            raise AssetError("requirements lock must pin every package and include sha256 hashes")
        shutil.copy2(lock,destination/"requirements.lock")
        shutil.copytree(vendor,destination/"vendor")
        return {"mode":"vendored","requirements":lines,
                "requirements_lock_sha256":hashlib.sha256(lock.read_bytes()).hexdigest(),
                "vendor_tree_sha256":self._tree_sha256(destination/"vendor"),
                "runtime":"isolated-python"}

    @staticmethod
    def _manual_payload(*, description, input_schema, output_schema, manual=None):
        value=dict(manual or {})
        value.setdefault("purpose",str(description))
        value.setdefault("when_to_use",[str(description)])
        input_fields=dict(input_schema.get("properties") or {})
        output_fields=dict(output_schema.get("properties") or {})
        value.setdefault("inputs",input_fields)
        value.setdefault("outputs",output_fields)
        value.setdefault("examples",[])
        value.setdefault("failure_modes",["May reject malformed input or return a documented structured error."])
        value.setdefault("limitations",[])
        required={"purpose":str,"when_to_use":list,"inputs":dict,"outputs":dict,
                  "examples":list,"failure_modes":list,"limitations":list}
        for key,kind in required.items():
            if not isinstance(value.get(key),kind):raise AssetError(f"Tool manual field {key} is invalid")
        if set(value["inputs"])!=set(input_fields):
            raise AssetError("Tool manual input fields must exactly match input_schema properties")
        if set(value["outputs"])!=set(output_fields):
            raise AssetError("Tool manual output fields must exactly match output_schema properties")
        return {key:value[key] for key in required}

    def _manual_dir(self,tool_id):
        name,sep,version=str(tool_id).partition(":")
        if not sep:raise AssetError("invalid Tool id")
        path=(self.root/"_manuals"/name/version).resolve()
        if self.root not in path.parents:raise AssetError("invalid Tool manual path")
        return path

    def _publish_manual(self,tool_id,manual,*,authored_by,evidence=None):
        path=self._manual_dir(tool_id);path.mkdir(parents=True,exist_ok=True)
        versions=[int(p.stem[1:]) for p in path.glob("r[0-9]*.json") if p.stem[1:].isdigit()]
        revision=max(versions,default=0)+1
        payload={"protocol":"embodied-codex-tool-manual-v1","tool_id":tool_id,
                 "manual_revision":revision,"authored_by":authored_by,
                 "manual":dict(manual),"evidence":list(evidence or []),
                 "created_unix":time.time()}
        destination=path/f"r{revision:03d}.json"
        destination.write_text(json.dumps(payload,indent=2)+"\n")
        return payload

    def manual(self,tool_id):
        inspected=self.inspect(tool_id);path=self._manual_dir(tool_id)
        revisions=sorted(path.glob("r[0-9]*.json"))
        if revisions:return json.loads(revisions[-1].read_text())
        manifest=inspected["manifest"]
        manual=self._manual_payload(description=manifest.get("description") or tool_id,
            input_schema=manifest.get("input_schema") or {},
            output_schema=manifest.get("output_schema") or {})
        return self._publish_manual(tool_id,manual,authored_by="registration_migration")

    def revise_manual(self,tool_id,manual,*,evidence_paths):
        inspected=self.inspect(tool_id);manifest=inspected["manifest"]
        normalized=self._manual_payload(description=manifest.get("description") or tool_id,
            input_schema=manifest.get("input_schema") or {},
            output_schema=manifest.get("output_schema") or {},manual=manual)
        evidence=[]
        for value in evidence_paths:
            path=Path(value).resolve()
            if not path.is_file():raise AssetError(f"Tool manual evidence is not a file: {path}")
            evidence.append({"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        if not evidence:raise AssetError("Tool manual revision requires evidence")
        return self._publish_manual(tool_id,normalized,authored_by="engineering_agent",
                                    evidence=evidence)

    def register_tool(self, *, name: str, source_path: str, description: str,
                      input_schema: Mapping[str, Any], output_schema: Mapping[str, Any],
                      source_urls: list[str], trained_on_current_task: bool,
                      manual: Mapping[str,Any]|None=None,
                      dependency_spec: Mapping[str,Any]|None=None,
                      provenance: Mapping[str,Any]|None=None) -> dict[str, Any]:
        name,requested_name=_normalize_name(name)
        provenance_record=_provenance_record(source_urls=source_urls,
            trained_on_current_task=trained_on_current_task,privileged_state_used=False,
            details=(provenance or {"training_data_declaration":
                "Deterministic source-code algorithm; no learned parameters.",
                "contamination_check":{"evaluated_benchmark":"current evaluation task",
                "method":"source and dependency inspection",
                "result":"not_applicable_source_code"}}))
        source = self._workspace_file(source_path); text = source.read_text()
        compile(text, str(source), "exec")
        self._validate_run_entrypoint(text)
        input_schema=_contract_schema(input_schema,"input_schema")
        output_schema=_contract_schema(output_schema,"output_schema")
        initial=self._manual_payload(description=description,input_schema=input_schema,
                                     output_schema=output_schema,manual=manual)
        versions = [int(p.name[1:]) for p in (self.root/name).glob("v[0-9]*")
                    if p.name[1:].isdigit()]
        version = max(versions, default=0) + 1
        path = self.root/name/f"v{version:03d}"
        staging=self.root/name/f".v{version:03d}.staging-{time.time_ns()}"
        staging.mkdir(parents=True)
        try:
            shutil.copy2(source,staging/"tool.py")
            dependencies=self._freeze_dependencies(source_text=text,
                dependency_spec=dependency_spec,destination=staging)
            digest=hashlib.sha256((staging/"tool.py").read_bytes()).hexdigest()
        except Exception:
            shutil.rmtree(staging,ignore_errors=True);raise
        existing_manifests=[]
        for manifest_path in (self.root/name).glob("v*/manifest.json"):
            existing=json.loads(manifest_path.read_text());existing_manifests.append(existing)
            if (existing.get("source_sha256")==digest
                    and existing.get("input_schema")==dict(input_schema)
                    and existing.get("output_schema")==dict(output_schema)
                    and existing.get("dependencies")==dependencies
                    and existing.get("provenance")==provenance_record
                    and self._accessible(existing)):
                shutil.rmtree(staging,ignore_errors=True)
                return {"tool_id":existing["tool_id"],"status":existing["status"],
                        "duplicate_of":existing["tool_id"]}
        manifest = {"protocol": "embodied-codex-tool-v1",
                    "tool_id": f"{name}:v{version:03d}", "name": name,
                    "version": version, "description": description,
                    "requested_name":requested_name,
                    "input_schema": dict(input_schema), "output_schema": dict(output_schema),
                    "source_urls": list(source_urls), "trained_on_current_task": False,
                    "privileged_state_used": False, "source_sha256": digest,
                    "dependencies":dependencies,
                    "provenance":provenance_record,
                    "visibility":"task_local","task_scope_id":self.scope_id,
                    "status": "registered", "tests": [], "created_unix": time.time()}
        if existing_manifests:
            manifest["supersedes"]=max(existing_manifests,
                key=lambda item:int(item.get("version",0)))["tool_id"]
        (staging/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        staging.replace(path)
        self._publish_manual(manifest["tool_id"],initial,authored_by="tool_author")
        return {"tool_id": manifest["tool_id"], "status": manifest["status"]}

    def register_package(self, *, name: str, bundle_path: str, description: str,
                         input_schema: Mapping[str,Any], output_schema: Mapping[str,Any],
                         source_urls: list[str], trained_on_current_task: bool,
                         provenance: Mapping[str,Any], package_spec: Mapping[str,Any],
                         manual: Mapping[str,Any]|None=None):
        """Freeze an Agent-acquired model/planner/service bundle as a Tool.

        Packages use the same JSON contract as lightweight Tools but may carry
        repositories, checkpoints, and GPU dependencies.  They run in a
        separate no-network worker and never enter the Adapter process.
        """
        name,requested_name=_normalize_name(name)
        source=self._workspace_dir(bundle_path);spec=dict(package_spec or {})
        kind=str(spec.get("kind") or "")
        # Capability Packages are self-contained, per-invocation JSON workers.
        # A ROS bridge or an already-running robot service needs host IPC and a
        # safety boundary owned by the deployment Adapter; accepting that label
        # here would claim a deployment mode this sandbox cannot provide.
        allowed={"algorithm","model","perception","planner","policy","service"}
        if kind not in allowed:raise AssetError(f"package_spec.kind must be one of {sorted(allowed)}")
        entry=Path(str(spec.get("entrypoint") or ""))
        if (not str(entry) or entry.is_absolute() or ".." in entry.parts
                or not (source/entry).is_file()):
            raise AssetError("package entrypoint must be a bundle-relative file")
        accelerator=str(spec.get("accelerator") or "cpu")
        if accelerator not in {"cpu","cuda"}:raise AssetError("package accelerator must be cpu or cuda")
        if spec.get("network",False) is not False:
            raise AssetError("deployed capability packages must be network-isolated")
        timeout=float(spec.get("timeout_seconds",120))
        if not 0.1<=timeout<=600:raise AssetError("package timeout must be in [0.1, 600]")
        runtime_requirements=[str(item) for item in (spec.get("runtime_requirements") or [])]
        pin=re.compile(r"^[A-Za-z0-9_.-]+==[^\s=]+$")
        if any(not pin.fullmatch(item) for item in runtime_requirements):
            raise AssetError("package runtime requirements must be exact name==version pins")
        entry_text=(source/entry).read_text();self._validate_run_entrypoint(entry_text);imports=set()
        for node in ast.walk(ast.parse(entry_text)):
            if isinstance(node,ast.Import):imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split(".")[0])
        external=sorted(item for item in imports
                        if item not in sys.stdlib_module_names and item!="__future__")
        vendor=source/"vendor";lock=source/"requirements.lock"
        vendored_lock_sha=None
        if vendor.is_dir():
            if not lock.is_file():raise AssetError("vendored package requires requirements.lock")
            locked=[line.strip() for line in lock.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
            locked_pattern=re.compile(r"^[A-Za-z0-9_.-]+==[^\s]+(?:\s+--hash=sha256:[0-9a-fA-F]{64})+$")
            if not locked or any(not locked_pattern.fullmatch(line) for line in locked):
                raise AssetError("vendored package lock must pin every wheel with sha256")
            vendored_lock_sha=_file_sha256(lock)
        elif external and not runtime_requirements:
            raise AssetError(f"package external imports require pinned runtime_requirements: {external}")
        input_schema=_contract_schema(input_schema,"input_schema")
        output_schema=_contract_schema(output_schema,"output_schema")
        initial=self._manual_payload(description=description,input_schema=input_schema,
                                     output_schema=output_schema,manual=manual)
        provenance_record=_provenance_record(source_urls=source_urls,
            trained_on_current_task=trained_on_current_task,privileged_state_used=False,
            details=provenance)
        declared=dict(provenance_record.get("checkpoint_sha256") or {})
        if kind in {"model","perception","policy"} and not declared:
            raise AssetError("learned capability package requires checkpoint hashes")
        for relative,digest in declared.items():
            checkpoint=(source/str(relative)).resolve()
            if source not in checkpoint.parents or not checkpoint.is_file():
                raise AssetError(f"declared checkpoint is absent from bundle: {relative}")
            actual=_file_sha256(checkpoint)
            if actual.casefold()!=str(digest).casefold():
                raise AssetError(f"checkpoint sha256 mismatch: {relative}")
        compile(entry_text,str(source/entry),"exec")
        versions=[int(p.name[1:]) for p in (self.root/name).glob("v[0-9]*")
                  if p.name[1:].isdigit()]
        version=max(versions,default=0)+1;path=self.root/name/f"v{version:03d}"
        staging=self.root/name/f".v{version:03d}.staging-{time.time_ns()}";staging.mkdir(parents=True)
        try:
            shutil.copytree(source,staging/"bundle")
            bundle_hash=self._tree_sha256(staging/"bundle")
            entry_hash=hashlib.sha256((staging/"bundle"/entry).read_bytes()).hexdigest()
            runtime_spec={"protocol":"isolated-json-worker-v2","entrypoint":str(entry),
                          "accelerator":accelerator,"network":False,
                          "timeout_seconds":timeout,"lifecycle":"per-invocation",
                          "transport":"json-stdio",
                          "external_imports":external,
                          "runtime_requirements":runtime_requirements,
                          "vendored_lock_sha256":vendored_lock_sha}
            existing_manifests=[]
            for manifest_path in (self.root/name).glob("v*/manifest.json"):
                existing=json.loads(manifest_path.read_text());existing_manifests.append(existing)
                if (existing.get("bundle_tree_sha256")==bundle_hash
                        and existing.get("input_schema")==dict(input_schema)
                        and existing.get("output_schema")==dict(output_schema)
                        and existing.get("runtime_spec")==runtime_spec
                        and existing.get("provenance")==provenance_record
                        and self._accessible(existing)):
                    shutil.rmtree(staging,ignore_errors=True)
                    return {"tool_id":existing["tool_id"],"status":existing["status"],
                            "asset_kind":kind,"runtime_spec":runtime_spec,
                            "duplicate_of":existing["tool_id"]}
            manifest={"protocol":"embodied-codex-capability-package-v1",
                "tool_id":f"{name}:v{version:03d}","name":name,"version":version,
                "requested_name":requested_name,"asset_kind":kind,"description":description,
                "input_schema":dict(input_schema),"output_schema":dict(output_schema),
                "source_urls":list(source_urls),"trained_on_current_task":False,
                "privileged_state_used":False,"source_sha256":entry_hash,
                "bundle_tree_sha256":bundle_hash,"runtime_spec":runtime_spec,
                "provenance":provenance_record,"status":"registered","tests":[],
                "visibility":"task_local","task_scope_id":self.scope_id,
                "created_unix":time.time()}
            if existing_manifests:
                manifest["supersedes"]=max(existing_manifests,
                    key=lambda item:int(item.get("version",0)))["tool_id"]
            (staging/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            staging.replace(path)
        except Exception:
            shutil.rmtree(staging,ignore_errors=True);raise
        self._publish_manual(manifest["tool_id"],initial,authored_by="package_author")
        return {"tool_id":manifest["tool_id"],"status":"registered",
                "asset_kind":kind,"runtime_spec":runtime_spec}

    def register_deployment_tool(self, *, name: str, implementation_path: str,
                                 description: str, input_schema: Mapping[str,Any],
                                 output_schema: Mapping[str,Any], provenance: Mapping[str,Any],
                                 manual: Mapping[str,Any]|None=None,
                                 dependency_paths: Mapping[str,str|Path]|None=None):
        name,requested_name=_normalize_name(name)
        input_schema=_contract_schema(input_schema,"input_schema")
        output_schema=_contract_schema(output_schema,"output_schema")
        provenance_record=_provenance_record(
            source_urls=provenance.get("source_urls") or [],
            trained_on_current_task=provenance.get("trained_on_current_task"),
            privileged_state_used=provenance.get("privileged_state_used"),details=provenance)
        hashes=dict(provenance_record.get("checkpoint_sha256") or {})
        files=dict(provenance_record.get("checkpoint_files") or {})
        if hashes:
            if set(files)!=set(hashes):
                raise AssetError("deployment model provenance must map every checkpoint to a local file")
            for label,digest in hashes.items():
                checkpoint=Path(str(files[label])).resolve()
                if (not checkpoint.is_file() or
                        _file_sha256(checkpoint).casefold()!=str(digest).casefold()):
                    raise AssetError(f"deployment checkpoint sha256 mismatch: {label}")
        source=Path(implementation_path).resolve()
        if not source.is_file():raise AssetError(f"deployment Tool source is not a file: {source}")
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        relative_imports=set()
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                relative_imports.add(node.module)
        declared=dict(dependency_paths or {})
        if set(declared)!=relative_imports:
            raise AssetError("deployment Tool dependency_paths must exactly match "
                             f"single-dot relative imports: expected {sorted(relative_imports)}, "
                             f"got {sorted(declared)}")
        relative_modules={}
        dependency_sources={}
        for module,value in sorted(declared.items()):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",str(module)):
                raise AssetError(f"invalid deployment Tool dependency module: {module}")
            dependency=Path(value).resolve()
            if not dependency.is_file():
                raise AssetError(f"deployment Tool dependency is not a file: {dependency}")
            filename=f"{module}.py"
            relative_modules[str(module)]={"path":filename,
                "sha256":hashlib.sha256(dependency.read_bytes()).hexdigest()}
            dependency_sources[str(module)]=dependency
        for manifest_path in (self.root/name).glob("v*/manifest.json"):
            manifest=json.loads(manifest_path.read_text())
            if (manifest.get("source_sha256")==digest
                    and manifest.get("input_schema")==dict(input_schema)
                    and manifest.get("output_schema")==dict(output_schema)
                    and (manifest.get("relative_modules") or {})==relative_modules
                    and manifest.get("provenance")==provenance_record
                    and (manifest.get("provenance") or {}).get("audit_status")=="complete"):
                self.manual(manifest["tool_id"])
                return {"tool_id":manifest["tool_id"],"status":manifest["status"]}
        prior_manifests=[json.loads(p.read_text()) for p in (self.root/name).glob("v*/manifest.json")]
        versions=[int(item.get("version",0)) for item in prior_manifests]
        version=max(versions,default=0)+1;path=self.root/name/f"v{version:03d}";path.mkdir(parents=True)
        shutil.copy2(source,path/"tool.py")
        for module,dependency in dependency_sources.items():
            shutil.copy2(dependency,path/relative_modules[module]["path"])
        manifest={"protocol":"embodied-codex-deployment-tool-v1",
                  "tool_id":f"{name}:v{version:03d}","name":name,"version":version,
                  "requested_name":requested_name,
                  "description":description,"input_schema":dict(input_schema),
                  "output_schema":dict(output_schema),"source_urls":list(provenance.get("source_urls") or []),
                  "trained_on_current_task":False,"privileged_state_used":False,
                  "source_sha256":digest,"status":"tested","tests":[{"deployment_validation":True}],
                  "relative_modules":relative_modules,
                  "execution_owned_by_deployment":True,"provenance":provenance_record,
                  "visibility":"shared",
                  "created_unix":time.time()}
        if prior_manifests:
            manifest["supersedes"]=max(prior_manifests,
                key=lambda item:int(item.get("version",0)))["tool_id"]
        (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        initial=self._manual_payload(description=description,input_schema=input_schema,
                                     output_schema=output_schema,manual=manual)
        self._publish_manual(manifest["tool_id"],initial,authored_by="adapter_author")
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
        self._require_accessible(manifest)
        runtime_spec=dict(manifest.get("runtime_spec") or {})
        source=(path/"bundle"/str(runtime_spec["entrypoint"]) if runtime_spec else path/"tool.py")
        if (not source.is_file() or
                hashlib.sha256(source.read_bytes()).hexdigest()!=manifest["source_sha256"]):
            raise AssetError("Tool hash mismatch")
        if runtime_spec and self._tree_sha256(path/"bundle")!=manifest.get("bundle_tree_sha256"):
            raise AssetError("capability package bundle hash mismatch")
        for module,record in (manifest.get("relative_modules") or {}).items():
            if (not isinstance(record,Mapping)
                    or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",str(module))):
                raise AssetError("invalid deployment Tool relative module manifest")
            relative=Path(str(record.get("path") or ""))
            dependency=(path/relative).resolve()
            if (not str(relative) or relative.is_absolute() or ".." in relative.parts
                    or dependency.parent!=path or not dependency.is_file()
                    or hashlib.sha256(dependency.read_bytes()).hexdigest()!=record.get("sha256")):
                raise AssetError("deployment Tool dependency hash mismatch")
        dependencies=manifest.get("dependencies") or {"mode":"stdlib"}
        if dependencies.get("mode")=="vendored":
            lock=path/"requirements.lock";vendor=path/"vendor"
            if (not lock.is_file() or not vendor.is_dir()
                    or hashlib.sha256(lock.read_bytes()).hexdigest()!=dependencies.get("requirements_lock_sha256")
                    or self._tree_sha256(vendor)!=dependencies.get("vendor_tree_sha256")):
                raise AssetError("Tool dependency bundle hash mismatch")
        return {"manifest": manifest, "source": source.read_text()}

    def run(self, tool_id: str, payload: Mapping[str, Any]):
        path=self._path(tool_id);manifest=self.inspect(tool_id)["manifest"]
        _validate_instance(dict(payload),manifest.get("input_schema") or {},"input")
        result=self.runtime.execute(path,dict(payload))
        _validate_instance(result,manifest.get("output_schema") or {},"output")
        return result

    def test_tool(self, tool_id: str, cases: list[Mapping[str, Any]]):
        if not cases: raise AssetError("Tool tests are required")
        path = self._path(tool_id); manifest_path = path/"manifest.json"
        manifest = json.loads(manifest_path.read_text()); results = []
        try:
            for index,case in enumerate(cases):
                _validate_instance(case.get("input") or {},manifest.get("input_schema") or {},
                                   "test input")
                actual = self.run(tool_id, case.get("input") or {})
                expected = case.get("expected")
                _validate_instance(expected,manifest.get("output_schema") or {},"expected output")
                results.append({"passed": _test_values_match(actual,expected), "actual": actual,
                                "expected": expected})
        except Exception as exc:
            # A crashing or contract-invalid test is still immutable evidence.
            # Persist it before returning the exception so an unusable version
            # cannot remain ambiguously `registered` in a growing catalog.
            results.append({"passed":False,"case_index":index,
                            "error":f"{type(exc).__name__}: {exc}"})
            manifest["tests"].append(results);manifest["status"]="test_failed"
            manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
            raise
        manifest["tests"].append(results)
        # Retesting a convenient subset may not erase a genuine prior failure.
        # Re-evaluate stored actual/expected pairs using the current comparator
        # so benign floating-point representation differences remain valid.
        historical=[row for batch in manifest["tests"] if isinstance(batch,list)
                    for row in batch if isinstance(row,Mapping)]
        passed=bool(historical) and all(row.get("passed") is not False and
            _test_values_match(row.get("actual"),row.get("expected")) for row in historical)
        manifest["status"] = "tested" if passed else "test_failed"
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        return {"tool_id": tool_id, "status": manifest["status"], "results": results}

    def promote_for_reuse(self, tool_id: str, *, evidence_paths: list[str|Path],
                          controller_sha256: str,
                          required_case_handles: list[str]|None=None):
        """Promote one task-local Tool only after causal development evidence.

        Unit tests establish the JSON contract.  This gate separately proves
        that the exact accepted Controller called the Tool and succeeded on
        every required development case without consuming evaluator labels.
        """
        path=self._path(tool_id);manifest_path=path/"manifest.json"
        manifest=json.loads(manifest_path.read_text());self._require_accessible(manifest)
        if manifest.get("execution_owned_by_deployment"):
            return {"tool_id":tool_id,"visibility":"shared","already_shared":True}
        if manifest.get("visibility","shared")=="shared":
            return {"tool_id":tool_id,"visibility":"shared","already_shared":True}
        if manifest.get("status")!="tested":
            raise AssetError("only a contract-tested Tool can be promoted")
        expected_sha=str(controller_sha256 or "")
        if not re.fullmatch(r"[0-9a-f]{64}",expected_sha):
            raise AssetError("promotion requires the accepted Controller sha256")
        required=set(str(item) for item in (required_case_handles or []) if str(item))
        covered=set();accepted=[]
        for raw in evidence_paths:
            source=Path(raw).resolve()
            if not source.is_file():continue
            try:document=json.loads(source.read_text())
            except (OSError,json.JSONDecodeError):continue
            execution=document.get("execution") or {}
            events=execution.get("rpc_events") or []
            used=any(event.get("method")=="use" and
                event.get("arguments",{}).get("tool_id")==tool_id for event in events)
            if (document.get("sensor_success_candidate") is not True or not used
                    or execution.get("program_sha256")!=expected_sha):
                continue
            report=document.get("sensor_report") or {}
            case=str(report.get("_harness_case_id") or "")
            if case:covered.add(case)
            accepted.append(source)
        if not accepted:
            raise AssetError("Tool promotion lacks a successful task execution that called it")
        if required and not required.issubset(covered):
            raise AssetError("Tool promotion lacks required cross-case coverage")
        evidence_dir=path/"task_validation";evidence_dir.mkdir(exist_ok=True)
        records=[]
        for index,source in enumerate(sorted(set(accepted)),1):
            destination=evidence_dir/f"{index:03d}_{source.name}"
            shutil.copy2(source,destination)
            records.append({"path":str(destination.relative_to(path)),
                "sha256":_file_sha256(destination),"original_path":str(source)})
        manifest["visibility"]="shared";manifest["promoted_from_scope_id"]=self.scope_id
        manifest.pop("task_scope_id",None)
        manifest["task_validation"]={
            "protocol":"embodied-codex-tool-task-validation-v1",
            "controller_sha256":expected_sha,
            "required_case_count":len(required),"covered_case_count":len(covered),
            "evidence":records,"promoted_unix":time.time()}
        temporary=manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest,indent=2)+"\n");temporary.replace(manifest_path)
        return {"tool_id":tool_id,"visibility":"shared",
                "evidence_count":len(records),"covered_case_count":len(covered)}

    def tested(self):
        manifests = [json.loads(p.read_text()) for p in self.root.glob("*/v*/manifest.json")]
        return sorted([m for m in manifests if m.get("status") == "tested"
                       and self._accessible(m)],
                      key=lambda m: m["tool_id"])

    def list_all(self):
        manifests=[json.loads(p.read_text()) for p in self.root.glob("*/v*/manifest.json")]
        return sorted([manifest for manifest in manifests if self._accessible(manifest)],
                      key=lambda manifest:manifest["tool_id"])

    def list_summaries(self):
        """Return the compositional contract without replaying test history.

        Full manifests can contain large numeric test fixtures and provenance
        blobs.  Sending all of that through the coding-agent `list_tools` call
        duplicates the contracts already present in its instruction and can
        crowd out the controller-engineering context.  `inspect_tool` remains
        the explicit path for source, provenance, and individual test details.
        """
        fields=("protocol","tool_id","name","version","supersedes","asset_kind","description",
                "input_schema","output_schema","status",
                "trained_on_current_task","privileged_state_used",
                "execution_owned_by_deployment","runtime_spec")
        rows=[]
        for manifest in self.list_all():
            row={key:manifest[key] for key in fields if key in manifest}
            row["manual"]=self.manual(manifest["tool_id"])["manual"]
            rows.append(row)
        return rows

    def search(self, query: str, limit: int=8):
        fields=("protocol","tool_id","name","version","supersedes","asset_kind","description","input_schema",
                "output_schema","status","trained_on_current_task","privileged_state_used",
                "execution_owned_by_deployment","runtime_spec")
        grouped={}
        for item in self.list_all():grouped.setdefault(item.get("name"),[]).append(item)
        current=[]
        for versions in grouped.values():
            latest=max(versions,key=lambda item:int(item.get("version",0)))
            tested=[item for item in versions if item.get("status")=="tested"]
            latest_tested=(max(tested,key=lambda item:int(item.get("version",0)))
                           if tested else None)
            if latest_tested is not None:current.append(latest_tested)
            if latest_tested is None or latest["tool_id"]!=latest_tested["tool_id"]:
                current.append(latest)
        rows=[{key:item[key] for key in fields if key in item} for item in current]
        return rank_records(query,rows,text_fields=("name","description","input_schema",
            "output_schema"),id_field="tool_id",limit=limit)

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
        bundle_hashes=dict(skill_manifest.get("tool_bundle_sha256") or {})
        imported=[]; deployment=[]
        for tool_id in skill_manifest.get("tool_ids") or []:
            source = skill/"tools"/str(tool_id).replace(":","_")
            manifest = json.loads((source/"manifest.json").read_text())
            if (tool_id not in bundle_hashes or
                    self._tree_sha256(source)!=bundle_hashes[tool_id]):
                raise AssetError(f"frozen Tool bundle hash mismatch: {tool_id}")
            if manifest.get("tool_id") != tool_id or manifest.get("status") != "tested":
                raise AssetError(f"invalid frozen Tool: {tool_id}")
            if (manifest.get("trained_on_current_task") is not False
                    or manifest.get("privileged_state_used") is not False):
                raise AssetError(f"forbidden frozen Tool provenance: {tool_id}")
            runtime_spec=dict(manifest.get("runtime_spec") or {})
            implementation=(source/"bundle"/str(runtime_spec["entrypoint"])
                            if runtime_spec else source/"tool.py")
            if (not implementation.is_file() or
                    hashlib.sha256(implementation.read_bytes()).hexdigest()!=manifest.get("source_sha256")):
                raise AssetError(f"frozen Tool hash mismatch: {tool_id}")
            if runtime_spec and self._tree_sha256(source/"bundle")!=manifest.get("bundle_tree_sha256"):
                raise AssetError(f"frozen capability bundle hash mismatch: {tool_id}")
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
               task_model: Mapping[str,Any]|None=None,
               interface: Mapping[str,Any]|None=None,
               evidence_paths: list[str|Path]|None=None,
               migration: Mapping[str,Any]|None=None,
               controller_tool_bindings: Mapping[str,str]|None=None):
        name,requested_name=_normalize_name(name)
        controller=Path(controller).resolve()
        if not controller.is_file():raise AssetError("Skill controller is not a file")
        family=self.root/name
        prior=[json.loads(path.read_text()) for path in family.glob("v*/manifest.json")]
        def skill_version(item):
            encoded=str(item.get("skill_id") or ":v000").partition(":v")[2]
            return int(item.get("version") or (encoded if encoded.isdigit() else 0))
        versions=[skill_version(item) for item in prior]
        version=max(versions,default=0)+1;path=family/f"v{version:03d}"
        staging=family/f".v{version:03d}.staging-{time.time_ns()}";staging.mkdir(parents=True)
        shutil.copy2(controller,staging/"controller.py")
        tool_bundle_hashes={}
        try:
            for tool_id in sorted(set(tool_ids)):
                inspected=tools.inspect(tool_id);manifest=inspected["manifest"]
                if manifest.get("status")!="tested":
                    raise AssetError(f"Skill dependency is not tested: {tool_id}")
                provenance=dict(manifest.get("provenance") or {})
                if (manifest.get("trained_on_current_task") is not False
                        or manifest.get("privileged_state_used") is not False
                        or provenance.get("audit_status")!="complete"):
                    raise AssetError(f"Skill dependency provenance is incomplete: {tool_id}")
                source=tools._path(tool_id)
                destination=staging/"tools"/tool_id.replace(":","_")
                shutil.copytree(source,destination)
                manual=tools.manual(tool_id)
                (destination/"manual.json").write_text(json.dumps(manual,indent=2)+"\n")
                tool_bundle_hashes[tool_id]=tools._tree_sha256(destination)
            phases=list((task_model or {}).get("phases") or [])
            requirements=list((task_model or {}).get("requirements") or [])
            derived_interface={"preconditions":sorted({str(item) for phase in phases
                        for item in (phase.get("capability_requirements") or [])}),
                "effects":[str(item.get("description")) for item in requirements if item.get("description")],
                "required_sensors":sorted({str(item) for phase in phases
                        for item in (phase.get("observations") or [])}),
                "required_robot_operations":sorted({str(item) for phase in phases
                        for item in (phase.get("required_robot_operations") or [])}),
                "parameters":[dict(item) for item in ((task_model or {}).get("entities") or [])
                              if isinstance(item,Mapping)],
                "failure_modes":["A declared precondition, required sensor, Tool, or action may be unavailable."],
                "tool_dependencies":sorted(set(tool_ids)),
                "composition_notes":"Use the frozen Controller as the reference implementation and bind live parameters from observations."}
            if interface is not None:
                supplied=dict(interface)
                required={"preconditions":list,"effects":list,"required_sensors":list,
                          "required_robot_operations":list,"parameters":list,
                          "failure_modes":list,"composition_notes":str}
                for key,kind in required.items():
                    if not isinstance(supplied.get(key),kind):
                        raise AssetError(f"Skill interface field {key} is invalid")
                derived_interface.update({key:supplied[key] for key in required})
            derived_interface["tool_dependencies"]=sorted(set(tool_ids))
            frozen_evidence=[]
            evidence_dir=staging/"evidence"
            for index,value in enumerate(evidence_paths or [],1):
                source=Path(value).resolve()
                if not source.is_file():raise AssetError(f"Skill evidence is not a file: {source}")
                evidence_dir.mkdir(exist_ok=True)
                destination=evidence_dir/f"{index:04d}_{source.parent.name}_{source.name}"
                shutil.copy2(source,destination)
                frozen_evidence.append({"path":str(destination.relative_to(staging)),
                    "original_path":str(source),"sha256":hashlib.sha256(
                        destination.read_bytes()).hexdigest(),"bytes":destination.stat().st_size})
            bindings={str(key):str(value) for key,value in
                      (controller_tool_bindings or {tool_id:tool_id for tool_id in tool_ids}).items()}
            if set(bindings.values())-set(tool_ids):
                raise AssetError("Skill controller Tool bindings must resolve to frozen Tool ids")
            manifest={"protocol":"embodied-codex-skill-v1","skill_id":f"{name}:v{version:03d}",
                  "version":version,
                  "requested_name":requested_name,
                  "task":task,"controller_sha256":hashlib.sha256((staging/"controller.py").read_bytes()).hexdigest(),
                  "tool_ids":sorted(set(tool_ids)),"development_evidence":dict(evidence),
                  "controller_tool_bindings":dict(sorted(bindings.items())),
                  "tool_bundle_sha256":tool_bundle_hashes,
                  "evidence_files":frozen_evidence,
                  "interface":derived_interface,
                  "status":"sensor_success","created_unix":time.time()}
            if migration is not None:
                migration_record=dict(migration)
                required={"source_skill_id":str,"reason":str,"tool_replacements":dict,
                          "controller_sha256_unchanged":bool,
                          "development_reexecuted":bool}
                for key,kind in required.items():
                    if not isinstance(migration_record.get(key),kind):
                        raise AssetError(f"Skill migration field {key} is invalid")
                if (not migration_record["reason"].strip()
                        or migration_record["controller_sha256_unchanged"] is not True):
                    raise AssetError("Skill packaging migration must preserve the controller and state a reason")
                manifest["packaging_migration"]={
                    "protocol":"embodied-codex-skill-packaging-migration-v1",
                    **migration_record}
            if experience is not None:
                experience_path=staging/"experience.json"
                experience_path.write_text(json.dumps(dict(experience),indent=2,default=str)+"\n")
                manifest["experience_sha256"]=hashlib.sha256(experience_path.read_bytes()).hexdigest()
            if task_model is not None:
                task_model_path=staging/"task_model.json"
                task_model_path.write_text(json.dumps(dict(task_model),indent=2,default=str)+"\n")
                manifest["task_model_sha256"]=hashlib.sha256(task_model_path.read_bytes()).hexdigest()
            identity={key:manifest.get(key) for key in ("task","controller_sha256","tool_ids",
                "tool_bundle_sha256","interface","experience_sha256","task_model_sha256",
                "packaging_migration","controller_tool_bindings")}
            identity["evidence_sha256"]=[item["sha256"] for item in frozen_evidence]
            for existing in prior:
                existing_identity={key:existing.get(key) for key in identity}
                existing_identity["evidence_sha256"]=[item.get("sha256") for item in
                                                       existing.get("evidence_files") or []]
                if existing_identity==identity:
                    shutil.rmtree(staging,ignore_errors=True)
                    existing_path=family/f"v{skill_version(existing):03d}"
                    return {"skill_id":existing["skill_id"],"path":str(existing_path),
                            "duplicate_of":existing["skill_id"]}
            if prior:
                manifest["supersedes"]=max(prior,key=skill_version)["skill_id"]
            (staging/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            staging.replace(path)
        except Exception:
            shutil.rmtree(staging,ignore_errors=True);raise
        return {"skill_id":manifest["skill_id"],"path":str(path)}

    def repackage(self, *, skill_dir: str|Path, tools: CapabilityLibrary,
                  tool_replacements: Mapping[str,str], reason: str):
        """Publish an audited immutable packaging migration without a new rollout."""
        source=Path(skill_dir).resolve()
        manifest_path=source/"manifest.json"
        if not manifest_path.is_file():raise AssetError("source Skill manifest is missing")
        manifest=json.loads(manifest_path.read_text())
        controller=source/"controller.py"
        if (not controller.is_file() or
                hashlib.sha256(controller.read_bytes()).hexdigest()!=manifest.get("controller_sha256")):
            raise AssetError("source Skill controller hash mismatch")
        replacements={str(key):str(value) for key,value in tool_replacements.items()}
        unknown=set(replacements)-set(manifest.get("tool_ids") or [])
        if unknown:raise AssetError(f"Skill migration names unknown Tool ids: {sorted(unknown)}")
        tool_ids=[replacements.get(tool_id,tool_id) for tool_id in manifest.get("tool_ids") or []]
        prior_bindings=dict(manifest.get("controller_tool_bindings") or
            {tool_id:tool_id for tool_id in manifest.get("tool_ids") or []})
        bindings={logical:replacements.get(packaged,packaged)
                  for logical,packaged in prior_bindings.items()}
        experience=(json.loads((source/"experience.json").read_text())
                    if (source/"experience.json").is_file() else None)
        task_model=(json.loads((source/"task_model.json").read_text())
                    if (source/"task_model.json").is_file() else None)
        evidence_paths=[]
        for record in manifest.get("evidence_files") or []:
            evidence=source/str(record.get("path") or "")
            if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest()!=record.get("sha256"):
                raise AssetError("source Skill evidence hash mismatch")
            evidence_paths.append(evidence)
        name=str(manifest.get("skill_id") or "").partition(":")[0]
        if not name:raise AssetError("source Skill id is invalid")
        return self.freeze(name=name,task=str(manifest.get("task") or ""),
            controller=controller,evidence=dict(manifest.get("development_evidence") or {}),
            tool_ids=tool_ids,tools=tools,experience=experience,task_model=task_model,
            interface=dict(manifest.get("interface") or {}),evidence_paths=evidence_paths,
            controller_tool_bindings=bindings,
            migration={"source_skill_id":manifest["skill_id"],"reason":str(reason),
                "tool_replacements":replacements,"controller_sha256_unchanged":True,
                "development_reexecuted":False})

    def list_summaries(self):
        rows=[]
        for path in self.root.glob("*/v*/manifest.json"):
            item=json.loads(path.read_text())
            rows.append({key:item.get(key) for key in
                ("skill_id","task","status","interface","tool_ids","created_unix")})
        return sorted(rows,key=lambda row:str(row.get("skill_id")))

    def search(self,query: str,limit: int=5):
        return rank_records(query,self.list_summaries(),
            text_fields=("task","interface","tool_ids"),id_field="skill_id",limit=limit)

    def _path(self,skill_id: str):
        name,sep,version=str(skill_id).partition(":")
        if not sep:raise AssetError("invalid Skill id")
        path=(self.root/name/version).resolve()
        if self.root not in path.parents or not (path/"manifest.json").is_file():
            raise FileNotFoundError(skill_id)
        return path

    def inspect(self,skill_id: str):
        path=self._path(skill_id);manifest=json.loads((path/"manifest.json").read_text())
        if hashlib.sha256((path/"controller.py").read_bytes()).hexdigest()!=manifest.get("controller_sha256"):
            raise AssetError("Skill controller hash mismatch")
        for tool_id,digest in (manifest.get("tool_bundle_sha256") or {}).items():
            bundle=path/"tools"/str(tool_id).replace(":","_")
            if not bundle.is_dir() or CapabilityLibrary._tree_sha256(bundle)!=digest:
                raise AssetError(f"Skill Tool bundle hash mismatch: {tool_id}")
        for filename,key in (("experience.json","experience_sha256"),
                             ("task_model.json","task_model_sha256")):
            expected=manifest.get(key)
            source=path/filename
            if expected and (not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest()!=expected):
                raise AssetError(f"Skill {filename} hash mismatch")
        for evidence in manifest.get("evidence_files") or []:
            source=path/str(evidence.get("path") or "")
            if (not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()
                    !=evidence.get("sha256")):
                raise AssetError(f"Skill evidence hash mismatch: {skill_id}")
        return manifest


class ExperienceLibrary:
    """Immutable, evidence-backed lessons shared across task workspaces."""
    def __init__(self, root: str|Path):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)

    def _path(self,experience_id: str):
        name,sep,version=str(experience_id).partition(":")
        if not sep:raise AssetError("invalid Experience id")
        path=(self.root/name/version).resolve()
        if self.root not in path.parents or not (path/"manifest.json").is_file():
            raise FileNotFoundError(experience_id)
        return path

    def register(self, *, name: str, summary: str, applicability: str,
                 keywords: list[str], evidence_paths: list[str|Path],
                 evidence_assessment: Mapping[str,Any]|None=None):
        name,requested_name=_normalize_name(name)
        if not str(summary).strip():raise AssetError("Experience summary is required")
        if not str(applicability).strip():raise AssetError("Experience applicability is required")
        if not evidence_paths:raise AssetError("Experience evidence is required")
        sources=[]
        for value in evidence_paths:
            source=Path(value).resolve()
            if not source.is_file():raise AssetError(f"Experience evidence is not a file: {source}")
            sources.append(source)
        family=self.root/name
        versions=[int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        version=max(versions,default=0)+1;path=family/f"v{version:03d}"
        path.mkdir(parents=True)
        evidence=[];evidence_dir=path/"evidence";evidence_dir.mkdir()
        for index,source in enumerate(sources,1):
            destination=evidence_dir/f"{index:03d}_{source.name}"
            shutil.copy2(source,destination)
            evidence.append({"path":str(destination.relative_to(path)),"original_path":str(source),
                "sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
                "bytes":destination.stat().st_size})
        assessment=dict(evidence_assessment or {})
        outcome=str(assessment.get("outcome") or "unknown")
        status={"success":"success_evidence","failure":"failure_evidence",
                "mixed":"mixed_evidence"}.get(outcome,"evidence_backed")
        manifest={"protocol":"embodied-codex-experience-v1",
                  "experience_id":f"{name}:v{version:03d}","name":name,
                  "requested_name":requested_name,
                  "version":version,"summary":str(summary).strip(),
                  "applicability":str(applicability).strip(),
                  "keywords":sorted(set(str(x).strip() for x in keywords if str(x).strip())),
                  "evidence":evidence,"evidence_assessment":assessment,
                  "status":status,"created_unix":time.time()}
        if versions:
            manifest["supersedes"]=f"{name}:v{max(versions):03d}"
        (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        return {"experience_id":manifest["experience_id"],"status":manifest["status"]}

    def list_summaries(self):
        latest={}
        for path in self.root.glob("*/v*/manifest.json"):
            item=self.inspect(json.loads(path.read_text())["experience_id"])
            name=str(item["name"])
            if name not in latest or int(item.get("version",0))>int(latest[name].get("version",0)):
                latest[name]=item
        rows=[]
        for item in latest.values():
            rows.append({key:item[key] for key in
                         ("experience_id","name","summary","applicability","keywords","status")})
        return sorted(rows,key=lambda row:row["experience_id"])

    def search(self,query: str,limit: int=8):
        return rank_records(query,self.list_summaries(),
            text_fields=("name","summary","applicability","keywords"),
            id_field="experience_id",limit=limit)

    def inspect(self, experience_id: str):
        path=self._path(experience_id)/"manifest.json";item=json.loads(path.read_text())
        evidence_paths=[]
        for evidence in item.get("evidence") or []:
            source=Path(evidence["path"])
            if not source.is_absolute():source=path.parent/source
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=evidence["sha256"]:
                raise AssetError(f"Experience evidence hash mismatch: {experience_id}")
            evidence_paths.append(source)
        assessment=execution_evidence_assessment(evidence_paths)
        if assessment["records"]:
            original_assessment=item.get("evidence_assessment") or {}
            original_status=item.get("status")
            outcome=assessment["outcome"]
            item["evidence_assessment"]=assessment
            item["status"]={"success":"success_evidence","failure":"failure_evidence",
                "mixed":"mixed_evidence"}.get(outcome,"evidence_backed")
            if (original_assessment!=assessment or original_status!=item["status"]):
                item["manifest_claim"]={"evidence_assessment":original_assessment,
                                        "status":original_status}
                item["manifest_claim_conflicts_with_evidence"]=(
                    original_status!=item["status"] and outcome!="unknown")
        return item


class CapabilityGapLibrary:
    """Immutable revisions of an evidence-backed capability-acquisition case.

    A Gap is deliberately separate from free-form agent reasoning: it makes the
    causal chain from observed failure through acquisition and validation an
    inspectable research artifact.  Updating a Gap publishes a new immutable
    revision and keeps the prior revision intact.
    """
    STATUSES={"observed","diagnosed","searching","integrating","validated",
              "rejected","unresolved"}
    TRANSITIONS={
        # Evidence may falsify a diagnosis or integrated candidate. Structure
        # enforces what each state must contain; it must not prevent the model
        # from revising a scientific hypothesis.
        "observed":{"observed","diagnosed","searching","integrating","unresolved"},
        "diagnosed":{"diagnosed","searching","integrating","rejected","unresolved"},
        "searching":{"diagnosed","searching","integrating","rejected","unresolved"},
        "integrating":{"diagnosed","searching","integrating","validated","rejected","unresolved"},
        "validated":{"validated","diagnosed"},
        "rejected":{"diagnosed","searching","integrating","rejected","unresolved"},
        "unresolved":{"diagnosed","searching","integrating","unresolved"}}

    def __init__(self,root: str|Path):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)

    def _path(self,gap_id: str):
        name,sep,version=str(gap_id).partition(":")
        if not sep:raise AssetError("invalid Capability Gap id")
        path=(self.root/name/version/"manifest.json").resolve()
        if self.root not in path.parents or not path.is_file():raise FileNotFoundError(gap_id)
        return path

    def publish(self, *, name: str, task: str, failure_summary: str,
                hypotheses: list[str], selected_diagnosis: str,
                required_capability: Mapping[str,Any],
                searched_candidates: list[Mapping[str,Any]],
                provenance_decision: Mapping[str,Any],
                integration_result: Mapping[str,Any],
                task_validation: Mapping[str,Any],
                reuse_evidence: Mapping[str,Any], status: str,
                evidence_paths: list[str|Path], previous_gap_id: str|None=None):
        name,requested_name=_normalize_name(name)
        status=str(status)
        if status not in self.STATUSES:raise AssetError(f"invalid Capability Gap status: {status}")
        if not str(task).strip() or not str(failure_summary).strip():
            raise AssetError("Capability Gap task and failure summary are required")
        hypotheses=[str(item).strip() for item in hypotheses if str(item).strip()]
        diagnosis=str(selected_diagnosis).strip();required_capability=dict(required_capability)
        searched_candidates=[dict(item) for item in searched_candidates]
        provenance_decision=dict(provenance_decision);integration_result=dict(integration_result)
        task_validation=dict(task_validation);reuse_evidence=dict(reuse_evidence)
        if status!="observed" and (not diagnosis or not required_capability):
            raise AssetError(f"Capability Gap status {status} requires diagnosis and capability contract")
        if status in {"searching","integrating","validated","rejected"} and not searched_candidates:
            raise AssetError(f"Capability Gap status {status} requires recorded search candidates")
        if status in {"integrating","validated"} and (not provenance_decision or not integration_result):
            raise AssetError(f"Capability Gap status {status} requires provenance and integration records")
        if status=="validated" and not task_validation:
            raise AssetError("validated Capability Gap requires task-level validation")
        if status=="rejected" and not (provenance_decision or integration_result):
            raise AssetError("rejected Capability Gap requires a rejection decision")
        previous=None
        if previous_gap_id:
            previous=self.inspect(previous_gap_id)
            if previous.get("name")!=name:
                raise AssetError("Capability Gap revision must preserve its normalized name")
            if status not in self.TRANSITIONS.get(previous.get("status"),set()):
                raise AssetError(f"invalid Capability Gap transition: {previous.get('status')} -> {status}")
        sources=[]
        for value in evidence_paths:
            source=Path(value).resolve()
            if not source.is_file():raise AssetError(f"Capability Gap evidence is not a file: {source}")
            sources.append(source)
        if not sources:raise AssetError("Capability Gap revision requires evidence")
        family=self.root/name
        versions=[int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()]
        if versions:
            expected=f"{name}:v{max(versions):03d}"
            if previous_gap_id!=expected:
                raise AssetError(f"Capability Gap revision must extend latest revision {expected}")
        version=max(versions,default=0)+1;path=family/f"v{version:03d}"
        path.mkdir(parents=True);evidence_dir=path/"evidence";evidence_dir.mkdir()
        evidence=[]
        for index,source in enumerate(sources,1):
            destination=evidence_dir/f"{index:03d}_{source.name}";shutil.copy2(source,destination)
            evidence.append({"path":str(destination.relative_to(path)),
                "original_path":str(source),
                "sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
                "bytes":destination.stat().st_size})
        gap_id=f"{name}:v{version:03d}"
        manifest={"protocol":"embodied-codex-capability-gap-v1","gap_id":gap_id,
            "name":name,"requested_name":requested_name,"version":version,
            "previous_gap_id":previous_gap_id,"task":str(task).strip(),
            "failure_summary":str(failure_summary).strip(),
            "hypotheses":hypotheses,"selected_diagnosis":diagnosis,
            "required_capability":required_capability,
            "searched_candidates":searched_candidates,
            "provenance_decision":provenance_decision,
            "integration_result":integration_result,
            "task_validation":task_validation,"reuse_evidence":reuse_evidence,
            "status":status,"evidence":evidence,"created_unix":time.time()}
        if previous is not None:
            manifest["lineage_root_gap_id"]=previous.get("lineage_root_gap_id") or previous["gap_id"]
        else:manifest["lineage_root_gap_id"]=gap_id
        (path/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        return {"gap_id":gap_id,"status":status,
                "previous_gap_id":previous_gap_id}

    def list_summaries(self):
        rows=[]
        for path in self.root.glob("*/v*/manifest.json"):
            item=self.inspect(json.loads(path.read_text())["gap_id"])
            rows.append({key:item.get(key) for key in ("gap_id","name","task",
                "failure_summary","hypotheses","selected_diagnosis",
                "required_capability","status","previous_gap_id",
                "authoritative_outcome","model_claim_conflicts_with_evidence")})
        return sorted(rows,key=lambda row:str(row.get("gap_id")))

    def latest_for_name(self,name: str):
        normalized,_requested=_normalize_name(name)
        candidates=[item for item in self.list_summaries()
                    if item.get("name")==normalized]
        if not candidates:return None
        latest=max(candidates,key=lambda item:int(
            str(item.get("gap_id") or ":v000").partition(":v")[2] or 0))
        return self.inspect(latest["gap_id"])

    def search(self,query: str,limit: int=8):
        latest={}
        for item in self.list_summaries():
            name=str(item.get("name") or "")
            version=int(str(item.get("gap_id") or ":v000").partition(":v")[2] or 0)
            if name not in latest or version>latest[name][0]:latest[name]=(version,item)
        return rank_records(query,[item for _version,item in latest.values()],text_fields=("name","task",
            "failure_summary","hypotheses","selected_diagnosis","required_capability"),
            id_field="gap_id",limit=limit)

    def inspect(self,gap_id: str):
        path=self._path(gap_id);item=json.loads(path.read_text())
        evidence_paths=[]
        for evidence in item.get("evidence") or []:
            source=path.parent/evidence["path"]
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=evidence["sha256"]:
                raise AssetError(f"Capability Gap evidence hash mismatch: {gap_id}")
            evidence_paths.append(source)
        assessment=execution_evidence_assessment(evidence_paths)
        if assessment["records"]:
            original_validation=item.get("task_validation") or {}
            item["task_validation"]=bind_authoritative_validation(original_validation,assessment)
            item["authoritative_outcome"]=assessment["outcome"]
            item["model_claim_conflicts_with_evidence"]=bool(
                item["task_validation"].get("model_claim_conflicts_with_evidence"))
        return item

__all__ = ["CapabilityLibrary", "SkillLibrary", "ExperienceLibrary",
           "CapabilityGapLibrary", "AssetError", "execution_evidence_assessment",
           "bind_authoritative_validation"]
