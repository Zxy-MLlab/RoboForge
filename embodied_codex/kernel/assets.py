"""Environment-neutral immutable asset stores used by the canonical kernel.

This module deliberately contains no benchmark, contamination, or task-policy
rules.  It stores versioned Tool/Skill/Experience/Gap records and delegates
untrusted Tool execution to the isolated ToolRuntime.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Mapping

from jsonschema import Draft202012Validator, ValidationError

from ..retrieval import rank_records
from ..tool_runtime import ToolRuntime
from .cas import ContentAddressedStore, ContentAddressedStoreError


class AssetError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


@contextmanager
def _registry_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".registry.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")[:63]
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,62}", slug):
        raise AssetError("asset name must be a stable identifier")
    return slug


def _schema(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    result = dict(value)
    try:
        Draft202012Validator.check_schema(result)
    except Exception as exc:
        raise AssetError(f"invalid {label}: {exc}") from exc
    return result


def _validate(value: Any, schema: Mapping[str, Any], label: str) -> None:
    try:
        Draft202012Validator(dict(schema)).validate(value)
    except ValidationError as exc:
        raise AssetError(f"{label} violates JSON Schema: {exc.message}") from exc


def _same(actual: Any, expected: Any) -> bool:
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float)) and not isinstance(expected, bool)):
        return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return set(actual) == set(expected) and all(_same(actual[key], expected[key]) for key in actual)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_same(a, b) for a, b in zip(actual, expected))
    return actual == expected


def _manual(description: str, input_schema: Mapping[str, Any], output_schema: Mapping[str, Any], value=None):
    result = dict(value or {})
    result.setdefault("purpose", str(description))
    result.setdefault("when_to_use", [str(description)])
    result.setdefault("inputs", dict(input_schema.get("properties") or {}))
    result.setdefault("outputs", dict(output_schema.get("properties") or {}))
    result.setdefault("examples", [])
    result.setdefault("failure_modes", ["Malformed input or unavailable runtime dependency."])
    result.setdefault("limitations", [])
    required = {"purpose": str, "when_to_use": list, "inputs": dict, "outputs": dict,
                "examples": list, "failure_modes": list, "limitations": list}
    for key, kind in required.items():
        if not isinstance(result.get(key), kind):
            raise AssetError(f"Tool manual field {key} is invalid")
    if set(result["inputs"]) != set(input_schema.get("properties") or {}):
        raise AssetError("Tool manual inputs must match the schema")
    if set(result["outputs"]) != set(output_schema.get("properties") or {}):
        raise AssetError("Tool manual outputs must match the schema")
    return {key: result[key] for key in required}


class CapabilityLibrary:
    """Versioned Tool store with schema tests and isolated runtime binding."""

    def __init__(self, root: str | Path, workspace_root: str | Path | None = None,
                 *, python: str | Path | None = None, scope_id: str | None = None,
                 allowed_input_roots: list[str | Path] | None = None,
                 sandbox: Any = None, require_runtime: bool = True):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(workspace_root or self.root).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.scope_id = str(scope_id or "shared")
        self.cas = ContentAddressedStore(self.root / "_cas")
        self.runtime = (ToolRuntime(python=python, allowed_input_roots=allowed_input_roots,
                                    sandbox=sandbox) if require_runtime else None)

    def _workspace_file(self, path: str) -> Path:
        candidate = (self.workspace / str(path)).resolve()
        if self.workspace not in candidate.parents or not candidate.is_file():
            raise AssetError("Tool source must be a workspace file")
        return candidate

    def _path(self, tool_id: str) -> Path:
        name, separator, version = str(tool_id).partition(":")
        path = (self.root / name / version).resolve()
        if not separator or self.root not in path.parents or not (path / "manifest.json").is_file():
            raise FileNotFoundError(tool_id)
        return path

    @staticmethod
    def _entrypoint(source: str) -> None:
        tree = ast.parse(source)
        entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
        if len(entries) != 1:
            raise AssetError("Tool source must define exactly one synchronous top-level def run(payload)")
        args = entries[0].args
        positional = [*args.posonlyargs, *args.args]
        if (len(positional) != 1 or args.vararg or args.kwarg or args.kwonlyargs
                or args.defaults or args.kw_defaults):
            raise AssetError("Tool entrypoint must define exactly one def run(payload)")

    def _write_manifest(self, name: str, manifest: dict[str, Any], source: Path,
                        manual: Mapping[str, Any]) -> dict[str, Any]:
        with _registry_lock(self.root):
            family = self.root / name
            versions = [int(path.name[1:]) for path in family.glob("v[0-9]*") if path.name[1:].isdigit()]
            version = max(versions, default=0) + 1
            target = family / f"v{version:03d}"
            staging = family / f".v{version:03d}.staging-{uuid.uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                shutil.copy2(source, staging / "tool.py")
                manifest["version"] = version
                manifest["tool_id"] = f"{name}:v{version:03d}"
                manifest["created_unix"] = time.time()
                (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                manual_dir = self.root / "_manuals" / name / f"v{version:03d}"
                manual_dir.mkdir(parents=True, exist_ok=True)
                (manual_dir / "r001.json").write_text(json.dumps({"tool_id": manifest["tool_id"],
                    "manual_revision": 1, "manual": dict(manual), "created_unix": time.time()}, indent=2) + "\n")
                staging.replace(target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return {"tool_id": manifest["tool_id"], "status": manifest["status"]}

    def register_tool(self, *, name: str, source_path: str, description: str,
                      input_schema: Mapping[str, Any], output_schema: Mapping[str, Any],
                      source_urls: list[str] | None = None,
                      manual: Mapping[str, Any] | None = None,
                      runtime_requirements: list[str] | None = None,
                      provenance: Mapping[str, Any] | None = None,
                      **_unused) -> dict[str, Any]:
        source = self._workspace_file(source_path)
        text = source.read_text()
        compile(text, str(source), "exec")
        self._entrypoint(text)
        input_schema = _schema(input_schema, "input_schema")
        output_schema = _schema(output_schema, "output_schema")
        requirements = [str(item) for item in runtime_requirements or []]
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s=]+", item)
               for item in requirements):
            raise AssetError("runtime requirements must be exact name==version pins")
        name = _name(name)
        digest = _sha256(source)
        for old in self.list_all():
            if (old.get("name") == name and old.get("source_sha256") == digest
                    and old.get("input_schema") == input_schema
                    and old.get("output_schema") == output_schema
                    and (old.get("dependencies") or {}).get("runtime_requirements", [])
                        == requirements):
                return {"tool_id": old["tool_id"], "status": old["status"], "duplicate_of": old["tool_id"]}
        manifest = {"protocol": "roboforge-tool-v2", "name": name, "description": str(description),
                    "input_schema": input_schema, "output_schema": output_schema,
                    "source_sha256": digest, "source_urls": list(source_urls or []),
                    "dependencies": {"runtime": "isolated-python",
                                     "runtime_requirements": requirements},
                    "provenance": dict(provenance or {"origin": "workspace"}),
                    "visibility": "shared", "status": "candidate", "tests": []}
        return self._write_manifest(name, manifest, source,
            _manual(description, input_schema, output_schema, manual))

    def register_package(self, *, name: str, bundle_path: str, description: str,
                         input_schema: Mapping[str, Any], output_schema: Mapping[str, Any],
                         package_spec: Mapping[str, Any], source_urls: list[str] | None = None,
                         provenance: Mapping[str, Any] | None = None,
                         **_unused) -> dict[str, Any]:
        bundle = (self.workspace / str(bundle_path)).resolve()
        if self.workspace not in bundle.parents or not bundle.is_dir():
            raise AssetError("capability bundle must be a workspace directory")
        spec = dict(package_spec)
        kind = str(spec.get("kind") or "algorithm")
        if kind not in {"algorithm", "perception", "planner", "policy", "model"}:
            raise AssetError("unsupported capability package kind")
        entry = Path(str(spec.get("entrypoint") or ""))
        if not str(entry) or entry.is_absolute() or ".." in entry.parts or not (bundle / entry).is_file():
            raise AssetError("package entrypoint must be bundle-relative")
        text = (bundle / entry).read_text()
        compile(text, str(bundle / entry), "exec")
        self._entrypoint(text)
        checkpoint_hashes = {str(key): str(value).casefold() for key, value in
                             dict(spec.get("checkpoint_sha256") or {}).items()}
        if kind in {"perception", "policy", "model"} and not checkpoint_hashes:
            raise AssetError("checkpoint-backed package requires checkpoint_sha256")
        for relative, expected in checkpoint_hashes.items():
            checkpoint = (bundle / relative).resolve()
            if (bundle not in checkpoint.parents or not checkpoint.is_file()
                    or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise AssetError(f"checkpoint sha256 mismatch: {relative}")
        accelerator = str(spec.get("accelerator", "cpu"))
        if accelerator not in {"cpu", "cuda"}:
            raise AssetError("package accelerator must be cpu or cuda")
        requirements = [str(item) for item in spec.get("runtime_requirements") or []]
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s=]+", item) for item in requirements):
            raise AssetError("runtime requirements must be exact name==version pins")
        input_schema = _schema(input_schema, "input_schema")
        output_schema = _schema(output_schema, "output_schema")
        name = _name(name)
        records = []
        try:
            for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
                if path.is_symlink():
                    raise AssetError("capability package symlinks are not allowed")
                relative = path.relative_to(bundle).as_posix()
                stored = self.cas.put(path,
                    expected_sha256=checkpoint_hashes.get(relative))
                records.append({"path": relative, **stored,
                                "executable": bool(path.stat().st_mode & 0o111)})
        except ContentAddressedStoreError as exc:
            raise AssetError(str(exc)) from exc
        missing_checkpoints = set(checkpoint_hashes) - {row["path"] for row in records}
        if missing_checkpoints:
            raise AssetError(f"checkpoint files are missing: {sorted(missing_checkpoints)}")
        tree_digest = hashlib.sha256()
        for record in records:
            tree_digest.update(str(record["path"]).encode() + b"\0")
            tree_digest.update(bytes.fromhex(str(record["sha256"])))
        manifest = {"protocol": "roboforge-capability-package-v2", "name": name,
                    "description": str(description), "input_schema": input_schema,
                    "output_schema": output_schema, "source_sha256": _sha256(bundle / entry),
                    "bundle_tree_sha256": tree_digest.hexdigest(),
                    "bundle_files": records, "asset_kind": kind,
                    "checkpoint_sha256": checkpoint_hashes, "runtime_spec": {
                        "entrypoint": str(entry), "accelerator": accelerator,
                        "network": False, "timeout_seconds": float(spec.get("timeout_seconds", 120)),
                        "runtime_requirements": requirements},
                    "source_urls": list(source_urls or []), "visibility": "shared",
                    "provenance": dict(provenance or {"origin": "workspace"}),
                    "status": "candidate", "tests": []}
        # _write_manifest expects a single source; stage package explicitly.
        with _registry_lock(self.root):
            family = self.root / name
            version = max([int(p.name[1:]) for p in family.glob("v[0-9]*") if p.name[1:].isdigit()] or [0]) + 1
            target = family / f"v{version:03d}"
            staging = family / f".v{version:03d}.staging-{uuid.uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                for record in records:
                    self.cas.materialize(str(record["blob_uri"]),
                        staging / "bundle" / str(record["path"]),
                        executable=bool(record.get("executable")))
                manifest.update(version=version, tool_id=f"{name}:v{version:03d}", created_unix=time.time())
                (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                (self.root / "_manuals" / name / f"v{version:03d}").mkdir(parents=True, exist_ok=True)
                (self.root / "_manuals" / name / f"v{version:03d}" / "r001.json").write_text(
                    json.dumps({"tool_id": manifest["tool_id"], "manual_revision": 1,
                        "manual": _manual(description, input_schema, output_schema)}, indent=2) + "\n")
                staging.replace(target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return {"tool_id": manifest["tool_id"], "status": "candidate"}

    def inspect(self, tool_id: str, *, include_source: bool = False) -> dict[str, Any]:
        path = self._path(tool_id)
        manifest = json.loads((path / "manifest.json").read_text())
        runtime = dict(manifest.get("runtime_spec") or {})
        source = path / "bundle" / str(runtime["entrypoint"]) if runtime else path / "tool.py"
        if not source.is_file() or _sha256(source) != manifest.get("source_sha256"):
            raise AssetError("Tool source hash mismatch")
        if runtime:
            records = manifest.get("bundle_files")
            if isinstance(records, list):
                digest = hashlib.sha256()
                for record in records:
                    bundled = path / "bundle" / str(record.get("path") or "")
                    try:
                        blob = self.cas.resolve(str(record.get("blob_uri") or ""))
                    except ContentAddressedStoreError as exc:
                        raise AssetError(str(exc)) from exc
                    if (not bundled.is_file() or bundled.is_symlink()
                            or bundled.stat().st_size != int(record.get("bytes", -1))
                            or (not os.path.samefile(bundled, blob)
                                and _sha256(bundled) != record.get("sha256"))):
                        raise AssetError("capability bundle CAS reference mismatch")
                    digest.update(str(record["path"]).encode() + b"\0")
                    digest.update(bytes.fromhex(str(record["sha256"])))
                if digest.hexdigest() != manifest.get("bundle_tree_sha256"):
                    raise AssetError("capability bundle manifest hash mismatch")
            elif _tree_sha256(path / "bundle") != manifest.get("bundle_tree_sha256"):
                raise AssetError("capability bundle hash mismatch")
        receipts = self._test_receipts(tool_id)
        if receipts:
            manifest = {**manifest, "status": receipts[-1]["status"],
                        "test_receipt_sha256": receipts[-1]["receipt_sha256"]}
        result = {"manifest": manifest, "manual": self.manual(tool_id)}
        if include_source:
            result["source"] = source.read_text()
        return result

    def manual(self, tool_id: str) -> dict[str, Any]:
        path = self.root / "_manuals" / tool_id.partition(":")[0] / tool_id.partition(":")[2]
        revisions = sorted(path.glob("r*.json")) if path.is_dir() else []
        if revisions:
            return json.loads(revisions[-1].read_text())
        manifest = self.inspect(tool_id)["manifest"]
        return {"tool_id": tool_id, "manual": _manual(manifest["description"],
            manifest["input_schema"], manifest["output_schema"])}

    def revise_manual(self, tool_id: str, manual: Mapping[str, Any], *, evidence_paths=None):
        if not evidence_paths:
            raise AssetError("manual revision requires evidence")
        manifest = self.inspect(tool_id)["manifest"]
        normalized = _manual(manifest["description"], manifest["input_schema"], manifest["output_schema"], manual)
        directory = self.root / "_manuals" / manifest["name"] / tool_id.partition(":")[2]
        with _registry_lock(self.root):
            directory.mkdir(parents=True, exist_ok=True)
            revisions = [int(path.stem[1:]) for path in directory.glob("r*.json")
                         if path.stem[1:].isdigit()]
            revision = max(revisions, default=0) + 1
            path = directory / f"r{revision:03d}.json"
            evidence_dir = directory / f"r{revision:03d}_evidence"
            temporary_dir = directory / f".r{revision:03d}_evidence-{uuid.uuid4().hex}"
            temporary_path = directory / f".r{revision:03d}-{uuid.uuid4().hex}.json"
            records = []
            try:
                temporary_dir.mkdir(exist_ok=False)
                for index, value in enumerate(evidence_paths, 1):
                    source = Path(value).resolve()
                    if not source.is_file():
                        raise AssetError(f"manual evidence does not exist: {source}")
                    destination = temporary_dir / f"{index:03d}_{source.name}"
                    shutil.copy2(source, destination)
                    records.append({"artifact_uri":
                        f"{evidence_dir.name}/{destination.name}",
                        "sha256": _sha256(destination)})
                temporary_path.write_text(json.dumps({"tool_id": tool_id,
                    "manual_revision": revision, "manual": normalized,
                    "evidence": records}, indent=2) + "\n")
                temporary_dir.replace(evidence_dir)
                temporary_path.replace(path)
            finally:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                temporary_path.unlink(missing_ok=True)
        return {"tool_id": tool_id, "manual_revision": revision}

    def run(self, tool_id: str, payload: Mapping[str, Any]):
        if self.runtime is None:
            raise AssetError("Tool runtime was not configured for this registry process")
        inspected = self.inspect(tool_id)
        manifest = inspected["manifest"]
        _validate(dict(payload), manifest["input_schema"], "Tool input")
        result = self.runtime.execute(self._path(tool_id), dict(payload))
        _validate(result, manifest["output_schema"], "Tool output")
        return result

    def test_tool(self, tool_id: str, cases: list[Mapping[str, Any]]):
        if not cases:
            raise AssetError("Tool tests are required")
        manifest = json.loads((self._path(tool_id) / "manifest.json").read_text())
        results = []
        for case in cases:
            try:
                actual = self.run(tool_id, case.get("input") or {})
                _validate(case.get("expected"), manifest["output_schema"], "expected Tool output")
                results.append({"passed": _same(actual, case.get("expected")),
                                "actual": actual, "expected": case.get("expected")})
            except Exception as exc:
                results.append({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        status = "verified" if all(item.get("passed") is True for item in results) else "rejected"
        directory = self.root / "_tests" / tool_id.partition(":")[0] / tool_id.partition(":")[2]
        receipt = {"protocol": "roboforge-tool-test-v1", "tool_id": tool_id,
                   "source_sha256": manifest["source_sha256"], "status": status,
                   "cases": results, "tested_unix": time.time()}
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        receipt["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        with _registry_lock(self.root):
            directory.mkdir(parents=True, exist_ok=True)
            sequence = len(list(directory.glob("r*.json"))) + 1
            target = directory / f"r{sequence:03d}.json"
            temporary = target.with_suffix(f".tmp-{uuid.uuid4().hex}")
            temporary.write_text(encoded); temporary.replace(target)
        if status != "verified":
            raise AssetError("Tool contract tests failed")
        return {"tool_id": tool_id, "status": "verified", "results": results}

    def _test_receipts(self, tool_id: str):
        directory = self.root / "_tests" / tool_id.partition(":")[0] / tool_id.partition(":")[2]
        receipts = []
        for path in sorted(directory.glob("r*.json")) if directory.is_dir() else []:
            value = json.loads(path.read_text())
            recorded = value.pop("receipt_sha256", None)
            encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
            expected = hashlib.sha256(encoded.encode()).hexdigest()
            if recorded != expected:
                raise AssetError(f"Tool test receipt hash mismatch: {path}")
            receipts.append({**value, "receipt_sha256": recorded})
        return receipts

    def list_all(self):
        rows=[]
        for path in self.root.glob("*/v*/manifest.json"):
            manifest=json.loads(path.read_text());receipts=self._test_receipts(manifest["tool_id"])
            if receipts:
                manifest={**manifest,"status":receipts[-1]["status"],
                          "test_receipt_sha256":receipts[-1]["receipt_sha256"]}
            promotions = self._promotion_receipts(manifest["tool_id"])
            if promotions:
                manifest = {**manifest, "status": "promoted",
                            "promotion_receipt_sha256": promotions[-1]["receipt_sha256"]}
            rows.append(manifest)
        return sorted(rows,key=lambda row:row.get("tool_id", ""))

    def tested(self):
        return [row for row in self.list_all() if row.get("status") in {"verified", "promoted"}]

    def promoted(self):
        return [row for row in self.list_all() if row.get("status") == "promoted"]

    def list_summaries(self):
        return [{key: row.get(key) for key in ("tool_id", "name", "version", "description",
            "input_schema", "output_schema", "status", "runtime_spec", "dependencies")}
            for row in self.list_all()]

    def search(self, query: str, limit: int = 8, statuses: set[str] | None = None):
        allowed = {"promoted"} if statuses is None else set(statuses)
        rows = [row for row in self.list_summaries() if row.get("status") in allowed]
        return rank_records(query, rows,
            text_fields=("tool_id", "name", "description", "input_schema", "output_schema"),
            id_field="tool_id", limit=limit)

    def runtime_functions(self):
        return {row["tool_id"]: (lambda payload, _id=row["tool_id"]: self.run(_id, payload))
                for row in self.tested()}

    def _promotion_receipts(self, tool_id: str) -> list[dict[str, Any]]:
        directory = self.root / "_admissions" / tool_id.partition(":")[0] / tool_id.partition(":")[2]
        rows = []
        for path in sorted(directory.glob("r*.json")) if directory.is_dir() else []:
            value = json.loads(path.read_text())
            recorded = value.pop("receipt_sha256", None)
            if recorded != hashlib.sha256((json.dumps(value, indent=2,
                    sort_keys=True) + "\n").encode()).hexdigest():
                raise AssetError(f"Tool promotion receipt hash mismatch: {path}")
            rows.append({**value, "receipt_sha256": recorded})
        return rows

    def promote(self, tool_id: str, *, evidence: list[Mapping[str, Any]],
                applicability: Mapping[str, Any] | None = None) -> dict[str, Any]:
        manifest = self.inspect(tool_id)["manifest"]
        if manifest.get("status") not in {"verified", "promoted"}:
            raise AssetError("only a verified Tool can be promoted")
        if not evidence:
            raise AssetError("Tool promotion requires successful integration evidence")
        receipt = {"protocol": "roboforge-tool-admission-v1", "tool_id": tool_id,
                   "source_sha256": manifest["source_sha256"],
                   "test_receipt_sha256": manifest.get("test_receipt_sha256"),
                   "evidence": [dict(item) for item in evidence],
                   "applicability": dict(applicability or {}),
                   "promoted_unix": time.time()}
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        receipt["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        with _registry_lock(self.root):
            directory = self.root / "_admissions" / tool_id.partition(":")[0] / tool_id.partition(":")[2]
            directory.mkdir(parents=True, exist_ok=True)
            sequence = len(list(directory.glob("r*.json"))) + 1
            target = directory / f"r{sequence:03d}.json"
            _atomic_json(target, receipt)
        return {"tool_id": tool_id, "status": "promoted",
                "receipt_sha256": receipt["receipt_sha256"]}


class _JsonAssetLibrary:
    id_field = "asset_id"
    folder = "assets"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _id(self, value: str):
        name, separator, version = str(value).partition(":")
        path = (self.root / name / version / "manifest.json").resolve()
        if not separator or self.root not in path.parents or not path.is_file():
            raise FileNotFoundError(value)
        return path

    def _save(self, name: str, payload: dict[str, Any], evidence_paths=None, attachments=None):
        name = _name(name)
        with _registry_lock(self.root):
            family = self.root / name
            versions = [int(path.name[1:]) for path in family.glob("v[0-9]*") if path.name[1:].isdigit()]
            version = max(versions, default=0) + 1
            target = family / f"v{version:03d}"
            staging = family / f".v{version:03d}.staging-{uuid.uuid4().hex}"
            staging.mkdir(parents=True, exist_ok=False)
            try:
                payload.update(name=name, version=version, **{self.id_field: f"{name}:v{version:03d}"},
                              created_unix=time.time())
                if evidence_paths:
                    directory = staging / "evidence"
                    directory.mkdir()
                    records = []
                    for index, value in enumerate(evidence_paths, 1):
                        source = Path(value).resolve()
                        if not source.is_file():
                            raise AssetError(f"evidence file does not exist: {source}")
                        destination = directory / f"{index:03d}_{source.name}"
                        shutil.copy2(source, destination)
                        records.append({"artifact_uri": str(destination.relative_to(staging)),
                                        "sha256": _sha256(destination)})
                    payload["evidence"] = records
                for filename, source_value in dict(attachments or {}).items():
                    source = Path(source_value).resolve()
                    relative = Path(str(filename))
                    if relative.is_absolute() or ".." in relative.parts or not source.is_file():
                        raise AssetError("invalid asset attachment")
                    destination = staging / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                (staging / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
                staging.replace(target)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
        return {self.id_field: payload[self.id_field], "status": payload.get("status", "recorded")}

    def inspect(self, asset_id: str):
        path = self._id(asset_id)
        payload = json.loads(path.read_text())
        for record in payload.get("evidence") or []:
            evidence = path.parent / str(record["artifact_uri"])
            if not evidence.is_file() or _sha256(evidence) != record.get("sha256"):
                raise AssetError(f"asset evidence hash mismatch: {asset_id}")
        promotions = self._promotion_receipts(asset_id)
        if promotions:
            payload = {**payload, "status": "promoted",
                       "promotion_receipt_sha256": promotions[-1]["receipt_sha256"]}
        return payload

    def list_summaries(self):
        rows = []
        for path in self.root.glob("*/v*/manifest.json"):
            item = self.inspect(json.loads(path.read_text())[self.id_field])
            rows.append({key: item.get(key) for key in (self.id_field, "name", "task", "summary",
                "applicability", "keywords", "status", "interface", "tool_ids")})
        return rows

    def search(self, query: str, limit: int = 8, statuses: set[str] | None = None):
        allowed = {"promoted"} if statuses is None else set(statuses)
        rows = [row for row in self.list_summaries() if row.get("status") in allowed]
        return rank_records(query, rows,
            text_fields=(self.id_field, "name", "task", "summary", "applicability", "keywords", "interface"),
            id_field=self.id_field, limit=limit)

    def _promotion_receipts(self, asset_id: str) -> list[dict[str, Any]]:
        name, _, version = str(asset_id).partition(":")
        directory = self.root / "_admissions" / name / version
        rows = []
        for path in sorted(directory.glob("r*.json")) if directory.is_dir() else []:
            value = json.loads(path.read_text())
            recorded = value.pop("receipt_sha256", None)
            encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
            if recorded != hashlib.sha256(encoded.encode()).hexdigest():
                raise AssetError(f"asset promotion receipt hash mismatch: {path}")
            rows.append({**value, "receipt_sha256": recorded})
        return rows

    def promote(self, asset_id: str, *, evidence: list[Mapping[str, Any]],
                applicability: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = self.inspect(asset_id)
        if payload.get("status") not in {"verified", "promoted"}:
            raise AssetError("only a verified asset can be promoted")
        if not evidence:
            raise AssetError("asset promotion requires verified evidence")
        receipt = {"protocol": "roboforge-asset-admission-v1",
                   self.id_field: asset_id, "evidence": [dict(item) for item in evidence],
                   "applicability": dict(applicability or {}),
                   "promoted_unix": time.time()}
        encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        receipt["receipt_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        with _registry_lock(self.root):
            name, _, version = str(asset_id).partition(":")
            directory = self.root / "_admissions" / name / version
            directory.mkdir(parents=True, exist_ok=True)
            sequence = len(list(directory.glob("r*.json"))) + 1
            _atomic_json(directory / f"r{sequence:03d}.json", receipt)
        return {self.id_field: asset_id, "status": "promoted",
                "receipt_sha256": receipt["receipt_sha256"]}


class SkillLibrary(_JsonAssetLibrary):
    id_field = "skill_id"

    def freeze(self, *, name: str, task: str, controller: str | Path,
               evidence: Mapping[str, Any] | None = None, tool_ids: list[str] | None = None,
               tools: CapabilityLibrary | None = None, interface: Mapping[str, Any] | None = None,
               evidence_paths=None, **_unused):
        source = Path(controller).resolve()
        if not source.is_file():
            raise AssetError("Skill controller does not exist")
        payload = {"protocol": "roboforge-skill-v1", "task": str(task), "controller_sha256": _sha256(source),
                   "tool_ids": sorted(set(tool_ids or [])), "development_evidence": dict(evidence or {}),
                   "interface": dict(interface or {}), "status": "verified"}
        payload["controller_path"] = "controller.py"
        result = self._save(name, payload, evidence_paths, {"controller.py": source})
        return {**result, "path": str(self._id(result["skill_id"]).parent)}

    def inspect(self, asset_id: str):
        payload = super().inspect(asset_id)
        controller = self._id(asset_id).parent / str(payload.get("controller_path") or "controller.py")
        if not controller.is_file() or _sha256(controller) != payload.get("controller_sha256"):
            raise AssetError(f"Skill controller hash mismatch: {asset_id}")
        return {**payload, "controller": controller.read_text()}


class ExperienceLibrary(_JsonAssetLibrary):
    id_field = "experience_id"

    def register(self, *, name: str, summary: str, applicability: str,
                 keywords: list[str] | None = None, evidence_paths=None, **payload):
        if not evidence_paths:
            raise AssetError("Experience evidence is required")
        if payload.get("outcome", "success") != "success":
            raise AssetError("failed or unresolved findings must be recorded as a Capability Gap")
        return self._save(name, {"protocol": "roboforge-experience-v2", "summary": str(summary),
            "applicability": str(applicability), "keywords": list(keywords or []),
            "outcome": "success", "provenance": dict(payload.get("provenance") or {}),
            "status": "verified"}, evidence_paths)


class CapabilityGapLibrary(_JsonAssetLibrary):
    id_field = "gap_id"

    def publish(self, *, name: str, task: str, failure_summary: str,
                evidence_paths=None, status: str = "observed", **payload):
        if not evidence_paths:
            raise AssetError("Gap evidence is required")
        return self._save(name, {"protocol": "roboforge-gap-v1", "task": str(task),
            "failure_summary": str(failure_summary), "status": str(status), **payload}, evidence_paths)


class AssetRegistry:
    """Progressive, summary-first facade used by ContextBuilder."""

    def __init__(self, *, tools=None, skills=None, experiences=None, gaps=None):
        self.tools, self.skills, self.experiences, self.gaps = tools, skills, experiences, gaps

    def search(self, query: str, limit: int = 5):
        result = {}
        for name, library in (("tools", self.tools), ("skills", self.skills),
                              ("experiences", self.experiences), ("gaps", self.gaps)):
            if library is not None:
                result[name] = library.search(query, limit=limit)
        return result

    def inspect(self, asset_id: str):
        for library in (self.tools, self.skills, self.experiences, self.gaps):
            if library is None:
                continue
            try:
                value = library.inspect(asset_id)
                if library is self.tools and isinstance(value, dict):
                    value = dict(value)
                    value.pop("source", None)
                    if hasattr(library, "manual"):
                        value["manual"] = library.manual(asset_id)
                return value
            except (FileNotFoundError, KeyError):
                continue
        raise AssetError(f"unknown asset: {asset_id}")

    def load_source(self, asset_id: str):
        if self.tools is None:
            raise AssetError(asset_id)
        try:
            return self.tools.inspect(asset_id, include_source=True)
        except TypeError:
            value = self.tools.inspect(asset_id)
            if isinstance(value, dict) and "source" in value:
                return value
            raise


__all__ = ["AssetError", "AssetRegistry", "CapabilityLibrary", "SkillLibrary",
           "ExperienceLibrary", "CapabilityGapLibrary"]
