"""Real capability acquisition, validation, registration and Adapter binding."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import stat
import tarfile
import tempfile
import urllib.parse
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Mapping

from ..web import download_public_file, fetch_web_page, search_web


class CapabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractionLimits:
    """Resource limits applied while unpacking untrusted archives."""

    max_files: int = 10000
    max_total_bytes: int = 8 * 1024 ** 3
    max_file_bytes: int = 2 * 1024 ** 3
    max_compression_ratio: float = 100.0

    def __post_init__(self) -> None:
        if int(self.max_files) < 1:
            raise ValueError("max_files must be positive")
        if int(self.max_total_bytes) < 1 or int(self.max_file_bytes) < 1:
            raise ValueError("archive byte limits must be positive")
        if float(self.max_compression_ratio) < 1.0:
            raise ValueError("max_compression_ratio must be at least 1")


class CapabilityManager:
    """Coordinates shared immutable assets and per-episode runtime binding."""

    def __init__(self, *, asset_root: str | Path, workspace: Any, adapter: Any,
                 tool_library: Any = None, skill_library: Any = None,
                 experience_library: Any = None, gap_library: Any = None,
                 extraction_limits: ExtractionLimits | None = None):
        self.asset_root = Path(asset_root).resolve(); self.asset_root.mkdir(parents=True, exist_ok=True)
        self.workspace, self.adapter = workspace, adapter
        self.tool_library, self.skill_library = tool_library, skill_library
        self.experience_library, self.gap_library = experience_library, gap_library
        self.extraction_limits = extraction_limits or ExtractionLimits()
        self._bound: dict[str, Any] = {}
        self._inspected_tools: set[str] = set()

    def search(self, query: str, limit: int = 5, *, include_gaps: bool = False,
               statuses: set[str] | None = None):
        result = {}
        for name, library in (("tools", self.tool_library), ("skills", self.skill_library),
                              ("experiences", self.experience_library)):
            if library is not None:
                result[name] = library.search(query, limit=limit, statuses=statuses)
        if include_gaps and self.gap_library is not None:
            result["gaps"] = self.gap_library.search(query, limit=limit,
                                                      statuses={"observed", "open"})
        return result

    def inspect(self, asset_id: str):
        for library in (self.tool_library, self.skill_library, self.experience_library, self.gap_library):
            if library is None: continue
            try:
                result = library.inspect(asset_id)
                manifest = result.get("manifest") if isinstance(result, Mapping) else None
                if library is self.tool_library and isinstance(manifest, Mapping):
                    tool_id = manifest.get("tool_id")
                    if tool_id:
                        self._inspected_tools.add(str(tool_id))
                return result
            except (FileNotFoundError, KeyError, ValueError): pass
        raise CapabilityError(f"unknown asset: {asset_id}")

    def load_tool_source(self, tool_id: str):
        if self.tool_library is None: raise CapabilityError("Tool library unavailable")
        if str(tool_id) not in self._inspected_tools:
            raise CapabilityError("inspect the Tool manual and schema before loading source")
        detail = self.tool_library.inspect(tool_id, include_source=True)
        source = str(detail.get("source") or "")
        path = self.workspace.root / "inspected_tools" / f"{str(tool_id).replace(':', '_')}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace.write_file(str(path.relative_to(self.workspace.root)), source)
        return {"tool_id": str(tool_id), "materialized": f"workspace://{path.relative_to(self.workspace.root).as_posix()}"}

    def materialize_skill(self, skill_id: str):
        if self.skill_library is None:
            raise CapabilityError("Skill library unavailable")
        detail = self.skill_library.inspect(str(skill_id), include_controller=True)
        path = self.workspace.root / "skills" / f"{str(skill_id).replace(':', '_')}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace.write_file(str(path.relative_to(self.workspace.root)), str(detail["controller"]))
        return {"skill_id": str(skill_id), "materialized": f"workspace://{path.relative_to(self.workspace.root).as_posix()}"}

    def activate_tool(self, tool_id: str):
        """Bind exactly one model-selected promoted Tool to this Adapter."""
        tool_id = str(tool_id)
        if self.tool_library is None:
            raise CapabilityError("Tool library unavailable")
        if tool_id in self._bound:
            return {"tool_id": tool_id, "bound": True, "already_bound": True}
        if tool_id not in self._inspected_tools:
            raise CapabilityError("inspect the Tool manual and schema before activation")
        manifest = self.tool_library.inspect(tool_id)["manifest"]
        if manifest.get("status") != "promoted":
            raise CapabilityError("only a promoted shared Tool can be activated")
        function = self.tool_library.runtime_function(tool_id)
        self.adapter.register_capability(tool_id, function, manifest)
        self._bound[tool_id] = function
        return {"tool_id": tool_id, "bound": True, "already_bound": False}

    @property
    def bound_tool_ids(self):
        return tuple(sorted(self._bound))

    def restore_tool_binding(self, tool_id: str):
        """Restore one previously checkpointed binding, never the whole store."""
        tool_id = str(tool_id)
        if tool_id in self._bound:
            return
        if self.tool_library is None:
            raise CapabilityError("Tool library unavailable")
        manifest = self.tool_library.inspect(tool_id)["manifest"]
        if manifest.get("status") not in {"verified", "promoted"}:
            raise CapabilityError("checkpoint Tool is no longer verified")
        function = self.tool_library.runtime_function(tool_id)
        self.adapter.register_capability(tool_id, function, manifest)
        self._bound[tool_id] = function

    def web_search(self, query: str, limit: int = 5): return search_web(query, limit)
    def fetch_page(self, url: str, max_chars: int = 30000): return fetch_web_page(url, max_chars)

    def download(self, url: str, filename: str, sha256: str | None = None) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https": raise CapabilityError("only HTTPS public assets are allowed")
        destination = (self.workspace.root / filename).resolve()
        if self.workspace.root not in destination.parents: raise CapabilityError("download escapes workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.download-{uuid.uuid4().hex}")
        try:
            download = download_public_file(url, staging)
            digest = str(download["sha256"])
            if sha256 and digest.casefold() != str(sha256).casefold():
                raise CapabilityError("download checksum mismatch")
            self.workspace.ensure_capacity(replacing=destination, new_file_count=1,
                                           new_bytes=int(download["bytes"]))
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
        # download_public_file hashes while streaming; do not read a model or
        # checkpoint back into memory just to verify the optional checksum.
        return {"path": str(destination.relative_to(self.workspace.root)), "sha256": digest,
                "bytes": destination.stat().st_size}

    def unpack(self, path: str, destination: str) -> dict[str, Any]:
        source = (self.workspace.root / path).resolve(); target = (self.workspace.root / destination).resolve()
        if self.workspace.root not in source.parents or not source.is_file(): raise CapabilityError("archive is outside workspace")
        if self.workspace.root not in target.parents: raise CapabilityError("destination escapes workspace")
        self._extract(source, target)
        return {"path": str(target.relative_to(self.workspace.root)),
                "files": [str(item.relative_to(target)) for item in target.rglob("*") if item.is_file()]}

    def build(self, directory: str, argv: list[str] | None = None) -> dict[str, Any]:
        """Build/check an acquired bundle inside the workspace sandbox."""
        target = (self.workspace.root / directory).resolve()
        if (target != self.workspace.root and self.workspace.root not in target.parents) or not target.is_dir():
            raise CapabilityError("build directory is outside workspace")
        command = argv or ["python", "-m", "compileall", "-q", "."]
        snapshot = self.workspace.snapshot()
        try:
            result = self.workspace.run_command(command, timeout_seconds=600, cwd=target)
            if result.get("exit_code") != 0:
                raise CapabilityError(f"capability build failed: {result.get('output', '')[-2000:]}")
        except Exception:
            self.workspace.restore(snapshot.snapshot_id)
            raise
        return {"directory": directory, "command": command, "build": result}

    @staticmethod
    def _archive_member_path(destination: Path, name: str) -> Path:
        # Treat both separators as path separators so a Windows-authored archive
        # cannot smuggle traversal through a Linux extractor.
        normalized = str(name).replace("\\", "/")
        if (not normalized or normalized.startswith("/")
                or PureWindowsPath(normalized).is_absolute()):
            raise CapabilityError("archive absolute paths are not allowed")
        parts = tuple(part for part in normalized.split("/") if part not in {"" , "."})
        if not parts or ".." in parts:
            raise CapabilityError("archive path traversal")
        target = (destination.joinpath(*parts)).resolve()
        if target != destination and destination not in target.parents:
            raise CapabilityError("archive path traversal")
        return target

    def _replace_directory(self, staging: Path, destination: Path) -> None:
        """Atomically replace destination after a complete staging validation."""
        def remove(path: Path) -> None:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = destination.with_name(f".{destination.name}.rollback-{uuid.uuid4().hex}")
        had_destination = destination.exists() or destination.is_symlink()
        try:
            if had_destination:
                destination.rename(backup)
            staging.rename(destination)
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            if destination.exists() or destination.is_symlink():
                remove(destination)
            if had_destination and backup.exists():
                backup.rename(destination)
            raise
        if had_destination:
            remove(backup)

    def _extract(self, source: Path, destination: Path):
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-",
                                         dir=destination.parent))
        try:
            if zipfile.is_zipfile(source):
                self._extract_zip(source, staging)
            elif tarfile.is_tarfile(source):
                self._extract_tar(source, staging)
            else:
                # Non-archives still use the same atomic commit boundary.
                if source.stat().st_size > self.extraction_limits.max_file_bytes:
                    raise CapabilityError("file exceeds extraction limit")
                shutil.copy2(source, staging / source.name)
            files = [item for item in staging.rglob("*") if item.is_file()]
            self.workspace.ensure_capacity(replacing=destination,
                new_file_count=len(files),
                new_bytes=sum(item.stat().st_size for item in files))
            self._replace_directory(staging, destination)
            staging = None  # ownership transferred to destination
        except Exception:
            raise
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def _check_count_and_declared(self, count: int, total: int) -> None:
        limits = self.extraction_limits
        if count > limits.max_files:
            raise CapabilityError("archive contains too many files")
        if total > limits.max_total_bytes:
            raise CapabilityError("archive expands beyond total byte limit")

    def _write_stream(self, stream, target: Path, declared: int, total: list[int]) -> None:
        limits = self.extraction_limits
        if declared < 0 or declared > limits.max_file_bytes:
            raise CapabilityError("archive member exceeds single-file limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with target.open("wb") as output:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                total[0] += len(chunk)
                if written > limits.max_file_bytes:
                    raise CapabilityError("archive member exceeds single-file limit")
                if total[0] > limits.max_total_bytes:
                    raise CapabilityError("archive expands beyond total byte limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written != declared:
            raise CapabilityError("archive member size changed during extraction")

    def _extract_zip(self, source: Path, staging: Path) -> None:
        limits = self.extraction_limits
        total_declared = 0
        total = [0]
        seen: set[str] = set()
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            validated: list[tuple[zipfile.ZipInfo, Path, bool, int]] = []
            for member in members:
                target = self._archive_member_path(staging, member.filename)
                key = target.relative_to(staging).as_posix()
                if key in seen:
                    raise CapabilityError("archive contains duplicate paths")
                seen.add(key)
                mode = (member.external_attr >> 16) & 0o170000
                if mode and mode not in {stat.S_IFREG, stat.S_IFDIR}:
                    raise CapabilityError("archive links and special files are not allowed")
                is_directory = member.is_dir() or mode == stat.S_IFDIR or member.filename.endswith(("/", "\\"))
                if len(seen) > limits.max_files:
                    raise CapabilityError("archive contains too many files")
                declared = int(member.file_size)
                if declared < 0 or declared > limits.max_file_bytes:
                    raise CapabilityError("archive member exceeds single-file limit")
                compressed = int(member.compress_size)
                if not is_directory:
                    ratio = float("inf") if compressed == 0 and declared else (declared / max(1, compressed))
                    if ratio > limits.max_compression_ratio:
                        raise CapabilityError("archive member compression ratio exceeds limit")
                    total_declared += declared
                self._check_count_and_declared(len(seen), total_declared)
                validated.append((member, target, is_directory, declared))
            for member, target, is_directory, declared in validated:
                if is_directory:
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with archive.open(member, "r") as stream:
                        self._write_stream(stream, target, declared, total)

    def _extract_tar(self, source: Path, staging: Path) -> None:
        limits = self.extraction_limits
        total_declared = 0
        total = [0]
        seen: set[str] = set()
        source_bytes = max(1, source.stat().st_size)
        with tarfile.open(source, mode="r:*") as archive:
            validated: list[tuple[tarfile.TarInfo, Path]] = []
            while True:
                member = archive.next()
                if member is None:
                    break
                target = self._archive_member_path(staging, member.name)
                key = target.relative_to(staging).as_posix()
                if key in seen:
                    raise CapabilityError("archive contains duplicate paths")
                seen.add(key)
                if not (member.isdir() or member.isreg()):
                    raise CapabilityError("archive links and special files are not allowed")
                declared = int(member.size)
                if declared < 0 or declared > limits.max_file_bytes:
                    raise CapabilityError("archive member exceeds single-file limit")
                if member.isreg():
                    total_declared += declared
                    if total_declared / source_bytes > limits.max_compression_ratio:
                        raise CapabilityError("archive compression ratio exceeds limit")
                self._check_count_and_declared(len(seen), total_declared)
                validated.append((member, target))
            for member, target in validated:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise CapabilityError("archive member cannot be read")
                    with stream:
                        self._write_stream(stream, target, int(member.size), total)

    def register_tool(self, **payload):
        if self.tool_library is None: raise CapabilityError("Tool library unavailable")
        result = self.tool_library.register_tool(**payload)
        tool_id = result["tool_id"]
        # A manifest is immutable at registration time but remains un-deployable
        # until contract tests pass. test_tool performs the actual binding.
        return {**result, "bound": False, "requires_test": True}

    def register_package(self, **payload):
        if self.tool_library is None:
            raise CapabilityError("Tool library unavailable")
        result = self.tool_library.register_package(**payload)
        return {**result, "bound": False, "requires_test": True}

    def revise_manual(self, **payload):
        if self.tool_library is None:
            raise CapabilityError("Tool library unavailable")
        return self.tool_library.revise_manual(**payload)

    def test_tool(self, tool_id: str, cases: list[Mapping[str, Any]]):
        if self.tool_library is None: raise CapabilityError("Tool library unavailable")
        result = self.tool_library.test_tool(tool_id, cases)
        if result.get("status") != "verified": raise CapabilityError("Tool contract tests failed")
        manifest = self.tool_library.inspect(tool_id)["manifest"]
        function = self.tool_library.runtime_functions().get(tool_id)
        if function is None: raise CapabilityError("tested Tool has no runtime function")
        self.adapter.register_capability(tool_id, function, manifest)
        self._bound[tool_id] = function
        return {**result, "bound": True, "cross_task_visible": False}

    def register_skill(self, **payload):
        if self.skill_library is None: raise CapabilityError("Skill library unavailable")
        values = dict(payload)
        controller = values.get("controller")
        if controller is not None and not Path(str(controller)).is_absolute():
            values["controller"] = str((self.workspace.root / str(controller)).resolve())
        values["evidence_paths"] = [self._asset_path(item) for item in values.get("evidence_paths", [])]
        evidence = self._evidence_records(values["evidence_paths"], require_success=True,
                                          require_current_controller=True)
        if not evidence:
            raise CapabilityError("Skill requires successful Adapter evidence for the frozen Controller")
        observed = set()
        for record in evidence:
            try:
                execution = json.loads(Path(record["path"]).read_text()).get("execution") or {}
                for event in execution.get("rpc_events") or []:
                    if event.get("method") == "use":
                        tool_id = (event.get("arguments") or {}).get("tool_id")
                        if tool_id:
                            observed.add(str(tool_id))
            except (OSError, json.JSONDecodeError):
                continue
        declared = set(values.get("tool_ids") or [])
        if declared != observed:
            raise CapabilityError(f"Skill Tool dependency declaration differs from execution evidence: declared={sorted(declared)}, observed={sorted(observed)}")
        all_dependencies = observed
        native_provider = getattr(self.adapter, "native_capability_manifest", None)
        native_manifest = dict(native_provider() or {}) if callable(native_provider) else {}
        native_ids = set(native_manifest) & all_dependencies
        shared_ids = all_dependencies - native_ids
        values["tool_ids"] = sorted(shared_ids)
        values["observed_tool_ids"] = sorted(all_dependencies)
        existing_requirements = dict(values.get("adapter_requirements") or {})
        existing_requirements["capabilities"] = [dict(native_manifest[item])
            for item in sorted(native_ids)]
        values["adapter_requirements"] = existing_requirements
        tested = {item["tool_id"] for item in self.tool_library.tested()} if self.tool_library is not None else set()
        missing = shared_ids - tested
        if missing:
            raise CapabilityError(f"Skill references untested Tools: {sorted(missing)}")
        values.setdefault("tools", self.tool_library)
        return self.skill_library.freeze(**values)

    def _evidence_records(self, paths: list[str], *, require_success: bool = False,
                          require_current_controller: bool = False) -> list[dict[str, Any]]:
        controller_sha = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest() \
            if self.workspace.controller.is_file() else None
        records = []
        for value in paths:
            try:
                evidence = json.loads(Path(value).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            receipt = evidence.get("verification_receipt") if isinstance(evidence, Mapping) else None
            if not isinstance(receipt, Mapping):
                continue
            identity = evidence.get("environment_identity")
            authentic = (isinstance(identity, Mapping)
                         and receipt.get("controller_sha256") == evidence.get("controller_sha256")
                         and receipt.get("environment_identity") == identity
                         and receipt.get("episode_id") == identity.get("episode_id")
                         and receipt.get("environment_generation") == identity.get("environment_generation")
                         and isinstance(receipt.get("verified"), bool))
            if require_success and receipt.get("verified") is not True:
                authentic = False
            if require_current_controller and evidence.get("controller_sha256") != controller_sha:
                authentic = False
            if authentic:
                records.append({"path": value, "controller_sha256": evidence.get("controller_sha256"),
                    "environment_identity": identity, "verification_receipt": receipt,
                    "sha256": hashlib.sha256(Path(value).read_bytes()).hexdigest()})
        return records

    def _verified_evidence(self, paths: list[str]) -> list[dict[str, Any]]:
        """Compatibility alias for authentic, successful current-controller evidence."""
        return self._evidence_records(paths, require_success=True, require_current_controller=True)

    def register_experience(self, **payload):
        if self.experience_library is None: raise CapabilityError("Experience library unavailable")
        values = dict(payload)
        values["evidence_paths"] = [self._asset_path(item) for item in values.get("evidence_paths", [])]
        evidence = self._evidence_records(values["evidence_paths"])
        if not evidence:
            raise CapabilityError("Experience requires authentic evidence")
        values["outcome"] = str(values.get("outcome") or "mixed")
        if values["outcome"] not in {"success", "failure", "mixed"}:
            raise CapabilityError("Experience outcome must be success, failure, or mixed")
        return self.experience_library.register(**values)

    def promote_asset(self, asset_id: str, evidence_paths: list[str],
                      applicability: Mapping[str, Any] | None = None):
        paths = [self._asset_path(item) for item in evidence_paths]
        evidence = self._evidence_records(paths)
        if not evidence:
            raise CapabilityError("asset promotion requires successful Adapter evidence")
        library = None
        if self.tool_library is not None:
            try:
                self.tool_library.inspect(asset_id); library = self.tool_library
            except (FileNotFoundError, KeyError, ValueError):
                pass
        for candidate in (self.skill_library, self.experience_library):
            if library is not None or candidate is None:
                continue
            try:
                candidate.inspect(asset_id); library = candidate
            except (FileNotFoundError, KeyError, ValueError):
                pass
        if library is None:
            raise CapabilityError(f"unknown promotable asset: {asset_id}")
        if library is self.tool_library:
            integration_ok = False
            for row in evidence:
                try:
                    execution = json.loads(Path(row["path"]).read_text()).get("execution", {})
                except (OSError, json.JSONDecodeError):
                    continue
                if execution.get("completed") is not True or execution.get("error"):
                    continue
                for event in execution.get("rpc_events", []):
                    if (event.get("method") == "use"
                            and (event.get("arguments") or {}).get("tool_id") == asset_id
                            and not (event.get("result") or {}).get("tool_error")):
                        integration_ok = True
                        break
            if not integration_ok:
                raise CapabilityError("Tool promotion requires a successful robot.use() integration evidence")
        return library.promote(asset_id, evidence=evidence,
                               applicability=applicability)

    def record_gap(self, **payload):
        if self.gap_library is None: raise CapabilityError("Gap library unavailable")
        values = dict(payload)
        values["evidence_paths"] = [self._asset_path(item) for item in values.get("evidence_paths", [])]
        return self.gap_library.publish(**values)

    def _asset_path(self, value: Any) -> str:
        encoded = str(value)
        if encoded.startswith("evidence://execution-"):
            sequence = encoded.removeprefix("evidence://execution-")
            if not sequence.isdigit():
                raise CapabilityError("invalid evidence reference")
            matches = list((self.workspace.root.parent / "evidence").glob(
                f"execution-{int(sequence):06d}-*.json"))
            if len(matches) != 1:
                raise CapabilityError("evidence reference is missing or ambiguous")
            path = matches[0]
        elif encoded.startswith("workspace://"):
            path = self.workspace.root / encoded.removeprefix("workspace://")
        elif encoded.startswith("run://"):
            path = self.workspace.root.parent / encoded.removeprefix("run://")
        elif encoded.startswith("artifact://adapter/"):
            path = Path(getattr(self.adapter, "artifact_dir", "")) / encoded.removeprefix("artifact://adapter/")
        else:
            path = Path(encoded)
            if not path.is_absolute():
                path = self.workspace.root / path
        path = path.resolve()
        roots = [self.workspace.root, self.workspace.root.parent / "evidence"]
        adapter_root = getattr(self.adapter, "artifact_dir", None)
        if adapter_root:
            roots.append(Path(adapter_root).resolve())
        roots.extend(Path(value).resolve() for value in
                     getattr(self.adapter, "artifact_roots", []) or [])
        if not any(path == root or root in path.parents for root in roots):
            raise CapabilityError("asset evidence must be inside a registered artifact root")
        return str(path)

__all__ = ["CapabilityError", "CapabilityManager", "ExtractionLimits"]
