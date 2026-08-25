"""Real capability acquisition, validation, registration and Adapter binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.parse
import zipfile
from typing import Any, Mapping

from ..web import download_public_file, fetch_web_page, search_web


class CapabilityError(RuntimeError):
    pass


class CapabilityManager:
    """Coordinates shared immutable assets and per-episode runtime binding."""

    def __init__(self, *, asset_root: str | Path, workspace: Any, adapter: Any,
                 tool_library: Any = None, skill_library: Any = None,
                 experience_library: Any = None, gap_library: Any = None):
        self.asset_root = Path(asset_root).resolve(); self.asset_root.mkdir(parents=True, exist_ok=True)
        self.workspace, self.adapter = workspace, adapter
        self.tool_library, self.skill_library = tool_library, skill_library
        self.experience_library, self.gap_library = experience_library, gap_library
        self._bound: dict[str, Any] = {}

    def search(self, query: str, limit: int = 5):
        result = {}
        for name, library in (("tools", self.tool_library), ("skills", self.skill_library),
                              ("experiences", self.experience_library), ("gaps", self.gap_library)):
            if library is not None: result[name] = library.search(query, limit=limit)
        return result

    def inspect(self, asset_id: str):
        for library in (self.tool_library, self.skill_library, self.experience_library, self.gap_library):
            if library is None: continue
            try: return library.inspect(asset_id)
            except (FileNotFoundError, KeyError, ValueError): pass
        raise CapabilityError(f"unknown asset: {asset_id}")

    def load_tool_source(self, tool_id: str):
        if self.tool_library is None: raise CapabilityError("Tool library unavailable")
        return self.tool_library.inspect(tool_id, include_source=True)

    def web_search(self, query: str, limit: int = 5): return search_web(query, limit)
    def fetch_page(self, url: str, max_chars: int = 30000): return fetch_web_page(url, max_chars)

    def download(self, url: str, filename: str, sha256: str | None = None) -> dict[str, Any]:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https": raise CapabilityError("only HTTPS public assets are allowed")
        destination = (self.workspace.root / filename).resolve()
        if self.workspace.root not in destination.parents: raise CapabilityError("download escapes workspace")
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_public_file(url, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if sha256 and digest.casefold() != str(sha256).casefold():
            destination.unlink(missing_ok=True); raise CapabilityError("download checksum mismatch")
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

    def _extract(self, source: Path, destination: Path):
        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if destination not in target.parents: raise CapabilityError("archive path traversal")
                    if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                        raise CapabilityError("archive symlinks are not allowed")
                archive.extractall(destination); return
        if tarfile.is_tarfile(source):
            with tarfile.open(source) as archive:
                for member in archive.getmembers():
                    target = (destination / member.name).resolve()
                    if destination not in target.parents: raise CapabilityError("archive path traversal")
                    if member.issym() or member.islnk() or member.isdev():
                        raise CapabilityError("archive links and devices are not allowed")
                archive.extractall(destination); return
        shutil.copy2(source, destination / source.name)

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
        if result.get("status") != "tested": raise CapabilityError("Tool contract tests failed")
        manifest = self.tool_library.inspect(tool_id)["manifest"]
        function = self.tool_library.runtime_functions().get(tool_id)
        if function is None: raise CapabilityError("tested Tool has no runtime function")
        self.adapter.register_capability(tool_id, function, manifest)
        self._bound[tool_id] = function
        return {**result, "bound": True}

    def register_skill(self, **payload):
        if self.skill_library is None: raise CapabilityError("Skill library unavailable")
        values = dict(payload)
        controller = values.get("controller")
        if controller is not None and not Path(str(controller)).is_absolute():
            values["controller"] = str((self.workspace.root / str(controller)).resolve())
        values["evidence_paths"] = [self._asset_path(item) for item in values.get("evidence_paths", [])]
        if not self._verified_skill_evidence(values["evidence_paths"]):
            raise CapabilityError(
                "Skill requires successful Adapter evidence for the current Controller")
        tested = {item["tool_id"] for item in self.tool_library.tested()} \
            if self.tool_library is not None else set()
        missing = set(values.get("tool_ids") or []) - tested
        if missing:
            raise CapabilityError(f"Skill references untested Tools: {sorted(missing)}")
        values.setdefault("tools", self.tool_library)
        return self.skill_library.freeze(**values)

    def _verified_skill_evidence(self, paths: list[str]) -> bool:
        if not self.workspace.controller.is_file():
            return False
        controller_sha = hashlib.sha256(self.workspace.controller.read_bytes()).hexdigest()
        identity_provider = getattr(self.adapter, "execution_identity", None)
        identity = identity_provider() if callable(identity_provider) else identity_provider
        for value in paths:
            try:
                evidence = json.loads(Path(value).read_text())
            except (OSError, json.JSONDecodeError):
                continue
            execution = evidence.get("execution") if isinstance(evidence, Mapping) else None
            report = evidence.get("sensor_report") if isinstance(evidence, Mapping) else None
            receipt = evidence.get("verification_receipt") if isinstance(evidence, Mapping) else None
            declared = [report.get(key) for key in ("sensor_success", "success", "verified",
                "sensor_verification_passed") if isinstance(report, Mapping) and key in report]
            if (isinstance(execution, Mapping) and execution.get("completed") is True
                    and not execution.get("error")
                    and isinstance(receipt, Mapping) and receipt.get("verified") is True
                    and evidence.get("controller_sha256") == controller_sha
                    and receipt.get("controller_sha256") == controller_sha
                    and receipt.get("environment_identity") == identity
                    and declared and any(item is True for item in declared)):
                return True
        return False

    def register_experience(self, **payload):
        if self.experience_library is None: raise CapabilityError("Experience library unavailable")
        values = dict(payload)
        values["evidence_paths"] = [self._asset_path(item) for item in values.get("evidence_paths", [])]
        return self.experience_library.register(**values)

    def record_gap(self, **payload):
        if self.gap_library is None: raise CapabilityError("Gap library unavailable")
        values = dict(payload)
        values["evidence_paths"] = [self._asset_path(item) for item in values.get("evidence_paths", [])]
        return self.gap_library.publish(**values)

    def _asset_path(self, value: Any) -> str:
        encoded = str(value)
        if encoded.startswith("workspace://"):
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
        if not any(path == root or root in path.parents for root in roots):
            raise CapabilityError("asset evidence must be inside a registered artifact root")
        return str(path)

    def bind_shared_tools(self):
        if self.tool_library is None: return []
        bound = []
        for manifest in self.tool_library.tested():
            tool_id = manifest["tool_id"]
            function = self.tool_library.runtime_functions().get(tool_id)
            if function is None: continue
            self.adapter.register_capability(tool_id, function, manifest)
            self._bound[tool_id] = function; bound.append(tool_id)
        return sorted(bound)


__all__ = ["CapabilityError", "CapabilityManager"]
