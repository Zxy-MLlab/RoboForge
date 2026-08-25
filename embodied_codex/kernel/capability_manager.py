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
        return self.tool_library.inspect(tool_id)

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
        if self.workspace.root not in target.parents or not target.is_dir():
            raise CapabilityError("build directory is outside workspace")
        command = argv or ["python", "-m", "compileall", "-q", "."]
        result = self.workspace.run_command(command, timeout_seconds=600)
        if result.get("exit_code") != 0:
            raise CapabilityError(f"capability build failed: {result.get('output', '')[-2000:]}")
        return {"directory": directory, "command": command, "build": result}

    def _extract(self, source: Path, destination: Path):
        destination.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if destination not in target.parents: raise CapabilityError("archive path traversal")
                archive.extractall(destination); return
        if tarfile.is_tarfile(source):
            with tarfile.open(source) as archive:
                for member in archive.getmembers():
                    target = (destination / member.name).resolve()
                    if destination not in target.parents: raise CapabilityError("archive path traversal")
                archive.extractall(destination); return
        shutil.copy2(source, destination / source.name)

    def register_tool(self, **payload):
        if self.tool_library is None: raise CapabilityError("Tool library unavailable")
        result = self.tool_library.register_tool(**payload)
        tool_id = result["tool_id"]
        # A manifest is immutable at registration time but remains un-deployable
        # until contract tests pass. test_tool performs the actual binding.
        return {**result, "bound": False, "requires_test": True}

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
        return self.skill_library.freeze(**payload)

    def register_experience(self, **payload):
        if self.experience_library is None: raise CapabilityError("Experience library unavailable")
        return self.experience_library.register(**payload)

    def record_gap(self, **payload):
        if self.gap_library is None: raise CapabilityError("Gap library unavailable")
        return self.gap_library.publish(**payload)

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
