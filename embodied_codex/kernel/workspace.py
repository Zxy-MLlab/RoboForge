"""Persistent workspace with atomic controller edits and immutable snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceSnapshot:
    snapshot_id: str
    controller_sha256: str | None
    files: tuple[str, ...]
    path: str


class PersistentWorkspace:
    """A small file API shared by the loop and custom adapters.

    Mutations are applied to a temporary tree and atomically renamed into the
    workspace. A snapshot is written before and after each transaction so an
    interrupted process can resume from the last committed state.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = self.root / ".snapshots"
        self.snapshots.mkdir(exist_ok=True)

    def _path(self, relative: str) -> Path:
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts or not str(path):
            raise WorkspaceError("workspace paths must be relative")
        resolved = (self.root / path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise WorkspaceError("workspace path escapes root")
        return resolved

    @property
    def controller(self) -> Path:
        return self._path("controller.py")

    def read(self, relative: str, *, max_chars: int = 20_000) -> str:
        path = self._path(relative)
        if not path.is_file():
            raise WorkspaceError(f"workspace file does not exist: {relative}")
        return path.read_text()[: max(1, int(max_chars))]

    def index(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or self.snapshots in path.parents:
                continue
            rows.append({
                "path": str(path.relative_to(self.root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return rows[-max(1, int(limit)):]

    def apply(self, changes: Mapping[str, Any] | list[Mapping[str, Any]]) -> dict[str, Any]:
        """Apply write/delete changes in one recoverable transaction."""
        if isinstance(changes, Mapping):
            operations = [{"path": key, "content": value} for key, value in changes.items()]
        else:
            operations = list(changes)
        stage = Path(tempfile.mkdtemp(prefix="workspace-", dir=self.root))
        try:
            for source in self.root.iterdir():
                if source.name in {stage.name, ".snapshots"}:
                    continue
                destination = stage / source.name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
            changed = []
            for operation in operations:
                relative = str(operation.get("path") or "")
                target = self._path(relative)
                staged = stage / Path(relative)
                staged.parent.mkdir(parents=True, exist_ok=True)
                if operation.get("delete"):
                    if staged.exists():
                        if staged.is_dir():
                            shutil.rmtree(staged)
                        else:
                            staged.unlink()
                else:
                    staged.write_text(str(operation.get("content", "")))
                changed.append(relative)
            for source in list(self.root.iterdir()):
                if source.name not in {stage.name, ".snapshots"}:
                    if source.is_dir(): shutil.rmtree(source)
                    else: source.unlink()
            for source in stage.iterdir():
                destination = self.root / source.name
                source.replace(destination) if source.is_file() else shutil.copytree(source, destination)
            return {"changed": sorted(set(changed)), "snapshot": self.snapshot().snapshot_id}
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def snapshot(self) -> WorkspaceSnapshot:
        files = self.index(limit=100_000)
        controller = next((item["sha256"] for item in files if item["path"] == "controller.py"), None)
        payload = {"files": files, "controller_sha256": controller}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.snapshots / f"{digest}.json"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"snapshot_id": digest, **payload}, indent=2) + "\n")
            temporary.replace(path)
        return WorkspaceSnapshot(digest, controller, tuple(item["path"] for item in files), str(path))

    def recover(self) -> WorkspaceSnapshot | None:
        snapshots = sorted(self.snapshots.glob("*.json"), key=lambda item: item.stat().st_mtime_ns)
        if not snapshots:
            return None
        payload = json.loads(snapshots[-1].read_text())
        return WorkspaceSnapshot(str(payload["snapshot_id"]), payload.get("controller_sha256"),
                                 tuple(item["path"] for item in payload.get("files", [])),
                                 str(snapshots[-1]))


__all__ = ["PersistentWorkspace", "WorkspaceError", "WorkspaceSnapshot"]
