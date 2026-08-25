"""Persistent, sandboxed coding workspace used by the canonical kernel."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import os
import stat
import uuid
from typing import Any, Mapping

from .sandbox import SandboxBackend, default_sandbox


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceSnapshot:
    snapshot_id: str
    controller_sha256: str | None
    files: tuple[str, ...]
    path: str


class PersistentWorkspace:
    """A workspace with bounded reads, atomic edits, snapshots and recovery."""

    def __init__(self, root: str | Path, *, sandbox: SandboxBackend | None = None,
                 require_sandbox: bool = True):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_root = self.root.parent / "workspace_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.stage_root = self.root.parent / "staged_worktree"
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self._locked_files: dict[str, str] = {}
        self._protected_paths: set[Path] = set()
        self.sandbox = sandbox or default_sandbox()
        if require_sandbox:
            try: self.sandbox.require()
            except Exception as exc: raise WorkspaceError(str(exc)) from exc

    def _path(self, relative: str) -> Path:
        candidate = Path(str(relative))
        if not str(candidate) or candidate.is_absolute() or ".." in candidate.parts:
            raise WorkspaceError("workspace paths must be relative")
        path = (self.root / candidate).resolve()
        if path != self.root and self.root not in path.parents:
            raise WorkspaceError("workspace path escapes root")
        return path

    @property
    def controller(self) -> Path:
        return self._path("controller.py")

    def list_files(self, pattern: str = "**/*") -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file() or self.snapshot_root in path.parents:
                continue
            rows.append({"path": str(path.relative_to(self.root)), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        return rows[:2000]

    def index(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.list_files()
        return rows[-max(1, int(limit)):]

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            return {"path": path, "exists": False, "content": "", "total_lines": 0}
        data = target.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError(
                "binary files cannot be read as text; use view_sensor_artifact") from exc
        if "\0" in text:
            raise WorkspaceError("binary files cannot be read as text; use view_sensor_artifact")
        lines = text.splitlines()
        start = max(1, int(start_line)); end = min(max(start, int(end_line)), start + 399, len(lines))
        return {"path": path, "exists": True, "start_line": start, "end_line": end,
                "total_lines": len(lines), "content": "\n".join(lines[start - 1:end]),
                "next_start_line": end + 1 if end < len(lines) else None}

    def read(self, relative: str, *, max_chars: int = 20_000) -> str:
        result = self.read_file(relative, 1, 10000)
        return str(result.get("content") or "")[:max(1, int(max_chars))]

    def _atomic_tree_update(self, operations: list[Mapping[str, Any]]) -> list[str]:
        stage = Path(tempfile.mkdtemp(prefix="workspace-stage-", dir=self.root))
        backup = self.root.parent / f".{self.root.name}.rollback-{uuid.uuid4().hex}"
        try:
            for source in self.root.iterdir():
                if source.name == stage.name: continue
                destination = stage / source.name
                if source.is_dir(): shutil.copytree(source, destination)
                else: shutil.copy2(source, destination)
            changed = []
            for operation in operations:
                relative = str(operation.get("path") or ""); self._path(relative)
                if relative in self._locked_files:
                    raise WorkspaceError(f"workspace file is immutable: {relative}")
                target = stage / relative; target.parent.mkdir(parents=True, exist_ok=True)
                if operation.get("delete"):
                    if target.exists():
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                else:
                    if "content_base64" in operation:
                        target.write_bytes(base64.b64decode(str(operation["content_base64"]), validate=True))
                    else:
                        target.write_text(str(operation.get("content", "")))
                    if "mode" in operation:
                        target.chmod(int(operation["mode"]) & 0o777)
                changed.append(relative)
            # Keep a recoverable rollback tree until every staged entry has
            # replaced its counterpart.  A failed commit restores the prior
            # workspace instead of leaving a half-applied controller.
            backup.mkdir(parents=True, exist_ok=False)
            for source in list(self.root.iterdir()):
                if source.name != stage.name:
                    destination = backup / source.name
                    source.rename(destination)
            try:
                for source in stage.iterdir():
                    destination = self.root / source.name
                    source.rename(destination)
            except Exception:
                for source in list(self.root.iterdir()):
                    shutil.rmtree(source) if source.is_dir() else source.unlink()
                for source in backup.iterdir():
                    source.rename(self.root / source.name)
                raise
            for source in list(backup.iterdir()):
                shutil.rmtree(source) if source.is_dir() else source.unlink()
            backup.rmdir()
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return sorted(set(changed))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            if backup.exists():
                # Never delete a rollback tree containing the only copy of a
                # user file.  Restore missing entries after any interrupted
                # commit phase and leave conflicts for explicit recovery.
                for source in list(backup.iterdir()):
                    destination = self.root / source.name
                    if not destination.exists():
                        source.rename(destination)
                if not any(backup.iterdir()):
                    backup.rmdir()

    def apply(self, changes: Mapping[str, Any] | list[Mapping[str, Any]]) -> dict[str, Any]:
        operations = ([{"path": key, "content": value} for key, value in changes.items()]
                      if isinstance(changes, Mapping) else list(changes))
        changed = self._atomic_tree_update(operations)
        snapshot = self.snapshot()
        return {"changed": changed, "snapshot": snapshot.snapshot_id}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        return self.apply({path: str(content)})

    def lock_file(self, path: str, expected_sha256: str) -> None:
        target = self._path(path)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != str(expected_sha256):
            raise WorkspaceError(f"cannot lock mismatched workspace file: {path}")
        self._locked_files[str(path)] = str(expected_sha256)

    def add_protected_path(self, path: str | Path) -> None:
        """Protect an external persistent tree during engineering commands."""
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise WorkspaceError(f"protected path does not exist: {resolved}")
        self._protected_paths.add(resolved)

    def replace_file_lines(self, path: str, start_line: int, end_line: int,
                           new_content: str, expected_old_sha256: str | None = None) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file(): raise WorkspaceError(f"file does not exist: {path}")
        lines = target.read_text().splitlines(keepends=True)
        start, end = int(start_line), int(end_line)
        if start < 1 or end < start or end > len(lines): raise WorkspaceError("invalid line range")
        old = "".join(lines[start - 1:end]); digest = hashlib.sha256(old.encode()).hexdigest()
        if expected_old_sha256 and digest != expected_old_sha256: raise WorkspaceError("file changed")
        updated = "".join(lines[:start - 1]) + str(new_content) + "".join(lines[end:])
        return self.write_file(path, updated)

    def run_command(self, argv: list[str], *, timeout_seconds: float = 120,
                    env: Mapping[str, str] | None = None,
                    cwd: str | Path | None = None) -> dict[str, Any]:
        if not argv or not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise WorkspaceError("argv must be a nonempty string list")
        safe_env = {"PYTHONNOUSERSITE": "1"}
        for key, value in dict(env or {}).items():
            if not str(key).startswith(("PYTHON", "CUDA", "MUJOCO", "HF_")):
                raise WorkspaceError(f"environment key not allowed: {key}")
            safe_env[str(key)] = str(value)
        working = self.root if cwd is None else Path(cwd).resolve()
        if working != self.root and self.root not in working.parents:
            raise WorkspaceError("command cwd escapes workspace")
        relative = working.relative_to(self.root)
        stage = Path(tempfile.mkdtemp(prefix="command-", dir=self.stage_root))
        for source in self.root.iterdir():
            destination = stage / source.name
            if source.is_symlink():
                shutil.rmtree(stage, ignore_errors=True)
                raise WorkspaceError("workspace symlinks are not allowed")
            if source.is_dir(): shutil.copytree(source, destination)
            elif source.is_file(): shutil.copy2(source, destination)
        stage_working = stage / relative
        command = []
        for index, value in enumerate(argv):
            candidate = Path(value)
            if candidate.is_absolute():
                resolved = candidate.resolve()
                if resolved == self.root or self.root in resolved.parents:
                    value = str(stage / resolved.relative_to(self.root))
                elif index > 0:
                    shutil.rmtree(stage, ignore_errors=True)
                    raise WorkspaceError("absolute command arguments must reference the workspace")
            command.append(value)
        before = {item["path"]: item["sha256"] for item in self.list_files()}
        try:
            modes = self._protect_run_state()
            try:
                completed = self.sandbox.run(command, cwd=stage_working, env=safe_env,
                    read_write_paths=[stage],
                    timeout_seconds=min(max(float(timeout_seconds), 0.1), 600))
            finally:
                self._restore_modes(modes)
            changed = []
            if completed.returncode == 0 and not completed.timed_out:
                staged = self._validate_staged_tree(stage)
                for locked, expected in self._locked_files.items():
                    item = staged.get(locked)
                    if item is None or item["sha256"] != expected:
                        raise WorkspaceError(f"workspace command changed immutable file: {locked}")
                operations = []
                for relative_path in sorted(staged):
                    if relative_path in self._locked_files:
                        continue
                    operations.append({"path": relative_path,
                        "content_base64": base64.b64encode(
                            (stage / relative_path).read_bytes()).decode("ascii"),
                        "mode": stat.S_IMODE((stage / relative_path).stat().st_mode)})
                operations.extend({"path": relative_path, "delete": True}
                    for relative_path in sorted(set(before) - set(staged)))
                self._atomic_tree_update(operations)
                after = {item["path"]: item["sha256"] for item in self.list_files()}
                changed = sorted(path for path in set(before) | set(after)
                                 if before.get(path) != after.get(path))
                self.snapshot()
            output = (completed.stdout + completed.stderr)[-30000:]
            return {"argv": argv, "exit_code": completed.returncode,
                    "timed_out": completed.timed_out, "output": output,
                    "sandbox": f"{self.sandbox.name}-workspace-v1",
                    "cwd": str(relative) or ".", "changed": changed,
                    "committed": completed.returncode == 0 and not completed.timed_out}
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _protect_run_state(self) -> list[tuple[Path, int]]:
        """Make canonical run state immutable for the lifetime of a command."""
        protected = []
        roots = [self.root.parent, *sorted(self._protected_paths)]
        paths = []
        for root in roots:
            paths.extend([root, *sorted(root.rglob("*"),
                                      key=lambda item: len(item.parts), reverse=True)])
        for path in paths:
            if path == self.stage_root or self.stage_root in path.parents or path.is_symlink():
                continue
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
                protected.append((path, mode))
                path.chmod(0o500 if path.is_dir() else 0o400)
            except FileNotFoundError:
                continue
        return protected

    @staticmethod
    def _restore_modes(modes: list[tuple[Path, int]]) -> None:
        for path, mode in reversed(modes):
            try: path.chmod(mode)
            except FileNotFoundError: pass

    @staticmethod
    def _validate_staged_tree(stage: Path) -> dict[str, dict[str, Any]]:
        files = {}; total = 0
        for path in stage.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise WorkspaceError(f"staged worktree contains unsupported file: {path}")
            if not path.is_file(): continue
            relative = path.relative_to(stage).as_posix()
            if relative.startswith("staged_worktree/"):
                raise WorkspaceError("staged worktree contains a reserved path")
            size = path.stat().st_size; total += size
            if len(files) >= 5000 or total > 512 * 1024 * 1024:
                raise WorkspaceError("staged worktree exceeds the file or byte limit")
            files[relative] = {"bytes": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return files

    def _changed_since_snapshot(self):
        current = {item["path"]: item["sha256"] for item in self.list_files()}
        previous = self.recover()
        if previous is None: return sorted(current)
        payload = json.loads(Path(previous.path).read_text())
        old = {item["path"]: item.get("sha256") for item in payload.get("files", [])}
        return sorted(key for key in set(current) | set(old) if current.get(key) != old.get(key))

    def snapshot(self) -> WorkspaceSnapshot:
        files = self.list_files(); controller = next((x["sha256"] for x in files if x["path"] == "controller.py"), None)
        # File contents are part of the snapshot, so a crash can restore the exact workspace.
        entries = []
        for item in files:
            path = self._path(item["path"])
            data = path.read_bytes()
            try:
                content = data.decode("utf-8")
                entries.append({**item, "encoding": "utf-8", "content": content})
            except UnicodeDecodeError:
                entries.append({**item, "encoding": "base64",
                                "content_base64": base64.b64encode(data).decode("ascii")})
        payload = {"files": entries, "controller_sha256": controller}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.snapshot_root / f"{digest}.json"
        if not path.exists():
            temporary = path.with_suffix(".tmp"); temporary.write_text(json.dumps({"snapshot_id": digest, **payload}, indent=2) + "\n"); temporary.replace(path)
        return WorkspaceSnapshot(digest, controller, tuple(item["path"] for item in files), str(path))

    def recover(self) -> WorkspaceSnapshot | None:
        snapshots = sorted(self.snapshot_root.glob("*.json"), key=lambda p: p.stat().st_mtime_ns)
        if not snapshots: return None
        payload = json.loads(snapshots[-1].read_text())
        return WorkspaceSnapshot(str(payload["snapshot_id"]), payload.get("controller_sha256"),
                                 tuple(item["path"] for item in payload.get("files", [])), str(snapshots[-1]))

    def restore(self, snapshot_id: str | None = None) -> WorkspaceSnapshot:
        snapshot = self.snapshot_root / f"{snapshot_id or self.recover().snapshot_id}.json"
        if not snapshot.is_file(): raise WorkspaceError("snapshot not found")
        payload = json.loads(snapshot.read_text())
        operations = []
        for item in payload.get("files", []):
            operation = {"path": item["path"]}
            if item.get("encoding") == "base64":
                operation["content_base64"] = item.get("content_base64", "")
            else:
                operation["content"] = item.get("content", "")
            operations.append(operation)
        current = {item["path"] for item in self.list_files()}
        operations.extend({"path": path, "delete": True} for path in current - {item["path"] for item in payload.get("files", [])})
        self._atomic_tree_update(operations)
        return WorkspaceSnapshot(str(payload["snapshot_id"]), payload.get("controller_sha256"),
                                 tuple(item["path"] for item in payload.get("files", [])), str(snapshot))


__all__ = ["PersistentWorkspace", "WorkspaceError", "WorkspaceSnapshot"]
