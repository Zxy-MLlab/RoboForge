"""Persistent coding workspace with staged commits and content-addressed snapshots."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import uuid
from typing import Any, Mapping

from .cas import ContentAddressedStore
from .sandbox import SandboxBackend, default_sandbox


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceSnapshot:
    snapshot_id: str
    controller_sha256: str | None
    files: tuple[str, ...]
    path: str


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class PersistentWorkspace:
    """Canonical workspace separated from staging, snapshots and CAS blobs."""

    # Large model/checkpoint files are immutable inputs while a command is
    # running.  Materialising another full copy in every staged worktree would
    # defeat the CAS and can exhaust disk space on long-lived runs.
    _LARGE_FILE_THRESHOLD = 16 * 1024 * 1024

    def __init__(self, root: str | Path, *, sandbox: SandboxBackend | None = None,
                 require_sandbox: bool = True,
                 cas: ContentAddressedStore | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        run_root = self.root.parent
        self.snapshot_root = run_root / "workspace_snapshots"
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        self.stage_root = run_root / "staged_worktree"
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.cas = cas or ContentAddressedStore(run_root / "workspace_cas")
        self._locked_files: dict[str, str] = {}
        self._protected_paths: set[Path] = set()
        self.sandbox = sandbox or default_sandbox()
        if require_sandbox:
            try:
                self.sandbox.require()
            except Exception as exc:
                raise WorkspaceError(str(exc)) from exc

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

    def _record(self, path: Path, relative: str | None = None,
                known: Mapping[str, Any] | None = None) -> dict[str, Any]:
        key = relative or path.relative_to(self.root).as_posix()
        fingerprint = list(_fingerprint(path))
        expected = (str(known["sha256"]) if known and
                    list(known.get("fingerprint") or []) == fingerprint else None)
        stored = self.cas.put(path, expected_sha256=expected)
        value = path.stat(follow_symlinks=False)
        return {"path": key, "bytes": value.st_size, "sha256": stored["sha256"],
                "mode": stat.S_IMODE(value.st_mode), "fingerprint": fingerprint,
                "blob_uri": stored["blob_uri"]}

    def _all_files(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise WorkspaceError(f"workspace symlinks are not allowed: {path}")
            if path.is_file():
                rows.append(self._record(path))
        return rows

    def list_files(self, pattern: str = "**/*") -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob(pattern)):
            if path.is_symlink():
                raise WorkspaceError(f"workspace symlinks are not allowed: {path}")
            if path.is_file():
                row = self._record(path)
                row.pop("fingerprint", None)
                rows.append(row)
        return rows[:2000]

    def index(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.list_files()
        return rows[-max(1, int(limit)):]

    def read_file(self, path: str, start_line: int = 1,
                  end_line: int = 400) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            return {"path": path, "exists": False, "content": "", "total_lines": 0}
        if target.stat().st_size > 16 * 1024 * 1024:
            raise WorkspaceError("large files require a specialized artifact viewer")
        data = target.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("binary files cannot be read as text; use view_sensor_artifact") from exc
        if "\0" in text:
            raise WorkspaceError("binary files cannot be read as text; use view_sensor_artifact")
        lines = text.splitlines()
        start = max(1, int(start_line))
        end = min(max(start, int(end_line)), start + 399, len(lines))
        return {"path": path, "exists": True, "start_line": start,
                "end_line": end, "total_lines": len(lines),
                "content": "\n".join(lines[start - 1:end]),
                "next_start_line": end + 1 if end < len(lines) else None}

    def read(self, relative: str, *, max_chars: int = 20_000) -> str:
        result = self.read_file(relative, 1, 10000)
        return str(result.get("content") or "")[:max(1, int(max_chars))]

    def _clone_tree(self, source_root: Path, destination_root: Path) -> None:
        for source in source_root.rglob("*"):
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            if source.is_symlink():
                raise WorkspaceError("workspace symlinks are not allowed")
            if source.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copystat(source, destination, follow_symlinks=False)
            elif source.is_file():
                row = self._record(source, relative.as_posix())
                self.cas.materialize(row["blob_uri"], destination,
                                     writable=source.stat().st_size < self._LARGE_FILE_THRESHOLD)

    def _commit_staged_tree(self, stage: Path) -> None:
        backup = self.root.parent / f".{self.root.name}.rollback-{uuid.uuid4().hex}"
        self.root.rename(backup)
        try:
            stage.rename(self.root)
            descriptor = os.open(self.root.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            if not self.root.exists() and backup.exists():
                backup.rename(self.root)
            raise
        shutil.rmtree(backup)

    def _atomic_tree_update(self, operations: list[Mapping[str, Any]]) -> list[str]:
        stage = Path(tempfile.mkdtemp(prefix="edit-", dir=self.stage_root))
        try:
            self._clone_tree(self.root, stage)
            changed = []
            for operation in operations:
                relative = str(operation.get("path") or "")
                self._path(relative)
                if relative in self._locked_files:
                    raise WorkspaceError(f"workspace file is immutable: {relative}")
                target = stage / relative
                if operation.get("delete"):
                    if target.exists():
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
                    if "blob_uri" in operation:
                        self.cas.materialize(str(operation["blob_uri"]), temporary, writable=True)
                    elif "content_base64" in operation:
                        temporary.write_bytes(base64.b64decode(
                            str(operation["content_base64"]), validate=True))
                    else:
                        temporary.write_text(str(operation.get("content", "")))
                    if "mode" in operation:
                        temporary.chmod(int(operation["mode"]) & 0o777)
                    os.replace(temporary, target)
                changed.append(relative)
            self._validate_staged_tree(stage)
            self._commit_staged_tree(stage)
            return sorted(set(changed))
        finally:
            shutil.rmtree(stage, ignore_errors=True)

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
        if not target.is_file() or self.cas.digest(target) != str(expected_sha256):
            raise WorkspaceError(f"cannot lock mismatched workspace file: {path}")
        self._locked_files[str(path)] = str(expected_sha256)

    def add_protected_path(self, path: str | Path) -> None:
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise WorkspaceError(f"protected path does not exist: {resolved}")
        self._protected_paths.add(resolved)

    def replace_file_lines(self, path: str, start_line: int, end_line: int,
                           new_content: str,
                           expected_old_sha256: str | None = None) -> dict[str, Any]:
        target = self._path(path)
        if not target.is_file():
            raise WorkspaceError(f"file does not exist: {path}")
        lines = target.read_text().splitlines(keepends=True)
        start, end = int(start_line), int(end_line)
        if start < 1 or end < start or end > len(lines):
            raise WorkspaceError("invalid line range")
        old = "".join(lines[start - 1:end])
        digest = hashlib.sha256(old.encode()).hexdigest()
        if expected_old_sha256 and digest != expected_old_sha256:
            raise WorkspaceError("file changed")
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
        before_rows = self._all_files()
        before = {row["path"]: row for row in before_rows}
        stage = Path(tempfile.mkdtemp(prefix="command-", dir=self.stage_root))
        try:
            self._clone_tree(self.root, stage)
            stage_before = {path.relative_to(stage).as_posix(): {
                "fingerprint": list(_fingerprint(path)), "sha256": before[path.relative_to(stage).as_posix()]["sha256"]}
                for path in stage.rglob("*") if path.is_file()}
            stage_working = stage / relative
            command = []
            for index, value in enumerate(argv):
                candidate = Path(value)
                if candidate.is_absolute():
                    resolved = candidate.resolve()
                    if resolved == self.root or self.root in resolved.parents:
                        value = str(stage / resolved.relative_to(self.root))
                    elif index > 0:
                        raise WorkspaceError("absolute command arguments must reference the workspace")
                command.append(value)
            completed = self.sandbox.run(command, cwd=stage_working, env=safe_env,
                read_write_paths=[stage],
                timeout_seconds=min(max(float(timeout_seconds), 0.1), 600))
            changed = []
            if completed.returncode == 0 and not completed.timed_out:
                staged = self._validate_staged_tree(stage, known=stage_before)
                for locked, expected in self._locked_files.items():
                    item = staged.get(locked)
                    if item is None or item["sha256"] != expected:
                        raise WorkspaceError(f"workspace command changed immutable file: {locked}")
                changed = sorted(path for path in set(before) | set(staged)
                                 if before.get(path, {}).get("sha256")
                                 != staged.get(path, {}).get("sha256"))
                self._commit_staged_tree(stage)
                self.snapshot()
            output = (completed.stdout + completed.stderr)[-30000:]
            return {"argv": argv, "exit_code": completed.returncode,
                    "timed_out": completed.timed_out, "output": output,
                    "sandbox": f"{self.sandbox.name}-workspace-v2",
                    "cwd": str(relative) or ".", "changed": changed,
                    "committed": completed.returncode == 0 and not completed.timed_out}
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _validate_staged_tree(self, stage: Path,
                              known: Mapping[str, Mapping[str, Any]] | None = None
                              ) -> dict[str, dict[str, Any]]:
        files: dict[str, dict[str, Any]] = {}
        total = 0
        for path in stage.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise WorkspaceError(f"staged worktree contains unsupported file: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            size = path.stat().st_size
            total += size
            if len(files) >= 100_000 or total > 8 * 1024 ** 4:
                raise WorkspaceError("staged worktree exceeds the file or byte limit")
            fingerprint = list(_fingerprint(path))
            old = (known or {}).get(relative)
            digest = (str(old["sha256"]) if old and old.get("fingerprint") == fingerprint
                      else self.cas.digest(path))
            files[relative] = {"bytes": size, "sha256": digest,
                               "fingerprint": fingerprint}
        return files

    def _changed_since_snapshot(self):
        current = {item["path"]: item["sha256"] for item in self._all_files()}
        previous = self.recover()
        if previous is None:
            return sorted(current)
        payload = json.loads(Path(previous.path).read_text())
        old = {item["path"]: item.get("sha256") for item in payload.get("files", [])}
        return sorted(key for key in set(current) | set(old)
                      if current.get(key) != old.get(key))

    def snapshot(self) -> WorkspaceSnapshot:
        files = self._all_files()
        entries = [{key: item[key] for key in ("path", "bytes", "sha256", "mode", "blob_uri")}
                   for item in files]
        controller = next((item["sha256"] for item in entries
                           if item["path"] == "controller.py"), None)
        payload = {"protocol": "roboforge-workspace-snapshot-v2",
                   "files": entries, "controller_sha256": controller}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
        path = self.snapshot_root / f"{digest}.json"
        if not path.exists():
            temporary = path.with_suffix(f".tmp-{uuid.uuid4().hex}")
            temporary.write_text(json.dumps({"snapshot_id": digest, **payload},
                                            indent=2, sort_keys=True) + "\n")
            os.replace(temporary, path)
        return WorkspaceSnapshot(digest, controller,
                                 tuple(item["path"] for item in entries), str(path))

    def recover(self) -> WorkspaceSnapshot | None:
        snapshots = sorted(self.snapshot_root.glob("*.json"),
                           key=lambda path: path.stat().st_mtime_ns)
        if not snapshots:
            return None
        payload = json.loads(snapshots[-1].read_text())
        return WorkspaceSnapshot(str(payload["snapshot_id"]),
            payload.get("controller_sha256"),
            tuple(item["path"] for item in payload.get("files", [])),
            str(snapshots[-1]))

    def restore(self, snapshot_id: str | None = None) -> WorkspaceSnapshot:
        recovered = self.recover()
        selected = snapshot_id or (recovered.snapshot_id if recovered else None)
        if not selected:
            raise WorkspaceError("snapshot not found")
        snapshot = self.snapshot_root / f"{selected}.json"
        if not snapshot.is_file():
            raise WorkspaceError("snapshot not found")
        payload = json.loads(snapshot.read_text())
        if payload.get("protocol") != "roboforge-workspace-snapshot-v2":
            raise WorkspaceError("unsupported workspace snapshot protocol")
        operations = [{"path": item["path"], "blob_uri": item["blob_uri"],
                       "mode": item.get("mode", 0o600)}
                      for item in payload.get("files", [])]
        current = {item["path"] for item in self._all_files()}
        expected = {item["path"] for item in payload.get("files", [])}
        operations.extend({"path": path, "delete": True} for path in current - expected)
        self._atomic_tree_update(operations)
        for item in payload.get("files", []):
            if self.cas.digest(self._path(item["path"])) != item["sha256"]:
                raise WorkspaceError(f"restored file checksum mismatch: {item['path']}")
        return WorkspaceSnapshot(str(payload["snapshot_id"]),
            payload.get("controller_sha256"), tuple(sorted(expected)), str(snapshot))


__all__ = ["PersistentWorkspace", "WorkspaceError", "WorkspaceSnapshot"]
