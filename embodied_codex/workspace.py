"""Persistent coding workspace exposed to the engineering model."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


class WorkspaceError(RuntimeError): pass


class TaskWorkspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise WorkspaceError("workspace path must be relative")
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise WorkspaceError("path escapes task workspace")
        return path

    def list_files(self, pattern: str = "**/*") -> list[str]:
        items = []
        for path in self.root.glob(pattern):
            if path.is_file(): items.append(str(path.relative_to(self.root)))
        return sorted(items)[:2000]

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> dict[str, Any]:
        target = self._path(path)
        if not target.exists():
            return {"path":path,"exists":False,"start_line":1,"end_line":0,
                    "total_lines":0,"content":""}
        if not target.is_file():raise WorkspaceError("workspace path is not a file")
        lines = target.read_text().splitlines()
        start = max(1, int(start_line)); end = max(start, min(int(end_line), start + 999))
        return {"path": path,"exists":True,"start_line": start, "end_line": min(end, len(lines)),
                "total_lines": len(lines), "content": "\n".join(lines[start-1:end])}

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        target = self._path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content))
        return {"path": path, "bytes": target.stat().st_size}

    def replace_in_file(self, path: str, old: str, new: str) -> dict[str, Any]:
        target = self._path(path); text = target.read_text()
        count = text.count(old)
        if count != 1: raise WorkspaceError(f"old text must occur exactly once; found {count}")
        target.write_text(text.replace(old, new, 1))
        return {"path": path, "replaced": True, "bytes": target.stat().st_size}

    def run_command(
        self, argv: list[str], *, timeout_seconds: float = 120,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise WorkspaceError("argv must be a nonempty string list")
        timeout = min(max(float(timeout_seconds), 0.1), 600.0)
        process_env = os.environ.copy()
        for key, value in dict(env or {}).items():
            if not key.startswith(("PYTHON", "CUDA", "MUJOCO", "HF_", "TRANSFORMERS_")):
                raise WorkspaceError(f"environment key not allowed: {key}")
            process_env[str(key)] = str(value)
        try:
            result = subprocess.run(
                argv, cwd=self.root, env=process_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            )
            return {"argv": argv, "exit_code": result.returncode,
                    "output": result.stdout[-30000:]}
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            return {"argv": argv, "exit_code": None, "timed_out": True,
                    "output": output[-30000:]}

__all__ = ["TaskWorkspace", "WorkspaceError"]
