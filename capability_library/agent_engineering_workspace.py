"""Evaluator-isolated Codex-style workspace for autonomous capability engineering."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_.-]{1,80}$")
_TEXT_SUFFIXES = {
    ".py", ".json", ".jsonl", ".md", ".txt", ".log", ".csv", ".yaml", ".yml",
}
_EXECUTABLES = {
    "python": "/data/zxy/envs/vla-report/bin/python",
    "pytest": "/data/zxy/envs/vla-report/bin/pytest",
    "git": "/usr/bin/git",
    "rg": "/usr/bin/rg",
    "ls": "/usr/bin/ls",
    "find": "/usr/bin/find",
}


class EngineeringWorkspaceError(ValueError):
    pass


class AgentEngineeringWorkspace:
    """A writable sandbox plus explicitly mounted sensor-only experiment evidence."""

    def __init__(
        self,
        root: str | Path,
        *,
        read_roots: Mapping[str, str | Path] | None = None,
        timeout_sec: int = 120,
        max_output_chars: int = 30000,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.read_roots = {
            str(name): Path(path).resolve()
            for name, path in (read_roots or {}).items()
            if _SAFE_NAME.fullmatch(str(name))
        }
        self.timeout_sec = int(timeout_sec)
        self.max_output_chars = int(max_output_chars)

    @staticmethod
    def _inside(root: Path, relative: str) -> Path:
        if not str(relative).strip() or Path(relative).is_absolute():
            raise EngineeringWorkspaceError("path must be non-empty and relative")
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise EngineeringWorkspaceError("path escapes its declared workspace root")
        return target

    def write(self, path: str, content: str) -> dict[str, Any]:
        target = self._inside(self.root, path)
        encoded = str(content).encode()
        if len(encoded) > 500_000:
            raise EngineeringWorkspaceError("engineering file exceeds 500 KB")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content))
        return {"success": True, "path": str(target.relative_to(self.root)), "bytes": len(encoded)}

    def read(self, path: str, *, max_chars: int = 50000) -> dict[str, Any]:
        target = self._inside(self.root, path)
        if not target.is_file():
            raise FileNotFoundError(path)
        if target.suffix.casefold() not in _TEXT_SUFFIXES:
            return {
                "success": True, "path": str(target.relative_to(self.root)),
                "binary": True, "bytes": target.stat().st_size,
            }
        text = target.read_text(errors="replace")
        limit = max(1000, min(int(max_chars), 100000))
        return {
            "success": True, "path": str(target.relative_to(self.root)),
            "content": text[:limit], "truncated": len(text) > limit,
        }

    def inspect_artifact(
        self, root_name: str, path: str, *, max_chars: int = 50000,
    ) -> dict[str, Any]:
        root = self.read_roots.get(str(root_name))
        if root is None:
            raise EngineeringWorkspaceError(f"unknown evidence root: {root_name}")
        target = self._inside(root, path)
        if not target.is_file():
            raise FileNotFoundError(path)
        relative = str(target.relative_to(root))
        if target.suffix.casefold() not in _TEXT_SUFFIXES:
            return {
                "success": True, "root": root_name, "path": relative,
                "binary": True, "bytes": target.stat().st_size,
            }
        text = target.read_text(errors="replace")
        limit = max(1000, min(int(max_chars), 100000))
        return {
            "success": True, "root": root_name, "path": relative,
            "content": text[:limit], "truncated": len(text) > limit,
        }

    def list_files(self, scope: str = "workspace", pattern: str = "**/*") -> dict[str, Any]:
        root = self.root if scope == "workspace" else self.read_roots.get(scope)
        if root is None:
            raise EngineeringWorkspaceError(f"unknown workspace/evidence scope: {scope}")
        matches = []
        for path in sorted(root.glob(str(pattern or "**/*"))):
            if path.is_file():
                matches.append({
                    "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                })
            if len(matches) >= 500:
                break
        return {"success": True, "scope": scope, "files": matches, "capped": len(matches) == 500}

    def _sandbox_prefix(self) -> list[str]:
        command = [
            "/usr/bin/bwrap", "--die-with-parent", "--unshare-pid", "--new-session",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
            "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/data/zxy/envs/vla-report", "/data/zxy/envs/vla-report",
            "--ro-bind", "/etc", "/etc", "--dir", "/run", "--dir", "/run/systemd",
            "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
            "--bind", str(self.root), "/workspace", "--dir", "/inputs",
        ]
        for name, root in sorted(self.read_roots.items()):
            command.extend(["--ro-bind", str(root), f"/inputs/{name}"])
        command.extend([
            "--proc", "/proc", "--dev", "/dev", "--chdir", "/workspace",
            "--setenv", "HOME", "/workspace", "--setenv", "PATH",
            "/data/zxy/envs/vla-report/bin:/usr/bin:/bin", "--setenv", "LANG", "C.UTF-8",
        ])
        return command

    def run(self, argv: list[str], *, timeout_sec: int | None = None) -> dict[str, Any]:
        if not argv or str(argv[0]) not in _EXECUTABLES:
            raise EngineeringWorkspaceError(
                f"first argv item must be one of {sorted(_EXECUTABLES)}"
            )
        if len(argv) > 100 or any(len(str(item)) > 4000 for item in argv):
            raise EngineeringWorkspaceError("command argv exceeds workspace limits")
        executable = _EXECUTABLES[str(argv[0])]
        command = self._sandbox_prefix() + [executable] + [str(item) for item in argv[1:]]
        effective_timeout = max(1, min(int(timeout_sec or self.timeout_sec), 600))
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, timeout=effective_timeout,
                env={},
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = None
            timed_out = True
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
        else:
            stdout, stderr = completed.stdout, completed.stderr
        return {
            "success": bool(completed is not None and completed.returncode == 0),
            "returncode": None if completed is None else completed.returncode,
            "timed_out": timed_out,
            "stdout": stdout[-self.max_output_chars:],
            "stderr": stderr[-self.max_output_chars:],
            "sandbox": "bubblewrap-evaluator-isolated-v1",
            "workspace": "/workspace",
            "evidence_mounts": {
                name: f"/inputs/{name}" for name in sorted(self.read_roots)
            },
        }


def register_agent_engineering_tools(registry: Any, workspace: AgentEngineeringWorkspace) -> None:
    @registry.tool(
        name="write_engineering_file",
        description=(
            "Write or replace one text file inside the evaluator-isolated engineering "
            "workspace. Use this for algorithms, plugins, tests, notes, and integration code."
        ),
    )
    def write_engineering_file(path: str, content: str):
        return workspace.write(path, content)

    @registry.tool(
        name="read_engineering_file",
        description="Read one text file from the writable engineering workspace.",
    )
    def read_engineering_file(path: str, max_chars: int = 50000):
        return workspace.read(path, max_chars=max_chars)

    @registry.tool(
        name="list_engineering_files",
        description=(
            "List files in the writable workspace or an explicitly mounted sensor-only "
            "evidence scope. Evaluator and benchmark internals are never mounted."
        ),
    )
    def list_engineering_files(scope: str = "workspace", pattern: str = "**/*"):
        return workspace.list_files(scope, pattern)

    @registry.tool(
        name="inspect_experiment_artifact",
        description=(
            "Read a text log, trace, generated program, or manifest from an explicitly "
            "mounted sensor-only experiment evidence root. Binary artifacts return metadata."
        ),
    )
    def inspect_experiment_artifact(
        root_name: str, path: str, max_chars: int = 50000,
    ):
        return workspace.inspect_artifact(root_name, path, max_chars=max_chars)

    @registry.tool(
        name="run_engineering_command",
        description=(
            "Run argv in an evaluator-isolated Bubblewrap workspace with public network "
            "access and no secrets. Supported executables: python, pytest, git, rg, ls, find. "
            "Use python -m pip --target /workspace/.deps for isolated public dependencies."
        ),
    )
    def run_engineering_command(argv: list[str], timeout_sec: int = 120):
        return workspace.run(argv, timeout_sec=timeout_sec)


__all__ = [
    "AgentEngineeringWorkspace", "EngineeringWorkspaceError",
    "register_agent_engineering_tools",
]
