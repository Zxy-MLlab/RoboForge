"""Persistent coding workspace exposed to the engineering model."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


class WorkspaceError(RuntimeError): pass


class TaskWorkspace:
    def __init__(self, root: str | Path, *, require_sandbox: bool=True) -> None:
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.require_sandbox=bool(require_sandbox)
        self.bwrap=shutil.which("bwrap")
        if self.require_sandbox and not self.bwrap:
            raise WorkspaceError("bubblewrap is required for the engineering workspace")

    @staticmethod
    def _system_binds():
        args=[]
        for value in ("/usr","/bin","/lib","/lib64","/etc"):
            if Path(value).exists():args.extend(["--ro-bind",value,value])
        return args

    @staticmethod
    def _safe_environment():
        allowed=("LANG","LC_ALL","LC_CTYPE","TERM","CUDA_VISIBLE_DEVICES",
                 "NVIDIA_VISIBLE_DEVICES","CUDA_HOME","LD_LIBRARY_PATH","MUJOCO_GL")
        result={key:os.environ[key] for key in allowed if key in os.environ}
        result["PYTHONNOUSERSITE"]="1"
        return result

    def _sandbox_command(self,argv,process_env,working_directory=None):
        if not self.bwrap:return list(argv),process_env
        prefix=Path(sys.prefix).resolve();runtime="/runtime"
        command=list(argv)
        executable=Path(command[0]) if Path(command[0]).is_absolute() else None
        if executable is not None and (executable==prefix or prefix in executable.parents):
            command[0]=str(Path(runtime)/executable.relative_to(prefix))
        sandbox_env=dict(process_env)
        sandbox_env["PATH"]=runtime+"/bin:/usr/local/bin:/usr/bin:/bin"
        sandbox_env["HOME"]="/workspace";sandbox_env["TMPDIR"]="/tmp"
        relative = Path(working_directory or self.root).resolve().relative_to(self.root)
        sandbox_cwd = "/workspace" if str(relative) == "." else "/workspace/" + relative.as_posix()
        wrapped=[self.bwrap,"--die-with-parent","--new-session","--unshare-pid",
                 "--unshare-ipc","--unshare-uts","--unshare-net",*self._system_binds(),
                 "--ro-bind",str(prefix),runtime,"--dev","/dev","--proc","/proc",
                 "--tmpfs","/tmp","--bind",str(self.root),"/workspace",
                 "--chdir",sandbox_cwd,"--",*command]
        return wrapped,sandbox_env

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

    def _engineering_snapshot(self) -> dict[str, tuple[int,int]]:
        """Cheap mutation fingerprint for command-driven workspace edits."""
        result={}
        ignored_suffixes={".md",".txt",".log",".pyc"}
        for path in self.root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() in ignored_suffixes:
                continue
            stat=path.stat()
            result[str(path.relative_to(self.root))]=(stat.st_size,stat.st_mtime_ns)
        return result

    def read_file(self, path: str, start_line: int = 1, end_line: int = 400) -> dict[str, Any]:
        target = self._path(path)
        if not target.exists():
            return {"path":path,"exists":False,"start_line":1,"end_line":0,
                    "total_lines":0,"content":""}
        if not target.is_file():raise WorkspaceError("workspace path is not a file")
        lines = target.read_text().splitlines()
        start=max(1,int(start_line));requested_end=max(start,int(end_line))
        end=min(requested_end,start+199,len(lines))
        return {"path":path,"exists":True,"start_line":start,"end_line":end,
                "total_lines":len(lines),"content":"\n".join(lines[start-1:end]),
                "content_truncated":end<min(requested_end,len(lines)),
                "next_start_line":end+1 if end<len(lines) else None}

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

    def replace_file_lines(
        self, path: str, start_line: int, end_line: int, new_content: str,
        expected_old_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Replace an already-inspected inclusive line range atomically."""
        target=self._path(path)
        if not target.is_file():raise WorkspaceError("workspace path is not a file")
        text=target.read_text();lines=text.splitlines(keepends=True)
        start=int(start_line);end=int(end_line)
        if start<1 or end<start or end>len(lines):
            raise WorkspaceError(
                f"invalid inclusive line range {start}:{end} for {len(lines)} lines")
        old="".join(lines[start-1:end])
        old_sha256=hashlib.sha256(old.encode()).hexdigest()
        if expected_old_sha256 is not None and str(expected_old_sha256)!=old_sha256:
            raise WorkspaceError("line range changed since inspection")
        replacement=str(new_content)
        if replacement and end<len(lines) and not replacement.endswith("\n"):
            replacement+="\n"
        updated="".join(lines[:start-1])+replacement+"".join(lines[end:])
        target.write_text(updated)
        return {"path":path,"start_line":start,"end_line":end,
                "replaced_line_count":end-start+1,
                "old_sha256":old_sha256,"bytes":target.stat().st_size}

    def run_command(
        self, argv: list[str], *, timeout_seconds: float = 120,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> dict[str, Any]:
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise WorkspaceError("argv must be a nonempty string list")
        timeout = min(max(float(timeout_seconds), 0.1), 600.0)
        # Public downloads are allowed, host credentials are not.
        process_env = self._safe_environment()
        for key, value in dict(env or {}).items():
            if not key.startswith(("PYTHON", "CUDA", "MUJOCO", "HF_", "TRANSFORMERS_")):
                raise WorkspaceError(f"environment key not allowed: {key}")
            process_env[str(key)] = str(value)
        working_directory = self.root if cwd is None else Path(cwd).resolve()
        if working_directory != self.root and self.root not in working_directory.parents:
            raise WorkspaceError("command cwd escapes task workspace")
        before=self._engineering_snapshot()
        try:
            command,process_env=self._sandbox_command(argv,process_env,working_directory)
            result = subprocess.run(
                command, cwd=working_directory, env=process_env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            )
            receipt={"argv": argv, "exit_code": result.returncode,
                     "output": result.stdout[-30000:],
                     "sandbox":"bubblewrap-workspace-v1" if self.bwrap else "none"}
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            receipt={"argv": argv, "exit_code": None, "timed_out": True,
                     "output": output[-30000:]}
        after=self._engineering_snapshot()
        changed=sorted(key for key in set(before)|set(after)
                       if before.get(key)!=after.get(key))
        if changed:
            receipt["workspace_mutated_paths"]=changed[:100]
            receipt["_embodied_codex_engineering_progress"]=True
            receipt["_embodied_codex_controller_mutated"]="controller.py" in changed
        return receipt

__all__ = ["TaskWorkspace", "WorkspaceError"]
