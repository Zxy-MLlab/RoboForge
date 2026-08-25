"""Explicit sandbox backend selection and runtime probing."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Protocol


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxProbe:
    backend: str
    available: bool
    detail: str


class SandboxBackend(Protocol):
    """Execution isolation backend contract used by CLI preflight checks."""

    name: str

    def probe(self) -> SandboxProbe: ...
    def require(self) -> SandboxProbe: ...


class BubblewrapBackend:
    name = "bubblewrap"

    def __init__(self, executable: str | None = None):
        self.executable = executable or shutil.which("bwrap")

    def probe(self) -> SandboxProbe:
        if not self.executable:
            return SandboxProbe(self.name, False, "bwrap executable not found")
        binds = []
        for value in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(value).exists():
                binds.extend(["--ro-bind", value, value])
        command = [self.executable, "--die-with-parent", "--new-session", "--unshare-pid",
                   "--unshare-net", *binds, "--dev", "/dev",
                   "--proc", "/proc", "--tmpfs", "/tmp", "--",
                   "/usr/bin/env", "-i", "/usr/bin/python3", "-c",
                   "import json; print(json.dumps({'sandbox': True}))"]
        try:
            completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=20)
        except Exception as exc:
            return SandboxProbe(self.name, False, f"{type(exc).__name__}: {exc}")
        if completed.returncode != 0:
            return SandboxProbe(self.name, False, completed.stdout[-2000:].strip())
        try:
            value = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception:
            return SandboxProbe(self.name, False, "sandbox returned an invalid receipt")
        return SandboxProbe(self.name, value == {"sandbox": True}, "namespace execution succeeded")

    def require(self) -> SandboxProbe:
        result = self.probe()
        if not result.available:
            raise SandboxUnavailable(f"{result.backend} unavailable: {result.detail}")
        return result


def default_sandbox() -> BubblewrapBackend:
    return BubblewrapBackend()


__all__ = ["SandboxBackend", "BubblewrapBackend", "SandboxProbe", "SandboxUnavailable",
           "default_sandbox"]
