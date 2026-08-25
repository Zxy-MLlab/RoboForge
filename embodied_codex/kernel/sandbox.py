"""Injectable execution sandboxes for Controller, Tool and workspace commands.

The default backend needs no root privileges or user namespaces.  It combines
no_new_privs, a libseccomp deny-list, resource limits, a scrubbed environment,
private temporary directories and process-group cleanup.  Landlock filesystem
rules are added when the running kernel implements them.  Bubblewrap remains an
explicit, probed enhancement rather than a startup requirement.
"""
from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass, field
import errno
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Protocol, Sequence


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxProbe:
    backend: str
    available: bool
    detail: str
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    backend: str


class SandboxBackend(Protocol):
    """One execution contract shared by all untrusted subprocess surfaces."""

    name: str
    safe: bool

    def probe(self) -> SandboxProbe: ...
    def require(self) -> SandboxProbe: ...
    def popen(self, argv: Sequence[str], *, cwd: str | Path,
              env: Mapping[str, str] | None = None,
              read_only_paths: Sequence[str | Path] = (),
              read_write_paths: Sequence[str | Path] = (), **kwargs) -> subprocess.Popen: ...
    def run(self, argv: Sequence[str], *, cwd: str | Path,
            env: Mapping[str, str] | None = None,
            read_only_paths: Sequence[str | Path] = (),
            read_write_paths: Sequence[str | Path] = (),
            input_text: str | None = None, timeout_seconds: float = 120) -> SandboxResult: ...
    def terminate(self, process: subprocess.Popen, grace_seconds: float = 2) -> None: ...


class ReadOnlyGuard:
    """Temporarily remove write permission from canonical state trees.

    Sandboxed children cannot undo these modes because every safe backend denies
    chmod-family syscalls.  The Harness restores the exact original modes after
    the process exits or while the parent performs one Adapter RPC.
    """

    def __init__(self, roots: Sequence[str | Path], *, exclude: Sequence[str | Path] = ()):
        self.roots = tuple(dict.fromkeys(Path(value).resolve() for value in roots
                                        if Path(value).exists()))
        self.exclude = tuple(Path(value).resolve() for value in exclude)
        self._modes: list[tuple[Path, int]] = []

    def _excluded(self, path: Path) -> bool:
        return any(path == value or value in path.parents for value in self.exclude)

    def protect(self) -> None:
        if self._modes:
            return
        paths = []
        for root in self.roots:
            paths.append(root)
            if root.is_dir(): paths.extend(root.rglob("*"))
        for path in dict.fromkeys(paths):
            if self._excluded(path) or path.is_symlink(): continue
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
                self._modes.append((path, mode))
                path.chmod(0o500 if path.is_dir() else 0o400)
            except FileNotFoundError:
                continue

    def restore(self) -> None:
        modes, self._modes = self._modes, []
        for path, mode in reversed(modes):
            try: path.chmod(mode)
            except FileNotFoundError: pass

    def __enter__(self):
        self.protect(); return self

    def __exit__(self, _type, _value, _traceback):
        self.restore()


_ENV_EXACT = {
    "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "HF_HOME", "LANG", "LC_ALL", "LC_CTYPE",
    "LD_LIBRARY_PATH", "MUJOCO_GL", "NVIDIA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
    "PATH", "PYTHONHASHSEED", "PYTHONNOUSERSITE", "PYTHONPATH", "TOKENIZERS_PARALLELISM",
}
_ENV_PREFIXES = ("ROBOFORGE_TOOL_",)


def _clean_environment(value: Mapping[str, str] | None, *, home: Path,
                       temporary: Path) -> dict[str, str]:
    supplied = dict(value or {})
    invalid = [key for key in supplied if key not in _ENV_EXACT
               and not key.startswith(_ENV_PREFIXES)]
    if invalid:
        raise SandboxUnavailable(f"sandbox environment keys are not allowed: {sorted(invalid)}")
    result = {
        "HOME": str(home),
        "LANG": str(supplied.pop("LANG", os.environ.get("LANG", "C.UTF-8"))),
        "PATH": str(supplied.pop("PATH", f"{Path(sys.executable).resolve().parent}:{os.defpath}")),
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": str(temporary),
    }
    result.update({str(key): str(item) for key, item in supplied.items()})
    return result


def _landlock_abi() -> int:
    if sys.platform != "linux":
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(444, 0, 0, 1)
    return int(result) if result >= 1 else 0


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14
_LL_READ = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR


def _landlock_rights(abi: int) -> int:
    rights = (1 << 13) - 1
    if abi >= 2:
        rights |= _LL_REFER
    if abi >= 3:
        rights |= _LL_TRUNCATE
    return rights


def _apply_landlock(read_only: Sequence[Path], read_write: Sequence[Path]) -> None:
    abi = _landlock_abi()
    if not abi:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    handled = _landlock_rights(abi)
    ruleset_attr = _LandlockRulesetAttr(handled)
    ruleset_fd = libc.syscall(444, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0)
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        entries: dict[Path, int] = {}
        for path in read_only:
            if path.exists():
                entries[path.resolve()] = entries.get(path.resolve(), 0) | _LL_READ
        for path in read_write:
            if path.exists():
                entries[path.resolve()] = handled
        for path, rights in entries.items():
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                path_attr = _LandlockPathBeneathAttr(rights, descriptor)
                if libc.syscall(445, ruleset_fd, 1, ctypes.byref(path_attr), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule: {path}")
            finally:
                os.close(descriptor)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(446, ruleset_fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(ruleset_fd)


_BLOCKED_SYSCALLS = (
    "add_key", "bpf", "chmod", "connect", "delete_module", "fchmod", "fchmodat",
    "fchmodat2", "finit_module", "init_module",
    "io_uring_setup", "keyctl", "kexec_file_load", "kexec_load", "listen", "mount",
    "name_to_handle_at", "open_by_handle_at", "perf_event_open", "pivot_root", "ptrace",
    "reboot", "request_key", "setns", "socket", "socketpair", "swapoff", "swapon",
    "umount2", "unshare", "userfaultfd",
)
_LIBSECCOMP = ctypes.util.find_library("seccomp") or "libseccomp.so.2"


def _apply_seccomp() -> None:
    seccomp = ctypes.CDLL(_LIBSECCOMP, use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                         ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise OSError(errno.ENOMEM, "seccomp_init")
    try:
        deny = 0x00050000 | errno.EPERM
        for name in _BLOCKED_SYSCALLS:
            number = seccomp.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and seccomp.seccomp_rule_add(context, deny, number, 0) != 0:
                raise OSError(errno.EINVAL, f"seccomp_rule_add: {name}")
        if seccomp.seccomp_load(context) != 0:
            raise OSError(ctypes.get_errno() or errno.EINVAL, "seccomp_load")
    finally:
        seccomp.seccomp_release(context)


def _limit_resources(timeout_seconds: float) -> None:
    cpu = max(1, min(int(math.ceil(timeout_seconds)) + 1, 601))
    limits = {
        resource.RLIMIT_CORE: (0, 0),
        resource.RLIMIT_CPU: (cpu, cpu),
        resource.RLIMIT_FSIZE: (512 * 1024 * 1024, 512 * 1024 * 1024),
        resource.RLIMIT_NOFILE: (256, 256),
        # RLIMIT_NPROC is per real UID and includes threads from unrelated jobs.
        # Shared research servers can already have thousands, so keep a finite
        # ceiling without setting it below the server's normal baseline.
        resource.RLIMIT_NPROC: (16384, 16384),
    }
    for kind, requested in limits.items():
        current_soft, current_hard = resource.getrlimit(kind)
        hard = requested[1] if current_hard == resource.RLIM_INFINITY else min(requested[1], current_hard)
        soft = min(requested[0], hard)
        resource.setrlimit(kind, (soft, hard))


def _hardened_child(timeout_seconds: float, read_only: tuple[Path, ...],
                    read_write: tuple[Path, ...]):
    def apply():
        _limit_resources(timeout_seconds)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        _apply_landlock(read_only, read_write)
        _apply_seccomp()
    return apply


class PosixSandboxBackend:
    """Rootless hardening backend that does not create Linux namespaces."""

    name = "posix-hardened"
    safe = True

    def __init__(self):
        self.landlock_abi = _landlock_abi()

    @staticmethod
    def _paths(values: Sequence[str | Path]) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(Path(value).resolve() for value in values if Path(value).exists()))

    def popen(self, argv: Sequence[str], *, cwd: str | Path,
              env: Mapping[str, str] | None = None,
              read_only_paths: Sequence[str | Path] = (),
              read_write_paths: Sequence[str | Path] = (),
              timeout_seconds: float = 120, temporary_dir: str | Path | None = None,
              **kwargs) -> subprocess.Popen:
        command = [str(item) for item in argv]
        if not command:
            raise SandboxUnavailable("sandbox command is empty")
        working = Path(cwd).resolve()
        temporary = Path(temporary_dir or working).resolve()
        read_only = self._paths([*read_only_paths, Path(sys.prefix), "/usr", "/bin",
                                 "/lib", "/lib64", "/etc", "/proc", "/dev"])
        read_write = self._paths([*read_write_paths, temporary])
        process_env = _clean_environment(env, home=working, temporary=temporary)
        return subprocess.Popen(command, cwd=working, env=process_env,
            start_new_session=True, preexec_fn=_hardened_child(timeout_seconds,
                read_only, read_write), **kwargs)

    def run(self, argv: Sequence[str], *, cwd: str | Path,
            env: Mapping[str, str] | None = None,
            read_only_paths: Sequence[str | Path] = (),
            read_write_paths: Sequence[str | Path] = (),
            input_text: str | None = None, timeout_seconds: float = 120) -> SandboxResult:
        with tempfile.TemporaryDirectory(prefix="roboforge-sandbox-") as temp_name:
            temporary = Path(temp_name)
            process = self.popen(argv, cwd=cwd, env=env, read_only_paths=read_only_paths,
                read_write_paths=[*read_write_paths, temporary], timeout_seconds=timeout_seconds,
                temporary_dir=temporary, stdin=subprocess.PIPE if input_text is not None else None,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                timed_out = True
                self.terminate(process)
                tail_out, tail_err = process.communicate()
                stdout += tail_out or ""; stderr += tail_err or ""
            return SandboxResult(tuple(str(item) for item in argv), process.returncode,
                stdout, stderr, timed_out, self.name)

    def terminate(self, process: subprocess.Popen, grace_seconds: float = 2) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + max(0.0, grace_seconds)
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if process.poll() is None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass

    def probe(self) -> SandboxProbe:
        script = (
            "import json,resource,socket\n"
            "status=dict(line.split(':',1) for line in open('/proc/self/status') "
            "if ':' in line)\n"
            "try:\n socket.socket(); network=False\n"
            "except PermissionError:\n network=True\n"
            "print(json.dumps({'no_new_privs':status.get('NoNewPrivs','').strip()=='1',"
            "'seccomp':status.get('Seccomp','').strip()=='2','network_denied':network,"
            "'nofile':resource.getrlimit(resource.RLIMIT_NOFILE)[0]}))\n")
        try:
            result = self.run([sys.executable, "-I", "-c", script], cwd=Path.cwd(),
                              timeout_seconds=20)
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            available = (not result.timed_out and result.returncode == 0
                         and payload.get("no_new_privs") is True
                         and payload.get("seccomp") is True
                         and payload.get("network_denied") is True
                         and int(payload.get("nofile", 0)) <= 256)
            detail = "rootless no_new_privs/seccomp/rlimit execution succeeded" if available \
                else f"hardening receipt failed: {payload} {result.stderr[-1000:]}"
        except Exception as exc:
            available = False; detail = f"{type(exc).__name__}: {exc}"
        return SandboxProbe(self.name, available, detail, {
            "requires_root": False, "uses_user_namespace": False,
            "no_new_privs": available, "seccomp": available, "rlimit": available,
            "environment_scrub": available, "process_group_cleanup": available,
            "landlock": self.landlock_abi > 0, "landlock_abi": self.landlock_abi,
        })

    def require(self) -> SandboxProbe:
        result = self.probe()
        if not result.available:
            raise SandboxUnavailable(f"{result.backend} unavailable: {result.detail}")
        return result


class BubblewrapBackend(PosixSandboxBackend):
    """Optional namespace enhancement, used only after a successful probe."""

    name = "bubblewrap"

    def __init__(self, executable: str | None = None):
        super().__init__()
        self.executable = executable or shutil.which("bwrap")

    def _wrapped(self, argv: Sequence[str], cwd: Path,
                 read_only_paths: Sequence[str | Path],
                 read_write_paths: Sequence[str | Path]):
        if not self.executable:
            raise SandboxUnavailable("bwrap executable not found")
        command = [self.executable, "--die-with-parent", "--new-session", "--unshare-pid",
            "--unshare-ipc", "--unshare-uts", "--unshare-net",
            "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev"]
        required = [cwd, *[Path(value).resolve() for value in read_only_paths],
                    *[Path(value).resolve() for value in read_write_paths]]
        required.extend(Path(str(value)).resolve() for value in argv
                        if Path(str(value)).is_absolute())
        host_tmp_required = any(path == Path("/tmp") or Path("/tmp") in path.parents
                                for path in required)
        if not host_tmp_required:
            command.extend(["--tmpfs", "/tmp"])
        for value in read_write_paths:
            path = Path(value).resolve()
            if path.exists():
                command.extend(["--bind", str(path), str(path)])
        command.extend(["--chdir", str(cwd), "--", *[str(item) for item in argv]])
        return command

    def popen(self, argv: Sequence[str], *, cwd: str | Path,
              env: Mapping[str, str] | None = None,
              read_only_paths: Sequence[str | Path] = (),
              read_write_paths: Sequence[str | Path] = (),
              timeout_seconds: float = 120,
              temporary_dir: str | Path | None = None, **kwargs) -> subprocess.Popen:
        working = Path(cwd).resolve()
        temporary = Path(temporary_dir or working).resolve()
        return subprocess.Popen(self._wrapped(argv, working, read_only_paths,
            read_write_paths), cwd=working,
            env=_clean_environment(env, home=working, temporary=temporary),
            start_new_session=True, **kwargs)

    def probe(self) -> SandboxProbe:
        if not self.executable:
            return SandboxProbe(self.name, False, "bwrap executable not found", {
                "requires_root": False, "uses_user_namespace": True})
        try:
            result = self.run([sys.executable, "-I", "-c",
                "import json;print(json.dumps({'sandbox':True}))"], cwd=Path.cwd(),
                timeout_seconds=20)
            lines = result.stdout.strip().splitlines()
            value = json.loads(lines[-1]) if lines else None
            available = result.returncode == 0 and value == {"sandbox": True}
            detail = "namespace execution succeeded" if available else (
                result.stderr or result.stdout)[-2000:].strip()
        except Exception as exc:
            available = False; detail = f"{type(exc).__name__}: {exc}"
        return SandboxProbe(self.name, available, detail, {
            "requires_root": False, "uses_user_namespace": True,
            "namespace_filesystem": available, "network_namespace": available})


class UnsafeSandboxBackend(PosixSandboxBackend):
    """Explicit development-only process runner with no syscall isolation."""

    name = "unsafe-dev"
    safe = False

    def popen(self, argv: Sequence[str], *, cwd: str | Path,
              env: Mapping[str, str] | None = None,
              read_only_paths: Sequence[str | Path] = (),
              read_write_paths: Sequence[str | Path] = (),
              timeout_seconds: float = 120, temporary_dir: str | Path | None = None,
              **kwargs) -> subprocess.Popen:
        working = Path(cwd).resolve(); temporary = Path(temporary_dir or working).resolve()
        return subprocess.Popen([str(item) for item in argv], cwd=working,
            env=_clean_environment(env, home=working, temporary=temporary),
            start_new_session=True, **kwargs)

    def probe(self) -> SandboxProbe:
        return SandboxProbe(self.name, True, "explicit unsafe development backend", {
            "safe": False, "requires_root": False, "uses_user_namespace": False})


def select_sandbox(name: str = "posix") -> SandboxBackend:
    if name in {"auto", "posix"}:
        return PosixSandboxBackend()
    if name == "bubblewrap":
        return BubblewrapBackend()
    if name == "unsafe":
        return UnsafeSandboxBackend()
    raise ValueError(f"unknown sandbox backend: {name}")


def default_sandbox() -> PosixSandboxBackend:
    return PosixSandboxBackend()


__all__ = ["SandboxBackend", "SandboxProbe", "SandboxResult", "SandboxUnavailable",
           "ReadOnlyGuard",
           "PosixSandboxBackend", "BubblewrapBackend", "UnsafeSandboxBackend",
           "default_sandbox", "select_sandbox"]
