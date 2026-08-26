"""Injectable execution sandboxes for Controller, Tool and workspace commands.

The preferred backend combines rootless Landlock, no_new_privs, seccomp,
resource limits, environment scrubbing and process-group cleanup. Bubblewrap is
a probed optional fallback. If neither backend proves filesystem confinement,
formal profiles fail closed instead of falling back to a subprocess.
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
import selectors
import platform
import shutil
import signal
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
    output_limited: bool = False


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
            input_text: str | None = None, timeout_seconds: float = 120,
            max_output_bytes: int = 1024 * 1024) -> SandboxResult: ...
    def terminate(self, process: subprocess.Popen, grace_seconds: float = 2) -> None: ...


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


_LANDLOCK_SYSCALLS = {
    "x86_64": (444, 445, 446),
    "amd64": (444, 445, 446),
    "aarch64": (444, 445, 446),
    "arm64": (444, 445, 446),
    "riscv64": (444, 445, 446),
}


def _landlock_syscalls() -> tuple[int, int, int] | None:
    return _LANDLOCK_SYSCALLS.get(platform.machine().casefold())


def _landlock_abi() -> int:
    if sys.platform != "linux":
        return 0
    calls = _landlock_syscalls()
    if calls is None:
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(calls[0], 0, 0, 1)
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
        raise OSError(errno.ENOSYS, "Landlock filesystem isolation is unavailable")
    calls = _landlock_syscalls()
    if calls is None:
        raise OSError(errno.ENOSYS, f"Landlock syscall mapping is unavailable for {platform.machine()}")
    libc = ctypes.CDLL(None, use_errno=True)
    handled = _landlock_rights(abi)
    ruleset_attr = _LandlockRulesetAttr(handled)
    ruleset_fd = libc.syscall(calls[0], ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0)
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
                if libc.syscall(calls[1], ruleset_fd, 1, ctypes.byref(path_attr), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule: {path}")
            finally:
                os.close(descriptor)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(calls[2], ruleset_fd, 0) < 0:
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
                    read_write: tuple[Path, ...], *, filesystem: bool = True):
    def apply():
        _limit_resources(timeout_seconds)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if filesystem:
            _apply_landlock(read_only, read_write)
        _apply_seccomp()
    return apply


_PROBE_SCRIPT = r'''
import json,resource,socket,sys
allowed_read,allowed_write,forbidden_read,forbidden_write=sys.argv[1:]
def can_read(path):
    try:
        open(path,"rb").read(1);return True
    except (PermissionError,FileNotFoundError):return False
def can_write(path):
    try:
        open(path,"wb").write(b"x");return True
    except (PermissionError,FileNotFoundError):return False
status=dict(line.split(':',1) for line in open('/proc/self/status') if ':' in line)
try:socket.socket();network_denied=False
except PermissionError:network_denied=True
print(json.dumps({
  "no_new_privs":status.get("NoNewPrivs","").strip()=="1",
  "seccomp":status.get("Seccomp","").strip()=="2",
  "network_denied":network_denied,
  "nofile":resource.getrlimit(resource.RLIMIT_NOFILE)[0],
  "allowed_read":can_read(allowed_read),
  "allowed_write":can_write(allowed_write),
  "unauthorized_read_denied":not can_read(forbidden_read),
  "unauthorized_write_denied":not can_write(forbidden_write),
}))
'''


def _probe_payload(backend: "SandboxBackend", *, bypass_filesystem: bool = False):
    with tempfile.TemporaryDirectory(prefix="roboforge-sandbox-probe-") as root_name:
        root = Path(root_name)
        allowed = root / "allowed"; forbidden = root / "forbidden"
        allowed.mkdir(); forbidden.mkdir()
        allowed_read = allowed / "read.txt"; allowed_read.write_text("allowed")
        allowed_write = allowed / "write.txt"
        forbidden_read = forbidden / "secret.txt"; forbidden_read.write_text("secret")
        forbidden_write = forbidden / "write.txt"
        argv = [sys.executable, "-I", "-c", _PROBE_SCRIPT, str(allowed_read),
                str(allowed_write), str(forbidden_read), str(forbidden_write)]
        if bypass_filesystem:
            process = subprocess.Popen(argv, cwd=allowed,
                env=_clean_environment(None, home=allowed, temporary=allowed),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
                preexec_fn=_hardened_child(20, (), (), filesystem=False))
            stdout, stderr = process.communicate(timeout=20)
            result = SandboxResult(tuple(argv), process.returncode, stdout, stderr,
                                   False, getattr(backend, "name", "probe"))
        else:
            result = backend.run(argv, cwd=allowed, read_only_paths=[allowed_read],
                                 read_write_paths=[allowed], timeout_seconds=20)
        lines = result.stdout.strip().splitlines()
        payload = json.loads(lines[-1]) if lines else {}
        payload["returncode"] = result.returncode
        payload["timed_out"] = result.timed_out
        payload["stderr"] = result.stderr[-1000:]
        return payload


class PosixSandboxBackend:
    """Rootless hardening backend that does not create Linux namespaces."""

    name = "posix-hardened"
    def __init__(self):
        self.landlock_abi = _landlock_abi()
        self._safe = False

    @property
    def safe(self) -> bool:
        return self._safe

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
        if self.landlock_abi < 1:
            raise SandboxUnavailable(
                "posix-hardened requires Landlock filesystem isolation; this kernel does not provide it")
        working = Path(cwd).resolve()
        temporary = Path(temporary_dir or working).resolve()
        read_only = self._paths([*read_only_paths, Path(sys.prefix), "/usr", "/bin",
                                 "/lib", "/lib64", "/etc", "/proc", "/dev"])
        read_write = self._paths([*read_write_paths, temporary])
        process_env = _clean_environment(env, home=working, temporary=temporary)
        return subprocess.Popen(command, cwd=working, env=process_env,
            start_new_session=True, preexec_fn=_hardened_child(timeout_seconds,
                read_only, read_write, filesystem=True), **kwargs)

    def run(self, argv: Sequence[str], *, cwd: str | Path,
            env: Mapping[str, str] | None = None,
            read_only_paths: Sequence[str | Path] = (),
            read_write_paths: Sequence[str | Path] = (),
            input_text: str | None = None, timeout_seconds: float = 120,
            max_output_bytes: int = 1024 * 1024) -> SandboxResult:
        maximum = max(1, int(max_output_bytes))
        with tempfile.TemporaryDirectory(prefix="roboforge-sandbox-") as temp_name:
            temporary = Path(temp_name)
            process = self.popen(argv, cwd=cwd, env=env, read_only_paths=read_only_paths,
                read_write_paths=[*read_write_paths, temporary], timeout_seconds=timeout_seconds,
                temporary_dir=temporary, stdin=subprocess.PIPE if input_text is not None else None,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if process.stdin is not None:
                try:
                    process.stdin.write(str(input_text).encode())
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            streams = {process.stdout.fileno(): ("stdout", process.stdout),
                       process.stderr.fileno(): ("stderr", process.stderr)}
            selector = selectors.DefaultSelector()
            for descriptor in streams:
                selector.register(descriptor, selectors.EVENT_READ)
            chunks = {"stdout": bytearray(), "stderr": bytearray()}
            deadline = time.monotonic() + float(timeout_seconds)
            timed_out = False
            output_limited = False
            try:
                while selector.get_map():
                    remaining_time = deadline - time.monotonic()
                    if remaining_time <= 0:
                        timed_out = True
                        self.terminate(process)
                        break
                    ready = selector.select(remaining_time)
                    if not ready:
                        timed_out = True
                        self.terminate(process)
                        break
                    for key, _mask in ready:
                        name, stream = streams[key.fd]
                        data = os.read(key.fd, 64 * 1024)
                        if not data:
                            selector.unregister(key.fd)
                            stream.close()
                            continue
                        used = len(chunks["stdout"]) + len(chunks["stderr"])
                        available = max(0, maximum - used)
                        chunks[name].extend(data[:available])
                        if len(data) > available:
                            output_limited = True
                            self.terminate(process)
                            break
                    if output_limited:
                        break
            finally:
                selector.close()
                self.terminate(process) if process.poll() is None else None
                process.wait(timeout=3)
            stdout = bytes(chunks["stdout"]).decode("utf-8", errors="replace")
            stderr = bytes(chunks["stderr"]).decode("utf-8", errors="replace")
            return SandboxResult(tuple(str(item) for item in argv), process.returncode,
                stdout, stderr, timed_out, self.name, output_limited)

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
        try:
            payload = _probe_payload(self, bypass_filesystem=self.landlock_abi < 1)
            available = (self.landlock_abi >= 1
                         and not payload.get("timed_out") and payload.get("returncode") == 0
                         and payload.get("no_new_privs") is True
                         and payload.get("seccomp") is True
                         and payload.get("network_denied") is True
                         and payload.get("allowed_read") is True
                         and payload.get("allowed_write") is True
                         and payload.get("unauthorized_read_denied") is True
                         and payload.get("unauthorized_write_denied") is True
                         and int(payload.get("nofile", 0)) <= 256)
            detail = ("rootless Landlock/seccomp/rlimit path confinement succeeded"
                      if available else "filesystem confinement failed: " + json.dumps(payload, sort_keys=True))
        except Exception as exc:
            available = False; detail = f"{type(exc).__name__}: {exc}"
            payload = {}
        self._safe = available
        return SandboxProbe(self.name, available, detail, {
            "requires_root": False, "uses_user_namespace": False,
            "no_new_privs": payload.get("no_new_privs") is True,
            "seccomp": payload.get("seccomp") is True,
            "rlimit": int(payload.get("nofile", 999999)) <= 256,
            "environment_scrub": True, "process_group_cleanup": True,
            "landlock": self.landlock_abi > 0, "landlock_abi": self.landlock_abi,
            "filesystem_isolation": available,
            "unauthorized_read_denied": payload.get("unauthorized_read_denied") is True,
            "unauthorized_write_denied": payload.get("unauthorized_write_denied") is True,
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
            "--unshare-ipc", "--unshare-uts", "--tmpfs", "/", "--proc", "/proc",
            "--dev", "/dev"]
        read_only = [Path(value).resolve() for value in read_only_paths if Path(value).exists()]
        read_only.extend(path for path in (Path("/usr"), Path("/bin"), Path("/lib"),
            Path("/lib64"), Path("/etc"), Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(), Path(sys.executable).resolve(),
            Path(__file__).resolve())
            if path.exists())
        read_write = [Path(value).resolve() for value in read_write_paths if Path(value).exists()]
        if not any(cwd == path or path in cwd.parents for path in read_write):
            read_only.append(cwd)
        bindings: dict[Path, bool] = {}
        for path in read_only:
            bindings.setdefault(path, False)
        for path in read_write:
            bindings[path] = True
        directories = {Path("/tmp")}
        for path in bindings:
            current = path if path.is_dir() else path.parent
            directories.update(current.parents)
            directories.add(current)
        for directory in sorted((item for item in directories if str(item) != "/"),
                                key=lambda item: len(item.parts)):
            command.extend(["--dir", str(directory)])
        command.extend(["--tmpfs", "/tmp"])
        for path, writable in sorted(bindings.items(), key=lambda item: len(item[0].parts)):
            command.extend(["--bind" if writable else "--ro-bind", str(path), str(path)])
        launcher = [sys.executable, str(Path(__file__).resolve()), "--sandbox-exec",
                    str(120), json.dumps([str(item) for item in argv])]
        command.extend(["--chdir", str(cwd), "--", *launcher])
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
            payload = _probe_payload(self)
            available = (payload.get("returncode") == 0
                         and payload.get("no_new_privs") is True
                         and payload.get("seccomp") is True
                         and payload.get("network_denied") is True
                         and payload.get("allowed_read") is True
                         and payload.get("allowed_write") is True
                         and payload.get("unauthorized_read_denied") is True
                         and payload.get("unauthorized_write_denied") is True)
            detail = "bubblewrap path confinement succeeded" if available else json.dumps(payload)
        except Exception as exc:
            available = False; detail = f"{type(exc).__name__}: {exc}"
            payload = {}
        self._safe = available
        return SandboxProbe(self.name, available, detail, {
            "requires_root": False, "uses_user_namespace": True,
            "namespace_filesystem": available, "network_namespace": False,
            "filesystem_isolation": available,
            "unauthorized_read_denied": payload.get("unauthorized_read_denied") is True,
            "unauthorized_write_denied": payload.get("unauthorized_write_denied") is True,
            "no_new_privs": payload.get("no_new_privs") is True,
            "seccomp": payload.get("seccomp") is True})


class UnsafeSandboxBackend(PosixSandboxBackend):
    """Explicit development-only process runner with no syscall isolation."""

    name = "unsafe-dev"

    @property
    def safe(self) -> bool:
        return False

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
    if name == "auto":
        posix = PosixSandboxBackend()
        if posix.probe().available:
            return posix
        bubblewrap = BubblewrapBackend()
        if bubblewrap.probe().available:
            return bubblewrap
        return posix
    if name == "posix":
        return PosixSandboxBackend()
    if name == "bubblewrap":
        return BubblewrapBackend()
    if name == "unsafe":
        return UnsafeSandboxBackend()
    raise ValueError(f"unknown sandbox backend: {name}")


def default_sandbox() -> SandboxBackend:
    return select_sandbox("auto")


def _sandbox_exec_main(argv: Sequence[str]) -> int:
    if len(argv) != 3 or argv[0] != "--sandbox-exec":
        return 64
    timeout_seconds = float(argv[1])
    command = json.loads(argv[2])
    if not isinstance(command, list) or not command:
        return 64
    _limit_resources(timeout_seconds)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    _apply_seccomp()
    os.execvpe(str(command[0]), [str(item) for item in command], dict(os.environ))
    return 70


__all__ = ["SandboxBackend", "SandboxProbe", "SandboxResult", "SandboxUnavailable",
           "PosixSandboxBackend", "BubblewrapBackend", "UnsafeSandboxBackend",
           "default_sandbox", "select_sandbox"]


if __name__ == "__main__":
    raise SystemExit(_sandbox_exec_main(sys.argv[1:]))
