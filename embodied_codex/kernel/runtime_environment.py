"""Immutable, content-bound Python environments for acquired Tools."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid
import venv
from typing import Any, Mapping

from .cas import ContentAddressedStore, ContentAddressedStoreError
from .sandbox import SandboxBackend, default_sandbox


class RuntimeEnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeEnvironment:
    runtime_id: str
    root: Path
    python: Path
    reused: bool


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENV_KEYS = ("python", "dependencies", "accelerator", "platform", "cuda")

_OFFLINE_PIP = r'''
import os,runpy,sys
# The sandbox blocks chmod because Landlock does not mediate this metadata
# operation on every supported ABI. Wheel installation does not need to alter
# modes: the Harness seals the completed environment itself.
os.chmod=lambda *args,**kwargs: None
os.fchmod=lambda *args,**kwargs: None
sys.argv=["pip",*sys.argv[1:]]
runpy.run_module("pip",run_name="__main__")
'''


class RuntimeEnvironmentManager:
    """Build and reuse offline venvs keyed by a canonical runtime spec."""

    def __init__(self, root: str | Path, *, cas: ContentAddressedStore,
                 python: str | Path | None = None,
                 sandbox: SandboxBackend | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cas = cas
        self.base_python = Path(python or sys.executable).resolve()
        self.sandbox = sandbox or default_sandbox()
        self.sandbox.require()
        self.lock_path = self.root / ".runtime.lock"

    @contextmanager
    def _locked(self):
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _current_python() -> dict[str, str]:
        return {"implementation": platform.python_implementation().casefold(),
                "version": platform.python_version(),
                "abi": str(sysconfig.get_config_var("SOABI") or "none")}

    @staticmethod
    def _current_platform() -> dict[str, str]:
        return {"system": platform.system().casefold(),
                "machine": platform.machine().casefold()}

    def default_spec(self) -> dict[str, Any]:
        return {"python": self._current_python(), "dependencies": [],
                "accelerator": "cpu", "platform": self._current_platform()}

    @staticmethod
    def _canonical(value: Mapping[str, Any]) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    @classmethod
    def _runtime_id(cls, value: Mapping[str, Any]) -> str:
        return "sha256-" + hashlib.sha256(cls._canonical(value)).hexdigest()

    def seal(self, value: Mapping[str, Any] | None, *,
             workspace_root: str | Path) -> dict[str, Any]:
        """Validate a model-facing spec and replace local paths with CAS URIs."""
        raw = dict(value or self.default_spec())
        python = dict(raw.get("python") or {})
        if python != self._current_python():
            raise RuntimeEnvironmentError(
                "runtime Python implementation, version, and ABI must exactly match the builder")
        target_platform = dict(raw.get("platform") or {})
        if target_platform != self._current_platform():
            raise RuntimeEnvironmentError("runtime platform must exactly match the current host")
        accelerator = str(raw.get("accelerator") or "cpu").casefold()
        if accelerator not in {"cpu", "cuda"}:
            raise RuntimeEnvironmentError("runtime accelerator must be cpu or cuda")
        cuda = dict(raw.get("cuda") or {})
        if accelerator == "cuda" and not all(str(cuda.get(key) or "")
                                              for key in ("toolkit", "minimum_driver")):
            raise RuntimeEnvironmentError(
                "CUDA runtimes require exact toolkit and minimum_driver compatibility metadata")
        workspace = Path(workspace_root).resolve()
        dependencies = []
        seen: set[str] = set()
        for item in list(raw.get("dependencies") or []):
            dependency = dict(item or {})
            name, version = str(dependency.get("name") or ""), str(dependency.get("version") or "")
            if not _NAME.fullmatch(name):
                raise RuntimeEnvironmentError("runtime dependency name is invalid")
            if not _VERSION.fullmatch(version) or any(char in version for char in "*<>=~, "):
                raise RuntimeEnvironmentError("runtime dependency requires an exact version")
            normalized_name = name.casefold().replace("_", "-")
            if normalized_name in seen:
                raise RuntimeEnvironmentError("runtime dependency names must be unique")
            seen.add(normalized_name)
            artifact = dict(dependency.get("artifact") or {})
            relative = Path(str(artifact.get("path") or ""))
            if not str(relative) or relative.is_absolute() or ".." in relative.parts:
                raise RuntimeEnvironmentError("runtime dependency requires a workspace artifact path")
            source = (workspace / relative).resolve()
            if workspace not in source.parents or not source.is_file() or source.is_symlink():
                raise RuntimeEnvironmentError("runtime dependency artifact is missing")
            filename = str(artifact.get("filename") or source.name)
            if Path(filename).name != filename or filename != source.name:
                raise RuntimeEnvironmentError("runtime dependency artifact filename is invalid")
            kind = str(artifact.get("kind") or "wheel")
            if kind != "wheel":
                raise RuntimeEnvironmentError("Runtime Environment v1 supports wheel artifacts only")
            if not filename.endswith(".whl"):
                raise RuntimeEnvironmentError("wheel artifact must use a .whl filename")
            expected = str(artifact.get("sha256") or "").casefold()
            if not _SHA256.fullmatch(expected):
                raise RuntimeEnvironmentError("runtime dependency artifact requires SHA256")
            try:
                stored = self.cas.put(source, expected_sha256=expected)
            except ContentAddressedStoreError as exc:
                raise RuntimeEnvironmentError(
                    "runtime dependency artifact checksum mismatch") from exc
            dependencies.append({"name": name, "version": version,
                "artifact": {"filename": filename, "kind": kind,
                             "sha256": expected, "blob_uri": stored["blob_uri"]}})
        dependencies.sort(key=lambda item: (item["name"].casefold(), item["version"]))
        sealed: dict[str, Any] = {"python": python, "dependencies": dependencies,
            "accelerator": accelerator, "platform": target_platform}
        if cuda:
            sealed["cuda"] = {key: str(cuda[key]) for key in sorted(cuda)}
        if accelerator == "cuda":
            self._require_host_cuda(sealed["cuda"])
        sealed["runtime_id"] = self._runtime_id(sealed)
        return sealed

    @staticmethod
    def _version_tuple(value: str) -> tuple[int, ...]:
        match = re.match(r"^\s*(\d+(?:\.\d+)*)", str(value))
        if not match:
            raise RuntimeEnvironmentError("CUDA driver version is invalid")
        return tuple(int(item) for item in match.group(1).split("."))

    def _require_host_cuda(self, value: Mapping[str, Any]) -> None:
        """Accept only CUDA wheels compatible with the currently visible driver."""
        try:
            completed = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                "--format=csv,noheader"], text=True, capture_output=True, timeout=10,
                check=False, env={"PATH": os.environ.get("PATH", os.defpath)})
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeEnvironmentError(
                "CUDA runtime is unsupported because the host driver is unavailable; "
                "use an Adapter-owned deployment capability") from exc
        versions = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.returncode != 0 or not versions:
            raise RuntimeEnvironmentError(
                "CUDA runtime is unsupported because the host driver is unavailable; "
                "use an Adapter-owned deployment capability")
        required = self._version_tuple(str(value["minimum_driver"]))
        if any(self._version_tuple(actual) < required for actual in versions):
            raise RuntimeEnvironmentError(
                "CUDA runtime requires a newer host driver; use an Adapter-owned deployment capability")

    def _validated_sealed(self, value: Mapping[str, Any]) -> dict[str, Any]:
        raw = {key: value.get(key) for key in _ENV_KEYS if key in value}
        if dict(raw.get("python") or {}) != self._current_python():
            raise RuntimeEnvironmentError("runtime Python or ABI is unavailable")
        if dict(raw.get("platform") or {}) != self._current_platform():
            raise RuntimeEnvironmentError("runtime platform is unavailable")
        if str(raw.get("accelerator") or "") not in {"cpu", "cuda"}:
            raise RuntimeEnvironmentError("runtime accelerator is invalid")
        if raw.get("accelerator") == "cuda":
            cuda = dict(raw.get("cuda") or {})
            if not all(str(cuda.get(key) or "") for key in ("toolkit", "minimum_driver")):
                raise RuntimeEnvironmentError("sealed CUDA runtime metadata is incomplete")
            self._require_host_cuda(cuda)
        dependencies = []
        seen: set[str] = set()
        for item in list(raw.get("dependencies") or []):
            dependency = dict(item or {})
            name, version = str(dependency.get("name") or ""), str(dependency.get("version") or "")
            artifact = dict(dependency.get("artifact") or {})
            if (not _NAME.fullmatch(name) or not _VERSION.fullmatch(version)
                    or any(char in version for char in "*<>=~, ")
                    or artifact.get("kind") != "wheel"
                    or not _SHA256.fullmatch(str(artifact.get("sha256") or ""))
                    or not str(artifact.get("blob_uri") or "").startswith(self.cas.prefix)
                    or Path(str(artifact.get("filename") or "")).name
                       != str(artifact.get("filename") or "")
                    or not str(artifact.get("filename") or "").endswith(".whl")):
                raise RuntimeEnvironmentError("sealed runtime dependency is invalid")
            normalized_name = name.casefold().replace("_", "-")
            if normalized_name in seen:
                raise RuntimeEnvironmentError("sealed runtime dependency names must be unique")
            seen.add(normalized_name)
            try:
                self.cas.resolve(str(artifact["blob_uri"]), verify=True)
            except ContentAddressedStoreError as exc:
                raise RuntimeEnvironmentError("runtime dependency CAS artifact is invalid") from exc
            dependencies.append({"name": name, "version": version,
                                 "artifact": artifact})
        raw["dependencies"] = sorted(dependencies,
            key=lambda item: (item["name"].casefold(), item["version"]))
        runtime_id = self._runtime_id(raw)
        if str(value.get("runtime_id") or "") != runtime_id:
            raise RuntimeEnvironmentError("runtime_id does not match the runtime spec")
        return {**raw, "runtime_id": runtime_id}

    def _environment(self, runtime_id: str, *, reused: bool) -> RuntimeEnvironment:
        root = self.root / runtime_id
        python = root / "bin" / "python"
        return RuntimeEnvironment(runtime_id, root, python, reused)

    def _valid_existing(self, spec: Mapping[str, Any]) -> bool:
        environment = self._environment(str(spec["runtime_id"]), reused=True)
        receipt = environment.root / "runtime.json"
        try:
            payload = json.loads(receipt.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        return (environment.python.is_file()
                and payload.get("protocol") == "roboforge-runtime-v1"
                and payload.get("runtime_id") == spec["runtime_id"]
                and payload.get("runtime_spec") == dict(spec))

    def ensure(self, value: Mapping[str, Any]) -> RuntimeEnvironment:
        spec = self._validated_sealed(value)
        with self._locked():
            if self._valid_existing(spec):
                return self._environment(str(spec["runtime_id"]), reused=True)
            target = self.root / str(spec["runtime_id"])
            if target.exists():
                raise RuntimeEnvironmentError("runtime directory exists without a valid receipt")
            staging = Path(tempfile.mkdtemp(prefix=f".{spec['runtime_id']}.staging-",
                                             dir=self.root))
            try:
                dependencies = list(spec.get("dependencies") or [])
                venv.EnvBuilder(with_pip=bool(dependencies), symlinks=False).create(staging)
                python = staging / "bin" / "python"
                if dependencies:
                    wheelhouse = staging / "wheelhouse"
                    wheelhouse.mkdir()
                    requirements = []
                    for dependency in dependencies:
                        artifact = dependency["artifact"]
                        self.cas.materialize(str(artifact["blob_uri"]),
                            wheelhouse / str(artifact["filename"]), verify=True)
                        requirements.append(f"{dependency['name']}=={dependency['version']} "
                            f"--hash=sha256:{artifact['sha256']}")
                    requirement_path = staging / "requirements.txt"
                    requirement_path.write_text("\n".join(requirements) + "\n")
                    command = [str(python), "-I", "-c", _OFFLINE_PIP,
                        "install", "--no-index",
                        "--no-deps", "--require-hashes", "--no-cache-dir",
                        "--no-build-isolation", "--no-compile",
                        "--find-links", str(wheelhouse),
                        "-r", str(requirement_path)]
                    completed = self.sandbox.run(command, cwd=staging,
                        read_write_paths=[staging], timeout_seconds=600)
                    if completed.timed_out or completed.returncode != 0:
                        raise RuntimeEnvironmentError(
                            "offline runtime dependency installation failed: "
                            + (completed.stdout + completed.stderr)[-2000:])
                    shutil.rmtree(wheelhouse)
                    requirement_path.unlink()
                receipt = {"protocol": "roboforge-runtime-v1",
                           "runtime_id": spec["runtime_id"],
                           "runtime_spec": spec}
                (staging / "runtime.json").write_text(
                    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
                for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts),
                                   reverse=True):
                    mode = path.stat(follow_symlinks=False).st_mode
                    if path.is_dir():
                        path.chmod(0o555)
                    elif path.is_file():
                        path.chmod(0o555 if mode & 0o111 else 0o444)
                staging.chmod(0o555)
                os.replace(staging, target)
                descriptor = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except Exception:
                try:
                    staging.chmod(0o700)
                    for path in staging.rglob("*"):
                        try:
                            path.chmod(0o700 if path.is_dir() else 0o600)
                        except OSError:
                            pass
                except OSError:
                    pass
                shutil.rmtree(staging, ignore_errors=True)
                raise
            return self._environment(str(spec["runtime_id"]), reused=False)


__all__ = ["RuntimeEnvironment", "RuntimeEnvironmentError",
           "RuntimeEnvironmentManager"]
