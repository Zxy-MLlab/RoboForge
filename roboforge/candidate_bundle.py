"""Immutable Candidate Bundle snapshots for canonical RoboForge trials."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from .store import canonical_json


class CandidateBundleError(RuntimeError):
    pass


_EXCLUDED_PARTS = {
    ".git", ".roboforge", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".venv", "node_modules", "experiments",
}
_DEPENDENCY_FILES = {
    "pyproject.toml", "uv.lock", "poetry.lock", "Pipfile.lock",
    "requirements.txt", "environment.yml", "environment.yaml",
}
_CRITICAL_PACKAGES = (
    "numpy", "torch", "mujoco", "robosuite", "libero",
    "opencv-python-headless", "scipy", "open3d",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, Any]:
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        repository = Path(top).resolve()
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        patch = subprocess.run(
            ["git", "-C", str(repository), "diff", "--binary", "HEAD", "--"],
            check=True, capture_output=True, timeout=30,
        ).stdout
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain=v1", "-uall"],
            check=True, capture_output=True, timeout=30,
        ).stdout
        return {
            "available": True,
            "commit": commit,
            "dirty": bool(status),
            "dirty_patch_sha256": _sha256_bytes(patch + status),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "dirty_patch_sha256": None,
        }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in _CRITICAL_PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _candidate_files(workspace: Path, excluded_roots: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(workspace.rglob("*")):
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded_roots):
            continue
        relative = path.relative_to(workspace)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise CandidateBundleError(f"candidate bundle forbids symlink: {relative}")
        if path.is_file():
            files.append(path)
    return files


class CandidateBundleStore:
    """Freeze and verify content-addressed candidate source trees."""

    def __init__(self, root: str | Path, *, max_files: int = 10000,
                 max_file_bytes: int = 64 * 1024 * 1024,
                 max_total_bytes: int = 512 * 1024 * 1024):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_files = int(max_files)
        self.max_file_bytes = int(max_file_bytes)
        self.max_total_bytes = int(max_total_bytes)

    def freeze(self, *, workspace: str | Path, entrypoint: str | Path,
               runtime_metadata: Mapping[str, Any] | None = None,
               excluded_roots: tuple[str | Path, ...] = ()) -> dict[str, Any]:
        workspace_path = Path(workspace).resolve()
        entrypoint_path = Path(entrypoint).resolve()
        try:
            entrypoint_relative = entrypoint_path.relative_to(workspace_path)
        except ValueError as exc:
            raise CandidateBundleError("entrypoint must be inside candidate workspace") from exc
        if not entrypoint_path.is_file():
            raise CandidateBundleError("candidate entrypoint is unavailable")

        exclusions = tuple(Path(item).resolve() for item in excluded_roots)
        paths = _candidate_files(workspace_path, exclusions)
        if len(paths) > self.max_files:
            raise CandidateBundleError("candidate bundle exceeds file limit")
        entries = []
        total = 0
        for path in paths:
            size = path.stat().st_size
            if size > self.max_file_bytes:
                raise CandidateBundleError(
                    f"candidate file exceeds byte limit: {path.relative_to(workspace_path)}"
                )
            total += size
            if total > self.max_total_bytes:
                raise CandidateBundleError("candidate bundle exceeds total byte limit")
            entries.append({
                "path": path.relative_to(workspace_path).as_posix(),
                "sha256": _sha256_file(path),
                "size_bytes": size,
                "mode": path.stat().st_mode & 0o777,
            })

        entrypoint_name = entrypoint_relative.as_posix()
        controller = next((item for item in entries if item["path"] == entrypoint_name), None)
        if controller is None:
            raise CandidateBundleError("candidate entrypoint was excluded from snapshot")
        dependency_entries = [
            item for item in entries
            if Path(item["path"]).name in _DEPENDENCY_FILES
            or Path(item["path"]).name.startswith("requirements-")
        ]
        capability_entries = [
            item for item in entries if Path(item["path"]).parts[:1] == ("capabilities",)
        ]
        robot_stack_prefixes = ("robot_sdk", "runtime_adapters", "services", "models")
        robot_stack_entries = [
            item for item in entries
            if Path(item["path"]).parts[:1] in robot_stack_prefixes
        ]
        runtime = dict(runtime_metadata or {})
        packages = _package_versions()
        dependency_payload = {
            "files": dependency_entries,
            "python": sys.version.split()[0],
            "packages": packages,
        }
        dependency_digest = _sha256_bytes(canonical_json(dependency_payload))
        capability_digests = sorted({str(item["sha256"]) for item in capability_entries})
        model_artifact_digests = sorted(
            str(value) for value in dict(runtime.get("model_artifact_digests") or {}).values()
            if isinstance(value, str)
        )
        manifest_body = {
            "schema_version": "roboforge-candidate-bundle-v1",
            "entrypoint": entrypoint_name,
            "files": entries,
            "controller_digest": controller["sha256"],
            "dependency_digest": dependency_digest,
            "dependencies": dependency_payload,
            "capability_digests": capability_digests,
            "editable_robot_stack": {
                "roots": list(robot_stack_prefixes),
                "files": robot_stack_entries,
            },
            "model_artifact_digests": model_artifact_digests,
            "runtime_provider_digest": runtime.get("runtime_provider_digest"),
            "runtime_api_version": runtime.get("runtime_api_version"),
            "capability_versions": dict(runtime.get("capability_versions") or {}),
            "model_services": dict(runtime.get("model_services") or {}),
            "source_control": {
                "workspace": _git_metadata(workspace_path),
                "harness": dict(runtime.get("source_control") or {}),
            },
        }
        digest = _sha256_bytes(canonical_json(manifest_body))
        manifest = {
            **manifest_body,
            "candidate_bundle_id": f"candidate://{digest}",
            "candidate_bundle_digest": digest,
        }
        target = self.root / digest
        if target.exists():
            self.verify(digest)
            existing = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            if existing != manifest:
                raise CandidateBundleError("candidate bundle digest collision")
            return existing

        temporary = Path(tempfile.mkdtemp(prefix=f".{digest}.", dir=self.root))
        try:
            source_root = temporary / "files"
            for item, source in zip(entries, paths):
                destination = source_root / item["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                os.chmod(destination, int(item["mode"]) & 0o555)
            (temporary / "manifest.json").write_bytes(canonical_json(manifest))
            os.chmod(temporary / "manifest.json", 0o444)
            try:
                os.replace(temporary, target)
            except FileExistsError:
                # Another freezer won the same content-addressed race. The
                # winner is acceptable only if it independently verifies as
                # the exact same immutable object.
                self.verify(digest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self.verify(digest)
        return manifest

    def verify(self, digest: str) -> dict[str, Any]:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise CandidateBundleError("invalid candidate bundle digest")
        root = self.root / digest
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CandidateBundleError("candidate bundle manifest is unavailable") from exc
        body = dict(manifest)
        recorded = body.pop("candidate_bundle_digest", None)
        identifier = body.pop("candidate_bundle_id", None)
        if recorded != digest or identifier != f"candidate://{digest}":
            raise CandidateBundleError("candidate bundle identity mismatch")
        if _sha256_bytes(canonical_json(body)) != digest:
            raise CandidateBundleError("candidate bundle manifest digest mismatch")
        expected = {str(item["path"]): item for item in manifest.get("files", [])}
        actual = {
            path.relative_to(root / "files").as_posix(): path
            for path in (root / "files").rglob("*") if path.is_file()
        }
        if set(actual) != set(expected):
            raise CandidateBundleError("candidate bundle file set mismatch")
        for relative, path in actual.items():
            if path.is_symlink() or _sha256_file(path) != expected[relative]["sha256"]:
                raise CandidateBundleError(f"candidate bundle file digest mismatch: {relative}")
        return manifest

    def entrypoint(self, digest: str) -> Path:
        manifest = self.verify(digest)
        return self.root / digest / "files" / str(manifest["entrypoint"])

    def source_root(self, digest: str) -> Path:
        self.verify(digest)
        return self.root / digest / "files"


__all__ = ["CandidateBundleError", "CandidateBundleStore"]
