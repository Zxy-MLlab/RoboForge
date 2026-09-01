from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class CorruptStore(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


class ExperimentStore:
    def __init__(self, root: str | Path, *, max_trials: int, max_diagnostics: int):
        self.root = Path(root).resolve()
        self.evidence_dir = self.root / "evidence"
        self.artifact_dir = self.root / "artifacts" / "immutable"
        self.controller_dir = self.root / "controllers"
        self.private_dir = self.root / "private"
        self.state_path = self.root / "state.json"
        self.lock_path = self.root / ".store.lock"
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.evidence_dir,
            self.artifact_dir,
            self.controller_dir,
            self.private_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            atomic_write(
                self.state_path,
                canonical_json(
                    {
                        "schema_version": 1,
                        "max_trials": max_trials,
                        "max_diagnostics": max_diagnostics,
                        "physical_trials": 0,
                        "diagnostics": 0,
                        "requests": {},
                        "evidence": {},
                        "latest_evidence": None,
                        "latest_diagnostic_evidence": None,
                        "latest_physical_evidence": None,
                    }
                ),
            )

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def load_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptStore("experiment state is unreadable") from exc
        required = {
            "schema_version",
            "max_trials",
            "max_diagnostics",
            "physical_trials",
            "diagnostics",
            "requests",
            "evidence",
        }
        if not isinstance(value, dict) or not required.issubset(value):
            raise CorruptStore("experiment state has an invalid shape")
        if not isinstance(value["requests"], dict) or not isinstance(
            value["evidence"], dict
        ):
            raise CorruptStore("experiment state indexes are invalid")
        value.setdefault("latest_evidence", None)
        value.setdefault("latest_diagnostic_evidence", None)
        value.setdefault("latest_physical_evidence", None)
        return value

    def save_state(self, state: dict[str, Any]) -> None:
        atomic_write(self.state_path, canonical_json(state))

    def put_controller(self, source: bytes) -> str:
        digest = sha256_bytes(source)
        path = self.controller_dir / f"{digest}.py"
        if path.exists():
            if sha256_bytes(path.read_bytes()) != digest:
                raise CorruptStore("immutable Controller snapshot digest mismatch")
        else:
            atomic_write(path, source)
        return digest

    def put_artifact(self, *, name: str, media_type: str, data: bytes) -> dict[str, Any]:
        digest = sha256_bytes(data)
        path = self.artifact_dir / digest
        if path.exists():
            if sha256_bytes(path.read_bytes()) != digest:
                raise CorruptStore("immutable artifact digest mismatch")
        else:
            atomic_write(path, data)
        return {
            "uri": f"artifact://{digest}",
            "sha256": digest,
            "media_type": media_type,
            "name": name,
            "size_bytes": len(data),
        }

    def read_artifact(self, uri: str, expected_sha256: str) -> bytes:
        prefix = "artifact://"
        if not uri.startswith(prefix):
            raise CorruptStore("invalid artifact handle")
        digest = uri.removeprefix(prefix)
        if digest != expected_sha256 or len(digest) != 64:
            raise CorruptStore("artifact handle digest mismatch")
        path = self.artifact_dir / digest
        try:
            value = path.read_bytes()
        except OSError as exc:
            raise CorruptStore("artifact is unavailable") from exc
        if sha256_bytes(value) != digest:
            raise CorruptStore("artifact content digest mismatch")
        return value

    def put_private_receipt(self, request_id: str, receipt: dict[str, Any]) -> None:
        digest = sha256_bytes(request_id.encode("utf-8"))
        atomic_write(self.private_dir / f"{digest}.json", canonical_json(receipt))

    def put_evidence(self, body_without_sha: dict[str, Any]) -> tuple[str, str]:
        digest = sha256_bytes(canonical_json(body_without_sha))
        ref = body_without_sha["ref"]
        safe_name = ref.replace("://", "-").replace("/", "-")
        body = {**body_without_sha, "evidence_sha256": digest}
        path = self.evidence_dir / f"{safe_name}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != body:
                raise CorruptStore("immutable evidence reference collision")
        else:
            atomic_write(path, canonical_json(body))
        return ref, digest

    def load_evidence_file(self, path: Path) -> dict[str, Any]:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptStore("evidence is unreadable") from exc
        digest = body.get("evidence_sha256")
        unsigned = {key: value for key, value in body.items() if key != "evidence_sha256"}
        if not isinstance(digest, str) or sha256_bytes(canonical_json(unsigned)) != digest:
            raise CorruptStore("evidence digest mismatch")
        return body

    def find_evidence_by_request(self, request_id: str) -> dict[str, Any] | None:
        found: list[dict[str, Any]] = []
        for path in sorted(self.evidence_dir.glob("*.json")):
            body = self.load_evidence_file(path)
            if body.get("request_id") == request_id:
                found.append(body)
        if len(found) > 1:
            raise CorruptStore("request id maps to multiple evidence records")
        return found[0] if found else None
