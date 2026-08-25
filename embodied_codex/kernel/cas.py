"""Process-safe content-addressed blob storage for persistent Harness assets."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid


class ContentAddressedStoreError(RuntimeError):
    pass


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _copy_sparse(source: Path, destination: Path) -> None:
    """Create a reflink when supported and otherwise preserve sparse extents."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(source, os.O_RDONLY)
    destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o600)
    cloned = False
    try:
        try:
            fcntl.ioctl(destination_fd, 0x40049409, source_fd)  # FICLONE
            cloned = True
        except OSError as exc:
            if exc.errno not in {errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV,
                                 errno.EINVAL, errno.ENOSYS}:
                raise
        size = os.fstat(source_fd).st_size
        if not cloned:
            os.ftruncate(destination_fd, size)
        position = size if cloned else 0
        while position < size:
            try:
                data_offset = os.lseek(source_fd, position, os.SEEK_DATA)
            except OSError as exc:
                if exc.errno == errno.ENXIO:
                    break
                if exc.errno == errno.EINVAL:
                    os.lseek(source_fd, 0, os.SEEK_SET)
                    os.lseek(destination_fd, 0, os.SEEK_SET)
                    while True:
                        chunk = os.read(source_fd, 4 * 1024 * 1024)
                        if not chunk:
                            break
                        os.write(destination_fd, chunk)
                    break
                raise
            hole_offset = os.lseek(source_fd, data_offset, os.SEEK_HOLE)
            os.lseek(source_fd, data_offset, os.SEEK_SET)
            os.lseek(destination_fd, data_offset, os.SEEK_SET)
            remaining = hole_offset - data_offset
            while remaining:
                chunk = os.read(source_fd, min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise ContentAddressedStoreError(
                        "source changed while it was copied into the CAS")
                os.write(destination_fd, chunk)
                remaining -= len(chunk)
            position = hole_offset
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)
        os.close(source_fd)


class ContentAddressedStore:
    """Immutable SHA-256 blobs shared by all versions under one asset root."""

    prefix = "cas://sha256/"

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.blob_root = self.root / "blobs"
        self.blob_root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / ".cas.lock"
        self.index_path = self.root / "fingerprints.json"

    @contextmanager
    def _locked(self):
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _path(self, digest: str) -> Path:
        digest = str(digest).casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ContentAddressedStoreError("invalid CAS digest")
        return self.blob_root / digest[:2] / digest[2:]

    @staticmethod
    def digest(path: str | Path) -> str:
        """Hash a regular file in bounded memory."""
        source = Path(path)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_index(self) -> dict[str, dict[str, object]]:
        try:
            value = json.loads(self.index_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _save_index(self, value: dict[str, dict[str, object]]) -> None:
        temporary = self.index_path.with_name(
            f".{self.index_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("w") as stream:
                json.dump(value, stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
        finally:
            temporary.unlink(missing_ok=True)

    def put(self, source: str | Path, *, expected_sha256: str | None = None
            ) -> dict[str, object]:
        source = Path(source).resolve()
        if not source.is_file() or source.is_symlink():
            raise ContentAddressedStoreError("CAS source must be a regular file")
        fingerprint = list(_fingerprint(source))
        # The performance index is deliberately path-opaque so a portable
        # shared asset store never persists an author machine's absolute path.
        key = hashlib.sha256(str(source).encode()).hexdigest()
        expected = str(expected_sha256 or "").casefold() or None
        with self._locked():
            index = self._load_index()
            cached = index.get(key) or {}
            digest = (self.digest(source) if expected is not None else
                      (str(cached.get("sha256"))
                       if cached.get("fingerprint") == fingerprint else self.digest(source)))
            if expected is not None and digest != expected:
                raise ContentAddressedStoreError(
                    f"CAS source checksum mismatch: {source.name}")
            destination = self._path(digest)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(
                    f".{destination.name}.tmp-{uuid.uuid4().hex}")
                try:
                    _copy_sparse(source, temporary)
                    if self.digest(temporary) != digest:
                        raise ContentAddressedStoreError("CAS copy checksum mismatch")
                    temporary.chmod(0o444)
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
            elif destination.stat().st_size != source.stat().st_size:
                raise ContentAddressedStoreError("CAS digest collision")
            elif expected is not None and self.digest(destination) != digest:
                raise ContentAddressedStoreError("existing CAS blob checksum mismatch")
            index[key] = {"fingerprint": fingerprint, "sha256": digest}
            self._save_index(index)
        return {"blob_uri": f"{self.prefix}{digest}", "sha256": digest,
                "bytes": source.stat().st_size}

    def resolve(self, uri: str, *, verify: bool = False) -> Path:
        if not str(uri).startswith(self.prefix):
            raise ContentAddressedStoreError("unsupported CAS URI")
        digest = str(uri)[len(self.prefix):]
        path = self._path(digest)
        if not path.is_file() or path.is_symlink():
            raise ContentAddressedStoreError(f"CAS blob is missing: {uri}")
        if verify and self.digest(path) != digest:
            raise ContentAddressedStoreError(f"CAS blob checksum mismatch: {uri}")
        return path

    def materialize(self, uri: str, destination: str | Path, *,
                    executable: bool = False, writable: bool = False,
                    verify: bool = False) -> Path:
        source = self.resolve(uri, verify=verify)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if writable or executable:
            _copy_sparse(source, destination)
            destination.chmod(0o555 if executable else stat.S_IMODE(source.stat().st_mode) | 0o200)
        else:
            try:
                os.link(source, destination)
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
                    raise
                _copy_sparse(source, destination)
                destination.chmod(0o444)
        return destination


__all__ = ["ContentAddressedStore", "ContentAddressedStoreError"]
