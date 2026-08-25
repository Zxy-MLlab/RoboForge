"""Crash-tolerant append-only event log with an integrity chain."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Mapping


class EventStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    payload: Mapping[str, Any]
    sequence: int


def _encoded(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, separators=(",", ":"),
                       default=str) + "\n").encode()


def _digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "record_sha256"}
    return hashlib.sha256(_encoded(payload)).hexdigest()


class EventStore:
    def __init__(self, root: str | Path, *, protect: bool = False):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self.lock_path = self.root / ".events.lock"
        self._thread_lock = threading.RLock()
        # Kept for API compatibility. Isolation is supplied by SandboxBackend,
        # never by racing permission changes on a shared directory.
        self.protect = bool(protect)

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _read_locked(self, *, repair_tail: bool) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        previous: str | None = None
        valid_offset = 0
        with self.path.open("rb") as stream:
            while True:
                start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                complete = line.endswith(b"\n")
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    if repair_tail and not complete and stream.read(1) == b"":
                        self._truncate_locked(valid_offset)
                        break
                    raise EventStoreError(f"event log is corrupt at byte {start}") from exc
                if not isinstance(row, dict):
                    raise EventStoreError(f"event record is not an object at byte {start}")
                recorded = row.get("record_sha256")
                if recorded is not None:
                    if row.get("previous_sha256") != previous or recorded != _digest(row):
                        raise EventStoreError(f"event integrity chain failed at byte {start}")
                    previous = str(recorded)
                else:
                    # Existing v1 records remain readable. New records chain to
                    # their canonical bytes without rewriting history.
                    previous = hashlib.sha256(_encoded(row)).hexdigest()
                rows.append(row)
                valid_offset = stream.tell()
                if not complete:
                    if repair_tail:
                        rows.pop()
                        self._truncate_locked(start)
                        break
                    raise EventStoreError("event log ends with an incomplete record")
        return rows

    def _truncate_locked(self, offset: int) -> None:
        descriptor = os.open(self.path, os.O_WRONLY)
        try:
            os.ftruncate(descriptor, int(offset))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def events(self) -> list[dict[str, Any]]:
        with self._locked():
            return self._read_locked(repair_tail=True)

    def commit(self, kind: str, payload: Mapping[str, Any], *,
               event_id: str | None = None) -> Event:
        data = dict(payload)
        identifier = str(event_id or uuid.uuid4())
        with self._locked():
            rows = self._read_locked(repair_tail=True)
            existing = next((row for row in rows
                             if row.get("event_id") == identifier), None)
            if existing:
                return Event(identifier, str(existing["kind"]),
                             existing.get("payload", {}), int(existing["sequence"]))
            previous = None
            if rows:
                previous = str(rows[-1].get("record_sha256")
                               or hashlib.sha256(_encoded(rows[-1])).hexdigest())
            sequence = int(rows[-1]["sequence"] if rows else 0) + 1
            record = {"event_id": identifier, "sequence": sequence,
                      "kind": str(kind), "payload": data,
                      "committed_unix": time.time(),
                      "previous_sha256": previous}
            record["record_sha256"] = _digest(record)
            encoded = _encoded(record)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return Event(identifier, str(kind), data, sequence)

    def has_event(self, event_id: str) -> bool:
        return any(row.get("event_id") == event_id for row in self.events())


__all__ = ["Event", "EventStore", "EventStoreError"]
