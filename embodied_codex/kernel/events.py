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
    def __init__(self, root: str | Path, *, protect: bool = False,
                 max_bytes: int = 512 * 1024 * 1024,
                 max_record_bytes: int = 1024 * 1024):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self.tail_path = self.root / "tail.json"
        self.lock_path = self.root / ".events.lock"
        self.max_bytes = int(max_bytes)
        self.max_record_bytes = int(max_record_bytes)
        if self.max_bytes < 1 or self.max_record_bytes < 1:
            raise ValueError("EventStore quotas must be positive")
        self._thread_lock = threading.RLock()
        self._rows: list[dict[str, Any]] = []
        # Kept for API compatibility. Isolation is supplied by SandboxBackend,
        # never by racing permission changes on a shared directory.
        self.protect = bool(protect)
        with self._locked():
            self._recover_locked()

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

    @staticmethod
    def _tail_digest(value: Mapping[str, Any]) -> str:
        payload = {key: item for key, item in value.items()
                   if key != "tail_sha256"}
        return hashlib.sha256(_encoded(payload)).hexdigest()

    def _save_tail_locked(self, *, sequence: int, previous_sha256: str | None,
                          offset: int) -> None:
        value = {"protocol": "roboforge-event-tail-v1",
                 "sequence": int(sequence), "previous_sha256": previous_sha256,
                 "offset": int(offset)}
        value["tail_sha256"] = self._tail_digest(value)
        temporary = self.tail_path.with_name(
            f".{self.tail_path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with temporary.open("wb") as stream:
                stream.write(_encoded(value)); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.tail_path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_tail_locked(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.tail_path.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        if (not isinstance(value, dict)
                or value.get("protocol") != "roboforge-event-tail-v1"
                or value.get("tail_sha256") != self._tail_digest(value)):
            return None
        return value

    def _recover_locked(self) -> None:
        self._rows = self._read_locked(repair_tail=True)
        previous = None
        sequence = 0
        if self._rows:
            last = self._rows[-1]
            previous = str(last.get("record_sha256")
                           or hashlib.sha256(_encoded(last)).hexdigest())
            sequence = int(last["sequence"])
        offset = self.path.stat().st_size if self.path.exists() else 0
        self._save_tail_locked(sequence=sequence, previous_sha256=previous,
                               offset=offset)

    def _append_tail_locked(self) -> dict[str, Any]:
        tail = self._load_tail_locked()
        size = self.path.stat().st_size if self.path.exists() else 0
        if tail is None or int(tail.get("offset", -1)) != size:
            self._recover_locked()
            tail = self._load_tail_locked()
        if tail is None:
            raise EventStoreError("event tail metadata cannot be recovered")
        return tail

    def events(self) -> list[dict[str, Any]]:
        with self._locked():
            # Explicit history reads are also recovery boundaries. Normal
            # append never enters this full-chain scan while tail metadata is valid.
            self._recover_locked()
            return [dict(row) for row in self._rows]

    def commit(self, kind: str, payload: Mapping[str, Any], *,
               event_id: str | None = None) -> Event:
        data = dict(payload)
        identifier = str(event_id or uuid.uuid4())
        with self._locked():
            tail = self._append_tail_locked()
            existing = (next((row for row in self._rows
                              if row.get("event_id") == identifier), None)
                        if event_id is not None else None)
            if existing is None and event_id is not None and int(tail["sequence"]) > len(self._rows):
                self._recover_locked()
                tail = self._append_tail_locked()
                existing = next((row for row in self._rows
                                 if row.get("event_id") == identifier), None)
            if existing:
                return Event(identifier, str(existing["kind"]),
                             existing.get("payload", {}), int(existing["sequence"]))
            previous = tail.get("previous_sha256")
            sequence = int(tail["sequence"]) + 1
            record = {"event_id": identifier, "sequence": sequence,
                      "kind": str(kind), "payload": data,
                      "committed_unix": time.time(),
                      "previous_sha256": previous}
            record["record_sha256"] = _digest(record)
            encoded = _encoded(record)
            if len(encoded) > self.max_record_bytes:
                raise EventStoreError("event record exceeds the disk quota")
            if int(tail["offset"]) + len(encoded) > self.max_bytes:
                raise EventStoreError("event log exceeds the disk quota")
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            offset = int(tail["offset"]) + len(encoded)
            self._save_tail_locked(sequence=sequence,
                                   previous_sha256=record["record_sha256"],
                                   offset=offset)
            self._rows.append(record)
        return Event(identifier, str(kind), data, sequence)

    def has_event(self, event_id: str) -> bool:
        with self._locked():
            tail = self._append_tail_locked()
            if int(tail["sequence"]) > len(self._rows):
                self._recover_locked()
            return any(row.get("event_id") == event_id for row in self._rows)


__all__ = ["Event", "EventStore", "EventStoreError"]
