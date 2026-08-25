"""Append-only event log with fsync and idempotent event identifiers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import threading
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    payload: Mapping[str, Any]
    sequence: int


class EventStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self._lock = threading.Lock()

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def commit(self, kind: str, payload: Mapping[str, Any], *, event_id: str | None = None) -> Event:
        data = dict(payload)
        identifier = event_id or hashlib.sha256(
            json.dumps({"kind": kind, "payload": data}, sort_keys=True, default=str).encode()
        ).hexdigest()
        with self._lock:
            existing = next((item for item in self.events() if item.get("event_id") == identifier), None)
            if existing:
                return Event(identifier, str(existing["kind"]), existing.get("payload", {}), int(existing["sequence"]))
            sequence = len(self.events()) + 1
            record = {"event_id": identifier, "sequence": sequence, "kind": str(kind),
                      "payload": data, "committed_unix": time.time()}
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                stream.flush(); os.fsync(stream.fileno())
        return Event(identifier, str(kind), data, sequence)


__all__ = ["Event", "EventStore"]
