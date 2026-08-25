"""Append-only event store. Identical payloads are independent events by default."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import os
import threading
import time
import uuid
from typing import Any, Mapping


@dataclass(frozen=True)
class Event:
    event_id: str; kind: str; payload: Mapping[str, Any]; sequence: int


class EventStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"; self._lock = threading.Lock()

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def commit(self, kind: str, payload: Mapping[str, Any], *, event_id: str | None = None) -> Event:
        data = dict(payload); identifier = str(event_id or uuid.uuid4())
        with self._lock:
            rows = self.events(); existing = next((x for x in rows if x.get("event_id") == identifier), None)
            if existing:
                return Event(identifier, str(existing["kind"]), existing.get("payload", {}), int(existing["sequence"]))
            sequence = len(rows) + 1
            record = {"event_id": identifier, "sequence": sequence, "kind": str(kind),
                      "payload": data, "committed_unix": time.time()}
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n"); stream.flush(); os.fsync(stream.fileno())
        return Event(identifier, str(kind), data, sequence)

    def has_event(self, event_id: str) -> bool:
        return any(row.get("event_id") == event_id for row in self.events())


__all__ = ["Event", "EventStore"]
