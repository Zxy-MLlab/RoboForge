from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..store import canonical_json


@dataclass(frozen=True)
class SplitManifest:
    task: str
    development: tuple[int, ...]
    contaminated: tuple[int, ...]
    final_held_out: tuple[int, ...]
    digest: str
    created_unix: int

    def as_dict(self) -> dict:
        return {
            "schema_version": "roboforge-state-split-v1",
            "task": self.task,
            "state_semantics": "LIBERO initial-state index (not random seed)",
            "development": list(self.development),
            "contaminated": list(self.contaminated),
            "final_held_out": list(self.final_held_out),
            "manifest_sha256": self.digest,
            "created_unix": self.created_unix,
        }


def create_split_manifest(path: str | Path, *, task: str,
                          development: Iterable[int], contaminated: Iterable[int],
                          final_held_out: Iterable[int]) -> SplitManifest:
    target = Path(path).resolve()
    dev, dirty, held = tuple(sorted(set(map(int, development)))), tuple(sorted(set(map(int, contaminated)))), tuple(sorted(set(map(int, final_held_out))))
    groups = [set(dev), set(dirty), set(held)]
    if any(not group for group in groups):
        raise ValueError("development, contaminated and final_held_out must be non-empty")
    if set().union(*groups) and sum(map(len, groups)) != len(set().union(*groups)):
        raise ValueError("state split groups overlap")
    created_unix = int(__import__("time").time())
    body = {"schema_version": "roboforge-state-split-v1", "task": str(task),
            "state_semantics": "LIBERO initial-state index (not random seed)",
            "development": list(dev), "contaminated": list(dirty),
            "final_held_out": list(held), "created_unix": created_unix}
    digest = hashlib.sha256(canonical_json(body)).hexdigest()
    payload = {**body, "manifest_sha256": digest}
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("split manifest is immutable and already differs")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(payload) + b"\n")
        os.chmod(target, 0o444)
    return SplitManifest(str(task), dev, dirty, held, digest, created_unix)
