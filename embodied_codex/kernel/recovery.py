"""Atomic run checkpoints and resumable state."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def checkpoint_path(root: str | Path) -> Path:
    path = Path(root).resolve() / "checkpoint"; path.mkdir(parents=True, exist_ok=True)
    return path / "state.json"


def load_checkpoint(root: str | Path) -> dict[str, Any] | None:
    path = checkpoint_path(root)
    return json.loads(path.read_text()) if path.is_file() else None


def save_checkpoint(root: str | Path, state: dict[str, Any]) -> Path:
    path = checkpoint_path(root); temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)
    return path


__all__ = ["checkpoint_path", "load_checkpoint", "save_checkpoint"]
