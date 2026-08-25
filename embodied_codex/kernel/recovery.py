"""Restart helpers for an interrupted harness process."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_checkpoint(root: str | Path) -> dict[str, Any] | None:
    path = Path(root).resolve() / "checkpoint.json"
    if not path.is_file(): return None
    return json.loads(path.read_text())


def save_checkpoint(root: str | Path, state: dict[str, Any]) -> Path:
    path = Path(root).resolve() / "checkpoint.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)
    return path


__all__ = ["load_checkpoint", "save_checkpoint"]
