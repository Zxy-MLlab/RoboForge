"""Atomic run checkpoints and resumable state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class RecoveryError(RuntimeError):
    pass


def _payload_bytes(state: dict[str, Any]) -> bytes:
    return (json.dumps(state, indent=2, sort_keys=True, default=str) + "\n").encode()


def checkpoint_path(root: str | Path) -> Path:
    path = Path(root).resolve() / "checkpoint"
    if not path.exists(): path.mkdir(parents=True)
    return path / "state.json"


def load_checkpoint(root: str | Path) -> dict[str, Any] | None:
    path = checkpoint_path(root)
    if not path.is_file():
        return None
    try:
        envelope = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"checkpoint cannot be decoded: {path}") from exc
    if (not isinstance(envelope, dict)
            or envelope.get("protocol") != "roboforge-checkpoint-envelope-v1"
            or not isinstance(envelope.get("payload"), dict)
            or not isinstance(envelope.get("payload_sha256"), str)):
        raise RecoveryError("checkpoint envelope is missing or unsupported")
    payload = dict(envelope["payload"])
    actual = hashlib.sha256(_payload_bytes(payload)).hexdigest()
    if actual != envelope["payload_sha256"]:
        raise RecoveryError("checkpoint payload checksum mismatch")
    return payload


def save_checkpoint(root: str | Path, state: dict[str, Any]) -> Path:
    path = checkpoint_path(root); temporary = path.with_suffix(".tmp")
    payload = dict(state)
    envelope = {"protocol": "roboforge-checkpoint-envelope-v1",
                "payload_sha256": hashlib.sha256(_payload_bytes(payload)).hexdigest(),
                "payload": payload}
    try:
        with temporary.open("w") as stream:
            stream.write(json.dumps(envelope, indent=2, sort_keys=True, default=str) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = ["RecoveryError", "checkpoint_path", "load_checkpoint", "save_checkpoint"]
