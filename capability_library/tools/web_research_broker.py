"""Audited open-web and external-model research broker.

Model answers are hypotheses only. They never bypass the provenance gate or
become action-selection inputs without a separately registered, verified tool.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping


def _append(path: str, event: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True) + "\n")


def consult_external_model(
    question: str,
    *,
    provider: str = "unconfigured",
    ask_fn: Callable[[str], str] | None = None,
    ledger_path: str = "artifacts/capability_acquisition.jsonl",
) -> dict[str, Any]:
    """Ask an external model for research leads and mark them unverified."""
    question = str(question).strip()
    if not question:
        return {"success": False, "reason": "question must not be empty"}
    event: dict[str, Any] = {
        "event": "external_model_consultation",
        "timestamp_unix": time.time(),
        "provider": str(provider),
        "question": question,
        "verified": False,
        "action_selection_allowed": False,
    }
    if ask_fn is None:
        event["status"] = "not_configured"
        _append(ledger_path, event)
        return {"success": False, "reason": "no external model adapter configured"}
    try:
        answer = str(ask_fn(question))
    except Exception as exc:
        event["status"] = "error"
        event["error"] = f"{type(exc).__name__}: {exc}"
        _append(ledger_path, event)
        return {"success": False, "reason": event["error"]}
    event.update({
        "status": "received_unverified",
        "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
        "answer_length": len(answer),
    })
    _append(ledger_path, event)
    return {
        "success": True,
        "provider": provider,
        "answer": answer,
        "verified": False,
        "action_selection_allowed": False,
    }


__all__ = ["consult_external_model"]
