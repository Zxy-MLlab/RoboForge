"""Minimal signed evaluation receipt primitives.

The signing key is intentionally supplied only to the external evaluator
process. Controllers and the OpenHands process only receive the public receipt.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from .store import canonical_json

RECEIPT_VERSION = "roboforge-evaluation-receipt-v1"


def sign_receipt(payload: dict[str, Any], key: bytes) -> dict[str, Any]:
    body = {"version": RECEIPT_VERSION, **payload}
    body.pop("signature", None)
    body["signature"] = base64.urlsafe_b64encode(
        hmac.new(key, canonical_json(body), hashlib.sha256).digest()
    ).decode("ascii")
    return body


def verify_receipt(receipt: dict[str, Any], key: bytes, *, now: float | None = None,
                   max_age_seconds: float = 3600.0) -> bool:
    if not isinstance(receipt, dict) or receipt.get("version") != RECEIPT_VERSION:
        return False
    signature = receipt.get("signature")
    issued = receipt.get("issued_at")
    if not isinstance(signature, str) or not isinstance(issued, (int, float)):
        return False
    if abs((time.time() if now is None else now) - float(issued)) > max_age_seconds:
        return False
    body = dict(receipt); body.pop("signature", None)
    expected = base64.urlsafe_b64encode(
        hmac.new(key, canonical_json(body), hashlib.sha256).digest()
    ).decode("ascii")
    return hmac.compare_digest(signature, expected)


def receipt_digest(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(receipt)).hexdigest()
