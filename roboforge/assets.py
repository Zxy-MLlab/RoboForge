"""Progressively disclosed embodied assets (experience, skill, capability)."""
from __future__ import annotations
import json, hashlib
import time
import re
from pathlib import Path
from typing import Any

KINDS = {"experiences", "skills", "capabilities"}
URI_KIND = {"experiences": "experience", "skills": "skill", "capabilities": "capability"}
TOKEN_CANONICAL = {
    "detection": "perception", "detect": "perception", "detector": "perception",
    "reference": "ref", "references": "ref", "handle": "ref", "handles": "ref",
    "extractor": "extract", "extraction": "extract",
}

def _tokens(value: str) -> set[str]:
    raw = {x for x in re.findall(r"[a-z0-9_]+", value.lower().replace("-", " ")) if len(x) > 2}
    return {TOKEN_CANONICAL.get(x, x) for x in raw}

class AssetLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        for kind in KINDS: (self.root / kind).mkdir(parents=True, exist_ok=True)
        self.audit_path = self.root / "usage.jsonl"

    def audit(self, operation: str, **payload: Any) -> None:
        row = {"timestamp_ns": time.time_ns(), "operation": operation, **payload}
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def was_read(self, asset_id: str, session_id: str | None = None) -> bool:
        if not self.audit_path.exists(): return False
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if (row.get("operation") == "read" and row.get("asset_id") == asset_id
                    and (session_id is None or row.get("session_id") == session_id)): return True
        return False

    def register(self, kind: str, *, name: str, purpose: str, description: str,
                 applicability: Any = None, evidence: Any = None,
                 provenance: Any = None, usage: str = "", implementation: Any = None) -> dict[str, Any]:
        if kind not in KINDS or not name.strip(): raise ValueError("invalid asset kind/name")
        payload = {"schema_version": 1, "name": name, "purpose": purpose,
                   "description": description, "applicability": applicability,
                   "evidence": evidence, "provenance": provenance,
                   "usage": usage, "implementation": implementation}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload["asset_id"] = f"{URI_KIND[kind]}://{digest}"
        path = self.root / kind / f"{digest}.json"
        if path.exists() and json.loads(path.read_text()) != payload: raise ValueError("asset collision")
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return self.summary(payload)

    def search(self, query: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        kinds = [kind] if kind else sorted(KINDS)
        tokens = _tokens(query)
        ranked = []
        for k in kinds:
            if k not in KINDS: raise ValueError("invalid asset kind")
            for p in sorted((self.root / k).glob("*.json")):
                try: data = json.loads(p.read_text())
                except Exception: continue
                hay = " ".join(str(data.get(x, "")) for x in
                    ("name", "purpose", "description", "applicability", "usage", "implementation"))
                hay = hay.replace("-", " ")
                hay_tokens = _tokens(hay)
                matched = sorted(tokens & hay_tokens)
                if not tokens or matched:
                    item = self.summary(data); item["matched_terms"] = matched
                    ranked.append((len(matched), item))
        out = [item for _, item in sorted(ranked, key=lambda row: (-row[0], str(row[1].get("name"))))]
        self.audit("search", query=query, kind=kind, result_ids=[x["asset_id"] for x in out])
        return out

    def read(self, asset_id: str, *, session_id: str | None = None) -> dict[str, Any]:
        digest = asset_id.split("://", 1)[-1]
        for k in KINDS:
            p = self.root / k / f"{digest}.json"
            if p.exists():
                value = json.loads(p.read_text()); self.audit("read", asset_id=asset_id,
                    session_id=session_id); return value
        raise KeyError(asset_id)

    @staticmethod
    def summary(data: dict[str, Any]) -> dict[str, Any]:
        return {k: data.get(k) for k in ("asset_id", "name", "purpose", "description", "applicability", "evidence", "provenance", "usage")}
