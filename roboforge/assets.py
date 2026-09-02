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
        (self.root / "decisions").mkdir(parents=True, exist_ok=True)

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
        verification_status = "candidate" if kind == "capabilities" else "recorded"
        payload = {"schema_version": 1, "name": name, "purpose": purpose,
                   "description": description, "applicability": applicability,
                   "evidence": evidence, "provenance": provenance,
                   "usage": usage, "implementation": implementation,
                   "verification_status": verification_status,
                   "verification_decision": None}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        payload["asset_id"] = f"{URI_KIND[kind]}://{digest}"
        path = self.root / kind / f"{digest}.json"
        if path.exists() and json.loads(path.read_text()) != payload: raise ValueError("asset collision")
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        return self.summary(payload)

    def decide_capability(self, asset_id: str, *, decision: str,
                          evidence: list[str], note: str) -> dict[str, Any]:
        if decision not in {"promoted", "rejected"}:
            raise ValueError("decision must be promoted or rejected")
        if not asset_id.startswith("capability://") or not evidence or not note.strip():
            raise ValueError("capability decision requires id, evidence, and note")
        digest = asset_id.split("://", 1)[1]
        path = self.root / "capabilities" / f"{digest}.json"
        if not path.is_file(): raise KeyError(asset_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        current = value.get("verification_status", "candidate")
        requested = {"decision": decision, "evidence": list(evidence), "note": note}
        existing_decisions = []
        for decision_path in sorted((self.root / "decisions").glob("*.json")):
            try:
                candidate_decision = json.loads(decision_path.read_text(encoding="utf-8"))
                if candidate_decision.get("capability_id") == asset_id:
                    existing_decisions.append(candidate_decision)
            except (OSError, json.JSONDecodeError):
                continue
        if existing_decisions:
            existing = existing_decisions[0]["decision"]
            if existing != requested:
                raise ValueError("capability decision is immutable")
            value["verification_status"] = existing["decision"]
            value["verification_decision"] = {**existing, "decision_id": existing_decisions[0]["decision_id"]}
            return self.summary(value)
        if current in {"promoted", "rejected"}:
            if value.get("verification_decision") != requested:
                raise ValueError("capability decision is immutable")
            return self.summary(value)
        if current != "candidate": raise ValueError("only candidates can be decided")
        decision_payload = {"schema_version": 1, "capability_id": asset_id,
                            "capability_digest": digest, "decision": requested}
        decision_digest = hashlib.sha256(json.dumps(decision_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        decision_payload["decision_id"] = f"decision://{decision_digest}"
        decision_path = self.root / "decisions" / f"{decision_digest}.json"
        encoded = json.dumps(decision_payload, sort_keys=True, indent=2).encode()
        if decision_path.exists() and decision_path.read_bytes() != encoded:
            raise ValueError("decision collision")
        decision_path.write_bytes(encoded)
        self.audit("capability_decision", asset_id=asset_id, **requested)
        value = dict(value)
        value["verification_status"] = decision
        value["verification_decision"] = {**requested, "decision_id": decision_payload["decision_id"]}
        return self.summary(value)

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
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise KeyError(asset_id)
        for k in KINDS:
            p = self.root / k / f"{digest}.json"
            if p.exists():
                value = json.loads(p.read_text())
                # Content-addressed objects are immutable: detect tampering
                # even when an attacker edits the JSON in place.
                canonical = dict(value)
                canonical.pop("asset_id", None)
                actual = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                if actual != digest:
                    raise ValueError("CAS object digest mismatch")
                if k == "capabilities":
                    decisions = sorted(self.root.joinpath("decisions").glob("*.json"))
                    for decision_path in decisions:
                        try: decision = json.loads(decision_path.read_text())
                        except (OSError, json.JSONDecodeError): continue
                        if decision.get("capability_id") == asset_id:
                            value["verification_status"] = decision["decision"]["decision"]
                            value["verification_decision"] = {**decision["decision"], "decision_id": decision["decision_id"]}
                            break
                self.audit("read", asset_id=asset_id,
                    session_id=session_id); return value
        raise KeyError(asset_id)

    @staticmethod
    def summary(data: dict[str, Any]) -> dict[str, Any]:
        return {k: data.get(k) for k in ("asset_id", "name", "purpose", "description", "applicability", "evidence", "provenance", "usage", "verification_status", "verification_decision")}
