"""Environment-neutral asset facade with progressive loading semantics."""
from __future__ import annotations

from typing import Any, Mapping


class AssetRegistry:
    def __init__(self, *, tools=None, skills=None, experiences=None, gaps=None):
        self.tools, self.skills = tools, skills
        self.experiences, self.gaps = experiences, gaps

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        result = {}
        for name, library in (("tools", self.tools), ("skills", self.skills),
                              ("experiences", self.experiences), ("gaps", self.gaps)):
            if library is not None:
                result[name] = library.search(str(query), limit=max(1, int(limit)))
        return result

    def inspect(self, asset_id: str, *, include_source: bool = False) -> Any:
        identifier = str(asset_id)
        for library, method in ((self.tools, "inspect"), (self.skills, "inspect"),
                                (self.experiences, "inspect"), (self.gaps, "inspect")):
            if library is None: continue
            try:
                value = getattr(library, method)(identifier)
                if library is self.tools and isinstance(value, dict):
                    # Tool source is a separate, explicit escalation. Manual
                    # and schema remain the default implementation detail.
                    value = dict(value)
                    if not include_source:
                        value.pop("source", None)
                        value["manual"] = library.manual(identifier).get("manual", {})
                return value
            except (FileNotFoundError, KeyError, ValueError): continue
        raise KeyError(f"unknown asset: {asset_id}")

    def load_source(self, asset_id: str) -> Any:
        if self.tools is None: raise KeyError(asset_id)
        value = self.tools.inspect(str(asset_id))
        return {"asset_id": asset_id, "source": value.get("source"),
                "manifest": value.get("manifest")}

    def save(self, capability: Mapping[str, Any]) -> Any:
        """Persist a model-selected asset through the existing immutable libraries."""
        kind = str(capability.get("kind") or capability.get("asset_kind") or "")
        payload = dict(capability.get("payload") or capability)
        if kind == "experience" and self.experiences:
            return self.experiences.register(**payload)
        if kind == "gap" and self.gaps:
            return self.gaps.publish(**payload)
        if kind == "skill" and self.skills:
            return self.skills.freeze(**payload)
        if kind == "tool" and self.tools:
            return self.tools.register_tool(**payload)
        raise ValueError(f"asset kind is unavailable: {kind}")

    def runtime_functions(self):
        return self.tools.runtime_functions() if self.tools is not None else {}


__all__ = ["AssetRegistry"]
