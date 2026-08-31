"""Structured function-calling tools exposed by the canonical coding agent."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from jsonschema import Draft202012Validator


CONSEQUENCE_LEVELS = {
    "READ_ONLY", "WORKSPACE_MUTATION", "VALIDATION", "ASSET_MUTATION",
    "ENVIRONMENT_MUTATION", "PHYSICAL_INTERVENTION",
}

DECISION_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": ["string", "null"]},
        "evidence_refs": {"type": "array", "items": {
            "type": "string", "minLength": 1,
            "pattern": "^(evidence|artifact|run)://",
            "description": "Opaque evidence, artifact, or run reference returned by Harness tools."},
            "maxItems": 16},
        "hypothesis": {"type": ["string", "null"]},
        "decision": {"type": ["string", "null"]},
        "expected_effect": {"type": ["string", "null"]},
        "uncertainty": {"type": ["string", "null"]},
    },
    "required": ["goal", "evidence_refs", "hypothesis", "decision",
                 "expected_effect", "uncertainty"],
    "additionalProperties": False,
}


@dataclass
class KernelTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any]
    group: str = "core"
    consequence: str = "READ_ONLY"
    model_visible: bool = True

    @property
    def schema(self):
        description = self.description
        parameters = deepcopy(dict(self.parameters))
        if self.consequence not in {"READ_ONLY", "VALIDATION"}:
            description += (f" Consequence: {self.consequence}. Include the complete "
                            "model-authored decision_context in this same call; Harness "
                            "records and links it before execution.")
            properties = dict(parameters.get("properties") or {})
            properties["decision_context"] = deepcopy(DECISION_CONTEXT_SCHEMA)
            parameters["properties"] = properties
            required = list(parameters.get("required") or [])
            if "decision_context" not in required:
                required.append("decision_context")
            parameters["required"] = required
        return {"type": "function", "function": {"name": self.name,
                "description": description, "parameters": parameters}}


class ToolRegistry:
    def __init__(self):
        self._items: dict[str, KernelTool] = {}
        self._active_groups = {"core"}
        self._group_descriptions = {
            "core": "Workspace, asset discovery, Controller execution, evidence, and completion.",
        }

    def declare_group(self, name: str, description: str) -> None:
        name = str(name)
        if not name or name == "core":
            raise ValueError("optional tool group requires a non-core name")
        self._group_descriptions[name] = str(description)

    def add(self, name: str, description: str, parameters: Mapping[str, Any],
            handler: Callable[..., Any], *, group: str = "core",
            consequence: str = "READ_ONLY", model_visible: bool = True):
        Draft202012Validator.check_schema(dict(parameters))
        if name in self._items: raise ValueError(f"duplicate kernel tool: {name}")
        if group not in self._group_descriptions:
            raise ValueError(f"undeclared tool group: {group}")
        consequence = str(consequence).upper()
        if consequence == "CONSEQUENTIAL":
            consequence = "PHYSICAL_INTERVENTION"
        if consequence not in CONSEQUENCE_LEVELS:
            raise ValueError(f"unsupported tool consequence: {consequence}")
        self._items[name] = KernelTool(name, description, parameters, handler, group,
                                       consequence, bool(model_visible))

    @property
    def schemas(self):
        return [item.schema for item in self._items.values()
                if item.group in self._active_groups and item.model_visible]

    @property
    def active_groups(self):
        return tuple(sorted(self._active_groups))

    def group_index(self):
        return [{"group": name, "description": description,
                 "active": name in self._active_groups}
                for name, description in self._group_descriptions.items()]

    def activate(self, group: str):
        group = str(group)
        if group == "core" or group not in self._group_descriptions:
            raise KeyError(f"unknown optional tool group: {group}")
        self._active_groups.add(group)
        return {"group": group, "active": True,
                "tools": [item.name for item in self._items.values()
                          if item.group == group]}

    def deactivate(self, group: str):
        group = str(group)
        if group == "core" or group not in self._group_descriptions:
            raise KeyError(f"unknown optional tool group: {group}")
        self._active_groups.discard(group)
        return {"group": group, "active": False}

    def invoke(self, name: str, arguments: Mapping[str, Any]):
        item = self._items.get(name)
        if item is None: raise KeyError(name)
        if item.group not in self._active_groups:
            raise PermissionError(f"tool group is not active: {item.group}")
        Draft202012Validator(dict(item.parameters)).validate(arguments)
        return item.handler(**dict(arguments))

    def metadata(self, name: str) -> KernelTool:
        item = self._items.get(name)
        if item is None:
            raise KeyError(name)
        return item

    def names(self, *, active_only: bool = False):
        if not active_only:
            return tuple(self._items)
        return tuple(item.name for item in self._items.values()
                     if item.group in self._active_groups)


__all__ = ["KernelTool", "ToolRegistry", "CONSEQUENCE_LEVELS",
           "DECISION_CONTEXT_SCHEMA"]
