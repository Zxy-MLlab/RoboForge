"""Structured function-calling tools exposed by the canonical coding agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping
from jsonschema import Draft202012Validator


@dataclass
class KernelTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Callable[..., Any]

    @property
    def schema(self):
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": dict(self.parameters)}}


class ToolRegistry:
    def __init__(self): self._items: dict[str, KernelTool] = {}

    def add(self, name: str, description: str, parameters: Mapping[str, Any], handler: Callable[..., Any]):
        Draft202012Validator.check_schema(dict(parameters))
        if name in self._items: raise ValueError(f"duplicate kernel tool: {name}")
        self._items[name] = KernelTool(name, description, parameters, handler)

    @property
    def schemas(self): return [item.schema for item in self._items.values()]

    def invoke(self, name: str, arguments: Mapping[str, Any]):
        item = self._items.get(name)
        if item is None: raise KeyError(name)
        Draft202012Validator(dict(item.parameters)).validate(arguments)
        return item.handler(**dict(arguments))

    def names(self): return tuple(self._items)


__all__ = ["KernelTool", "ToolRegistry"]
