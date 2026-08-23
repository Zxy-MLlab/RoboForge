from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass
class Function:
    name: str; description: str; parameters: Mapping[str, Any]; call: Callable[..., Any]
    available: Callable[[],bool]|None = None
    @property
    def schema(self):
        return {"type": "function", "function": {"name": self.name,
                "description": self.description, "parameters": dict(self.parameters)}}


class FunctionRegistry:
    def __init__(self): self.items: dict[str, Function] = {}
    def add(self, name, description, parameters, call, available=None):
        if name in self.items: raise ValueError(name)
        self.items[name] = Function(name, description, parameters, call, available)
    @property
    def schemas(self):
        return [item.schema for item in self.items.values()
                if item.available is None or item.available()]
    def invoke(self, name, arguments):
        if name not in self.items: raise KeyError(name)
        item=self.items[name]
        if item.available is not None and not item.available():
            raise KeyError(f"Tool unavailable in current lifecycle phase: {name}")
        return item.call(**dict(arguments))

__all__ = ["FunctionRegistry"]
