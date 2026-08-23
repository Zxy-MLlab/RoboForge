"""Small standalone function-tool registry for model-driven engineering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class FunctionTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    function: Callable[..., Any]

    @property
    def model_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name, "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


class ToolRegistry:
    def __init__(self) -> None: self._tools: dict[str, FunctionTool] = {}

    def register(
        self, *, name: str, description: str, parameters: Mapping[str, Any],
        function: Callable[..., Any],
    ) -> None:
        if name in self._tools: raise ValueError(f"duplicate tool: {name}")
        self._tools[name] = FunctionTool(name, description, dict(parameters), function)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [tool.model_schema for tool in self._tools.values()]

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name not in self._tools: raise KeyError(name)
        return self._tools[name].function(**dict(arguments))

    def names(self) -> list[str]: return sorted(self._tools)


__all__ = ["ToolRegistry"]
