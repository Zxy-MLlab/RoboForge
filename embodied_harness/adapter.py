"""Environment-neutral Robot Adapter boundary owned by the deployment."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class RobotAdapter(Protocol):
    """One persistent robot episode exposed through an evaluator-blind API."""

    @property
    def initial_context(self) -> Mapping[str, Any]: ...

    def register_capability(
        self, tool_id: str,
        function: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None: ...

    def dispatch(self, method: str, arguments: Mapping[str, Any]) -> Any: ...

    def sensor_report(self, execution: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


RPC_METHODS = frozenset({"instruction", "sense", "act", "use", "verify", "record"})


__all__ = ["RPC_METHODS", "RobotAdapter"]
