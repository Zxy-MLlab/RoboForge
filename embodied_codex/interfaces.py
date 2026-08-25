"""Small deployment boundary; task intelligence lives in workspace programs."""
from __future__ import annotations

from typing import Any, Mapping, Protocol


class RobotDeployment(Protocol):
    @property
    def instruction(self) -> str: ...
    def dispatch(self, method: str, arguments: Mapping[str, Any]) -> Any: ...
    def project_rpc_output(self, method: str, arguments: Mapping[str,Any],
                           result: Any) -> Any: ...
    def register_capability(self, tool_id: str, function: Any,
                            contract: Mapping[str, Any]) -> None: ...
    def sensor_report(self, execution: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


ALLOWED_RPC = frozenset({"observe", "act", "use", "verify", "record"})

__all__ = ["ALLOWED_RPC", "RobotDeployment"]
