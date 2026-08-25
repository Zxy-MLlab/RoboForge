from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class SafetyPolicy:
    """Adapter-independent envelope checks; physical limits belong to adapters."""
    allowed_rpc: tuple[str, ...] = ("observe", "use", "act", "verify", "record")

    def validate_rpc(self, method: str, arguments: Mapping[str, Any]) -> None:
        if method not in self.allowed_rpc:
            raise ValueError(f"unsupported adapter operation: {method}")
        if not isinstance(arguments, Mapping):
            raise TypeError("adapter arguments must be an object")
