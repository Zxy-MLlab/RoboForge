"""Executable benchmark policy lifecycle, outside the kernel dependency graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationPolicy:
    name: str = "evaluation"
    def before_run(self, loop: Any) -> None: pass
    def after_run(self, loop: Any, result: dict[str, Any]) -> None: pass


class BenchmarkPolicy(EvaluationPolicy):
    pass
