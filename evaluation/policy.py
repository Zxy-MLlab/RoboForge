"""Optional benchmark policy composition. The kernel never imports this module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EvaluationPolicy:
    """A policy hook evaluated around a completed kernel run."""
    name: str = "evaluation"
    before: Callable[[Any], None] | None = None
    after: Callable[[Any], Any] | None = None

    def evaluate(self, run: Any) -> Any:
        if self.before: self.before(run)
        return self.after(run) if self.after else run


class BenchmarkPolicy(EvaluationPolicy):
    """Marker base for anti-cheating/generalization/sealed policies."""


__all__ = ["EvaluationPolicy", "BenchmarkPolicy"]
