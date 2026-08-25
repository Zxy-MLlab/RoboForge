from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class BudgetPolicy:
    max_steps: int = 60
    max_executions: int = 20
    timeout_seconds: float = 3600
    started: float = 0.0

    def __post_init__(self):
        self.started = self.started or time.monotonic()

    def exhausted(self, steps: int = 0, executions: int = 0) -> bool:
        return (steps >= self.max_steps or executions >= self.max_executions
                or time.monotonic() - self.started >= self.timeout_seconds)
