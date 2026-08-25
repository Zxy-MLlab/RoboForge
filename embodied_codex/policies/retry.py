from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    retryable_errors: tuple[str, ...] = ("TimeoutError", "ConnectionError")

    def allows(self, error: BaseException, attempt: int) -> bool:
        return attempt < self.max_attempts and type(error).__name__ in self.retryable_errors
