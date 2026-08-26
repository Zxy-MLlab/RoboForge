"""External benchmark orchestration around the policy-free Harness loop."""
from __future__ import annotations

from typing import Any, Iterable

from .evidence import EvaluatorEvidence


class BenchmarkRunner:
    """Apply evaluation lifecycle outside ``AgentLoop``.

    Policies receive the loop only from this Evaluation-layer runner. The
    Harness loop neither imports policies nor invokes evaluator callbacks.
    """

    def __init__(self, loop: Any, policies: Iterable[Any] = ()):
        self.loop = loop
        self.policies = tuple(policies)
        self.latest_evidence: EvaluatorEvidence | None = None

    def run(self, task: str | None = None):
        for policy in self.policies:
            before = getattr(policy, "before_run", None)
            if callable(before):
                before(self.loop)
        result = self.loop.run(task)
        reference = result.get("latest_evidence") if isinstance(result, dict) else None
        if isinstance(reference, dict) and reference.get("artifact_uri"):
            self.latest_evidence = EvaluatorEvidence.load(self.loop.root, reference)
        for policy in self.policies:
            after = getattr(policy, "after_run", None)
            if callable(after):
                after(self.loop, result)
        return result


__all__ = ["BenchmarkRunner"]
