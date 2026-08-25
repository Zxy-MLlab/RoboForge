"""Research-only sealed evaluator barrier."""
from .policy import BenchmarkPolicy


class SealedEvaluationPolicy(BenchmarkPolicy):
    name = "sealed_evaluation"
