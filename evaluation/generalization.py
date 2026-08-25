"""Research-only development/generalization gate."""
from .policy import BenchmarkPolicy


class GeneralizationPolicy(BenchmarkPolicy):
    name = "generalization"
