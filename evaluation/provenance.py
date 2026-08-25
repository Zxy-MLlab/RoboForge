"""Research-only provenance and contamination audit hook."""
from .policy import BenchmarkPolicy


class ProvenancePolicy(BenchmarkPolicy):
    name = "provenance"
