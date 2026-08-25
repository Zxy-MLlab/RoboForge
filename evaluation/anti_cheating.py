"""Research-only anti-cheating policy hook."""
from .policy import BenchmarkPolicy


class AntiCheatingPolicy(BenchmarkPolicy):
    name = "anti_cheating"
