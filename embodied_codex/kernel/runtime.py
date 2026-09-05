"""Compatibility export for the canonical candidate runtime.

The implementation lives in :mod:`roboforge.candidate_runtime.controller`.
Historical imports remain valid for regression tests, while canonical code
imports the RoboForge module directly.
"""
from roboforge.candidate_runtime.controller import (
    ControllerRuntime, ControllerRuntimeError, _ARGUMENT_KEYS, _trace_value,
)

__all__ = ["ControllerRuntime", "ControllerRuntimeError"]
