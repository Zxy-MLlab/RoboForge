"""Compatibility import; implementation lives in embodied_codex.legacy."""
from .legacy import task_model as _implementation
globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
