"""Compatibility import; implementation lives in embodied_codex.legacy."""
from .legacy import engineering as _implementation
globals().update({name: value for name, value in vars(_implementation).items() if not name.startswith("__")})
import sys as _sys
_sys.modules[__name__] = _implementation
