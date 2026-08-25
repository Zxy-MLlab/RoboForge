from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable


_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_adapter(name: str, factory: Callable[..., Any]) -> None:
    if not callable(factory): raise TypeError("adapter factory must be callable")
    _FACTORIES[str(name)] = factory


def _load(spec: str):
    module, sep, name = str(spec).partition(":")
    if not sep: raise ValueError(f"adapter must be package:object: {spec}")
    return getattr(importlib.import_module(module), name)


def load_adapter(spec: str, *, task: str, run_dir: str | Path):
    if spec == "libero" or str(spec).startswith("libero@"):
        from .libero import create
        state = 0 if spec == "libero" else int(str(spec).split("@", 1)[1])
        return create(task=task, state=state, root=run_dir)
    if spec == "embodied_codex.adapters.libero":
        from .libero import create
        return create(task=task, root=run_dir)
    factory = _FACTORIES.get(spec) or _load(spec)
    if not inspect.isclass(factory) and not callable(factory): return factory
    for kwargs in ({"task": task, "root": Path(run_dir)}, {"task": task},
                   {"instruction": task}, {}):
        try: return factory(**kwargs)
        except TypeError: continue
    return factory(task)
