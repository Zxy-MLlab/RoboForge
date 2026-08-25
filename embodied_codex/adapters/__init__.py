"""Adapter plugin loading. The kernel knows only this factory contract."""
from .factory import load_adapter, register_adapter

__all__ = ["load_adapter", "register_adapter"]
