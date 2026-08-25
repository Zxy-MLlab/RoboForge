"""Adapter plugin loading. The kernel knows only this factory contract."""
from .factory import adapter_preflight, load_adapter, register_adapter

__all__ = ["adapter_preflight", "load_adapter", "register_adapter"]
