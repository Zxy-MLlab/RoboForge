"""Adapter plugin loading. The kernel knows only this factory contract."""
from .factory import adapter_doctor_task, adapter_preflight, load_adapter, register_adapter

__all__ = ["adapter_doctor_task", "adapter_preflight", "load_adapter", "register_adapter"]
