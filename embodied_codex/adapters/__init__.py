"""Adapter plugin loading. The kernel knows only this factory contract."""
from .factory import adapter_doctor_task, adapter_preflight, load_adapter, register_adapter

__all__ = ["adapter_doctor_task", "adapter_preflight", "load_adapter", "register_adapter",
           "FrankaLiberoApi", "FrankaLiberoApiProxy"]


def __getattr__(name):
    if name in {"FrankaLiberoApi", "FrankaLiberoApiProxy"}:
        from .franka_libero_api import FrankaLiberoApi, FrankaLiberoApiProxy
        return {"FrankaLiberoApi": FrankaLiberoApi,
                "FrankaLiberoApiProxy": FrankaLiberoApiProxy}[name]
    raise AttributeError(name)
