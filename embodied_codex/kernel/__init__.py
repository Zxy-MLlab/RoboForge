"""Small, environment-neutral Embodied Coding Agent kernel."""

from .agent_loop import AgentLoop, LoopBudget, ProtocolError
from .capability_manager import CapabilityManager, ExtractionLimits
from .assets import AssetRegistry
from .context import ContextBuilder, MinimalSystemPrompt
from .context_window import ContextWindowManager
from .events import EventStore
from .workspace import PersistentWorkspace
from .tools import ToolRegistry
from .sandbox import (SandboxBackend, PosixSandboxBackend, BubblewrapBackend,
                      UnsafeSandboxBackend, SandboxUnavailable, default_sandbox,
                      select_sandbox)

__all__ = [
    "AgentLoop",
    "LoopBudget",
    "ProtocolError",
    "AssetRegistry",
    "CapabilityManager",
    "ExtractionLimits",
    "ToolRegistry",
    "ContextBuilder",
    "ContextWindowManager",
    "EventStore",
    "MinimalSystemPrompt",
    "PersistentWorkspace",
    "BubblewrapBackend",
    "PosixSandboxBackend",
    "UnsafeSandboxBackend",
    "SandboxBackend",
    "SandboxUnavailable",
    "default_sandbox",
    "select_sandbox",
]
