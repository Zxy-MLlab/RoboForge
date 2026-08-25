"""Small, environment-neutral Embodied Coding Agent kernel."""

from .agent_loop import AgentLoop, LoopBudget, ProtocolError
from .capability_manager import CapabilityManager
from .assets import AssetRegistry
from .context import ContextBuilder, MinimalSystemPrompt
from .events import EventStore
from .workspace import PersistentWorkspace
from .tools import ToolRegistry

__all__ = [
    "AgentLoop",
    "LoopBudget",
    "ProtocolError",
    "AssetRegistry",
    "CapabilityManager",
    "ToolRegistry",
    "ContextBuilder",
    "EventStore",
    "MinimalSystemPrompt",
    "PersistentWorkspace",
]
