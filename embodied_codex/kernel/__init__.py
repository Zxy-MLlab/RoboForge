"""Small, environment-neutral Embodied Coding Agent kernel."""

from .agent_loop import AgentDecision, AgentLoop
from .assets import AssetRegistry
from .context import ContextBuilder, MinimalSystemPrompt
from .events import EventStore
from .workspace import PersistentWorkspace

__all__ = [
    "AgentDecision",
    "AgentLoop",
    "AssetRegistry",
    "ContextBuilder",
    "EventStore",
    "MinimalSystemPrompt",
    "PersistentWorkspace",
]
