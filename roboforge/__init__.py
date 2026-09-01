"""RoboForge v2 architecture spike.

This package is intentionally independent from the frozen RoboForge v1
repository. The experiment service is the authority for physical execution;
coding-harness integrations are clients of that service.
"""

from .models import AdapterResult, ArtifactHandle, ExperimentEvidence, RawArtifact
from .service import (
    BudgetExhausted,
    ExperimentService,
    IndeterminateExperiment,
    ProtocolError,
)
from .assets import AssetLibrary


def _keep_embodied_loop(agent):
    """Adapt OpenHands' text-terminal behavior for embodied sessions.

    A plain assistant message must not end a physical debugging run; the SDK
    remains responsible for scheduling the next step and persisting history.
    """
    from types import MethodType
    from openhands.sdk.agent.response_dispatch import ResponseDispatchMixin
    from openhands.sdk.conversation.state import ConversationExecutionStatus
    from openhands.sdk.event import MessageEvent
    from openhands.sdk.llm import Message, TextContent

    original = agent._handle_content_response

    def handle(self, message, llm_response, conversation, state, on_event):
        original(message, llm_response, conversation, state, on_event)
        if state.execution_status == ConversationExecutionStatus.FINISHED:
            state.execution_status = ConversationExecutionStatus.RUNNING
            on_event(MessageEvent(
                source="user", llm_message=Message(
                    role="user", content=[TextContent(
                        text="Continue the embodied task: use an available tool now; do not stop after commentary.")])) )

    agent._handle_content_response = MethodType(handle, agent)

def create_openhands_conversation(*, llm, workspace, persistence_dir, service, controller_path,
                                  asset_root=None, conversation_id=None, max_iterations=500):
    """Construct an OpenHands LocalConversation with RoboForge embodied tools."""
    from openhands.sdk import Agent
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from .runtime import register_spike_tools
    tools = register_spike_tools(service, workspace=workspace, controller_path=controller_path, asset_root=asset_root)
    agent = Agent(llm=llm, tools=tools, include_default_tools=["FinishTool", "ThinkTool"])
    _keep_embodied_loop(agent)
    return LocalConversation(agent=agent, workspace=workspace, persistence_dir=persistence_dir,
                             conversation_id=conversation_id,
                             max_iteration_per_run=max_iterations,
                             visualizer=None, delete_on_close=False)

__all__ = [
    "AdapterResult",
    "ArtifactHandle",
    "BudgetExhausted",
    "ExperimentEvidence",
    "ExperimentService",
    "IndeterminateExperiment",
    "ProtocolError",
    "RawArtifact",
    "AssetLibrary",
    "create_openhands_conversation",
]
