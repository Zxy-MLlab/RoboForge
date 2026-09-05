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


def create_openhands_conversation(*, llm, workspace, persistence_dir, service, controller_path,
                                  asset_root=None, conversation_id=None, max_iterations=500,
                                  hook_config=None, terminal_env=None,
                                  max_budget_per_run=None, callbacks=None):
    """Construct an OpenHands conversation with native coding tools only."""
    from openhands.sdk import Agent
    from openhands.sdk.context.condenser import default_condenser
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from .runtime import register_native_tools
    tools = register_native_tools(service, workspace=workspace, controller_path=controller_path,
                                 asset_root=asset_root, terminal_env=terminal_env)
    # Match the public OpenHands default Agent configuration.  Long-running robot
    # campaigns contain many image and Terminal events; without a condenser the
    # complete event history is repeatedly sent to the model until the provider
    # rejects or disconnects the oversized request.  Keep condenser accounting
    # separate from the primary Agent usage, as OpenHands does for its own default
    # Agent and spawned subagents.
    condenser = default_condenser(
        llm.model_copy(update={"usage_id": "condenser"})
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        include_default_tools=["FinishTool", "ThinkTool"],
        condenser=condenser,
    )
    return LocalConversation(agent=agent, workspace=workspace, persistence_dir=persistence_dir,
                             conversation_id=conversation_id,
                             max_iteration_per_run=max_iterations,
                             max_budget_per_run=max_budget_per_run,
                             visualizer=None, delete_on_close=False,
                             hook_config=hook_config, callbacks=callbacks)

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
