"""Composition root for the self-evolving Embodied Frontier Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from asset_provenance_gate import register_asset_provenance_tool
from public_resource_market import register_public_resource_tools
from self_evolve import SelfEvolutionController, register_self_evolution_tool
from web_research_broker import consult_external_model
from asset_registry import find_assets, record_asset_reuse, register_asset
from controller_harness import ControllerWorkspace, register_controller_authoring_tools
from controller_program_workspace import (
    ControllerProgramWorkspace,
    register_controller_program_tools,
)
from capability_workspace import CapabilityWorkspace, register_capability_workspace_tools
from agent_engineering_workspace import (
    AgentEngineeringWorkspace,
    register_agent_engineering_tools,
)
from stage_node_workspace import StageNodeWorkspace, register_stage_node_tools
from controller_graph_workspace import (
    ControllerGraphWorkspace,
    register_controller_graph_tools,
)


def make_frontier_registrar(
    current_tasks: Iterable[str],
    *,
    ledger_path: str = "artifacts/capability_acquisition.jsonl",
    state_path: str = "artifacts/self_evolution_state.json",
    integrate_fn: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    retry_fn: Callable[[], dict[str, Any]] | None = None,
    controller_workspace: str | Path | None = None,
    controller_timeout_sec: int = 1800,
    max_controller_executions: int | None = None,
    capability_workspace: str | Path | None = None,
    include_controller_tools: bool = True,
    controller_program_workspace: str | Path | None = None,
    controller_program_executor: Callable[[str], Mapping[str, Any]] | None = None,
    required_controller_revision: Mapping[str, Any] | None = None,
    stage_node_workspace: str | Path | None = None,
    controller_graph_workspace: str | Path | None = None,
    controller_graph_executor: Callable[[str], Mapping[str, Any]] | None = None,
    required_graph_revision: Mapping[str, Any] | None = None,
    engineering_workspace: str | Path | None = None,
    engineering_read_roots: Mapping[str, str | Path] | None = None,
):
    """Return a Thea ``builtin_registrar`` for a task-scoped capability market.

    ``current_tasks`` is a provenance boundary, never an action-selection input.
    The returned registrar can be passed directly to ``Harness``. State is
    exported after each self-evolution tool call so a process restart does not
    erase rejected candidates or failed acquisition attempts.
    """
    task_tuple = tuple(str(item) for item in current_tasks)
    controller = SelfEvolutionController(
        current_tasks=task_tuple,
        ledger_path=ledger_path,
    )

    def register(registry: Any) -> None:
        capability_store = (
            CapabilityWorkspace(
                capability_workspace,
                python="/data/zxy/envs/vla-report/bin/python",
                library_path=Path(__file__).resolve().parent / "library.json",
            )
            if capability_workspace is not None
            else None
        )
        register_public_resource_tools(registry)
        registry.tool(
            name="register_capability_asset",
            description="Persist a provenance-audited tool, skill, model, or experience in the shared capability library.",
        )(register_asset)

        @registry.tool(
            name="register_public_research_lead",
            description=(
                "Register one open-web research lead as an unverified capability "
                "record. It is a discovery record only and cannot drive actions "
                "until provenance, installation, and tests are completed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "minLength": 3},
                    "kind": {"type": "string", "enum": ["tool", "skill", "model", "algorithm"]},
                    "name": {"type": "string", "minLength": 1},
                    "source_urls": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                },
                "required": ["asset_id", "kind", "name", "source_urls"],
                "additionalProperties": False,
            },
        )
        def register_public_research_lead(
            asset_id: str,
            kind: str,
            name: str,
            source_urls: list[str],
            description: str = "",
        ) -> dict[str, Any]:
            return register_asset(
                {
                    "asset_id": str(asset_id), "kind": str(kind), "name": str(name),
                    "version": "1", "status": "discovered",
                    "source_urls": list(source_urls), "description": str(description),
                    "tested_tasks": [], "reused_tasks": [],
                    "sensors": ["RGB", "RGB-D", "language", "proprioception"],
                    "current_task_data_used": False, "privileged_state_used": False,
                },
                event="public_research_lead_registered",
            )
        registry.tool(
            name="find_capability_assets",
            description="Find reusable capability assets by kind, sensor, or prior task.",
        )(find_assets)
        registry.tool(
            name="record_capability_reuse",
            description="Record whether a shared capability was reused on a later task.",
        )(record_asset_reuse)
        registry.tool(
            name="consult_external_model",
            description=(
                "Ask an external LLM for open-web research leads. Responses "
                "are unverified hypotheses and cannot directly select actions."
            ),
        )(consult_external_model)
        register_asset_provenance_tool(registry)
        register_self_evolution_tool(
            registry,
            controller,
            state_path=state_path,
            integrate_fn=integrate_fn,
            retry_fn=retry_fn,
        )
        if include_controller_tools:
            if controller_graph_workspace is not None:
                if stage_node_workspace is None:
                    raise ValueError(
                        "controller_graph_workspace requires stage_node_workspace"
                    )
                node_store = StageNodeWorkspace(
                    stage_node_workspace,
                    capability_workspace=capability_workspace,
                )
                graph_store = ControllerGraphWorkspace(
                    controller_graph_workspace, nodes=node_store,
                    required_revision=required_graph_revision,
                    require_checkpoint_success=True,
                    available_initial_context_fields={"task_instruction"},
                )
                register_stage_node_tools(registry, node_store)
                register_controller_graph_tools(
                    registry, graph_store, executor=controller_graph_executor,
                )
            elif controller_program_workspace is not None:
                register_controller_program_tools(
                    registry,
                    ControllerProgramWorkspace(
                        controller_program_workspace,
                        python="/data/zxy/envs/vla-report/bin/python",
                        timeout_sec=controller_timeout_sec,
                        capability_workspace=capability_workspace,
                        required_revision=required_controller_revision,
                    ),
                    executor=controller_program_executor,
                )
            else:
                register_controller_authoring_tools(
                    registry,
                    ControllerWorkspace(
                        controller_workspace
                        if controller_workspace is not None
                        else Path("generated_controllers"),
                        timeout_sec=controller_timeout_sec,
                        max_executions=max_controller_executions,
                        capability_workspace=capability_store,
                    ),
                )
        if capability_store is not None:
            register_capability_workspace_tools(registry, capability_store)
        if engineering_workspace is not None:
            register_agent_engineering_tools(
                registry,
                AgentEngineeringWorkspace(
                    engineering_workspace, read_roots=engineering_read_roots,
                ),
            )

    register.controller = controller  # type: ignore[attr-defined]
    return register


__all__ = ["make_frontier_registrar"]
