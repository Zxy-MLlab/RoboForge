"""Model-facing authoring API for the standalone Embodied Harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .asset_store import AssetStore
from .graph_store import GraphStore
from .node_store import NodeStore
from .tool_registry import ToolRegistry
from .tool_store import ToolStore
from .web import search_public_web


def _object(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object", "properties": dict(properties),
        "required": required, "additionalProperties": False,
    }


class AuthoringAPI:
    """Narrow engineering surface given to the model for one evolution round.

    Graph compilation always happens before the environment factory is called.
    At most one environment episode can be started in a round.
    """

    def __init__(
        self, *, nodes: NodeStore, graphs: GraphStore, tools: ToolStore,
        assets: AssetStore, adapter_factory: Callable[[], Any],
        artifact_dir: str | Path, require_capability_activity: bool = False,
    ) -> None:
        self.nodes = nodes
        self.graphs = graphs
        self.tools = tools
        self.assets = assets
        self.adapter_factory = adapter_factory
        self.artifact_dir = Path(artifact_dir).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.execution_count = 0
        self.last_execution: dict[str, Any] | None = None
        self.require_capability_activity = bool(require_capability_activity)
        self.capability_activity = 0

    def create_stage_node(self, **kwargs: Any) -> dict[str, Any]:
        return self.nodes.create(**kwargs)

    def inspect_stage_node(self, node_id: str) -> dict[str, Any]:
        return self.nodes.inspect(node_id)

    def create_controller_graph(self, **kwargs: Any) -> dict[str, Any]:
        return self.graphs.create(**kwargs)

    def compile_controller_graph(self, graph_id: str) -> dict[str, Any]:
        return self.graphs.compile(graph_id)

    def execute_controller_graph(self, graph_id: str) -> dict[str, Any]:
        if self.execution_count:
            raise RuntimeError("only one robot episode is allowed per evolution round")
        if self.require_capability_activity and not self.capability_activity:
            raise RuntimeError(
                "repeated failure requires public research or Tool activity before execution"
            )
        # This is intentionally before adapter construction. A malformed graph
        # must not consume or alter a robot episode.
        compile_report = self.graphs.compile(graph_id)
        self.execution_count += 1
        adapter = self.adapter_factory()
        try:
            installed_tools = self.tools.install_runtime_capabilities(adapter)
            execution = self.graphs.execute(graph_id, adapter)
            sensor_report = dict(adapter.sensor_report(execution))
        finally:
            adapter.close()
        result = {
            "graph_id": graph_id, "compile_report": compile_report,
            "installed_runtime_tools": installed_tools,
            "execution": execution, "sensor_report": sensor_report,
        }
        self.last_execution = result
        (self.artifact_dir / "execution.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        return result

    def create_capability_tool(self, **kwargs: Any) -> dict[str, Any]:
        result = self.tools.create(**kwargs)
        self.capability_activity += 1
        return result

    def test_capability_tool(
        self, tool_id: str, cases: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        result = self.tools.test(tool_id, cases)
        self.capability_activity += 1
        return result

    def list_tested_capability_tools(self) -> list[dict[str, Any]]:
        return self.tools.tested()

    def inspect_capability_tool(self, tool_id: str) -> dict[str, Any]:
        return self.tools.inspect(tool_id)

    def inspect_controller_graph(self, graph_id: str) -> dict[str, Any]:
        return self.graphs.inspect(graph_id)

    def search_public_web(self, query: str, limit: int = 5) -> dict[str, Any]:
        result = search_public_web(query, limit=limit)
        self.capability_activity += 1
        self.assets.record("public_web_search", result)
        return result

    def register_research_lead(
        self, title: str, url: str, relevance: str,
    ) -> dict[str, Any]:
        return self.assets.record("research_lead", {
            "title": title, "url": url, "relevance": relevance,
        })

    def record_experience(
        self, observation: str, diagnosis: str, reusable_lesson: str,
    ) -> dict[str, Any]:
        return self.assets.record("experience", {
            "observation": observation, "diagnosis": diagnosis,
            "reusable_lesson": reusable_lesson,
        })

    def registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        string = {"type": "string"}
        array_string = {"type": "array", "items": string}
        free_object = {"type": "object", "additionalProperties": True}
        source_schema = {
            "type": "string",
            "description": (
                "Complete executable Python defining exactly run_stage(adapter, context). "
                "Use only adapter.instruction/sense/act/use/verify/record and return "
                "literal {'outcome': NAME, 'updates': {...}} objects for every outcome."
            ),
        }
        definitions = [
            ("create_stage_node", "Create an immutable typed executable robot stage node. "
             "checkpoint_outcomes must be empty except for sensor verification nodes.",
             _object({
                 "name": string, "kind": string, "description": string,
                 "source": source_schema, "requires": array_string,
                 "provides_by_outcome": {"type": "object", "additionalProperties": array_string},
                 "checkpoint_outcomes": array_string,
             }, ["name", "kind", "description", "source", "requires",
                 "provides_by_outcome", "checkpoint_outcomes"]), self.create_stage_node),
            ("inspect_stage_node", "Inspect an immutable stage node.",
             _object({"node_id": string}, ["node_id"]), self.inspect_stage_node),
            ("create_controller_graph", "Create and compile an immutable controller graph.",
             _object({
                 "name": string, "description": string, "entry": string,
                 "bindings": {"type": "object", "additionalProperties": string},
                 "edges": {"type": "array", "items": free_object},
                 "initial_fields": array_string, "max_visits": {"type": "integer"},
                 "base_graph_id": {"type": ["string", "null"]},
                 "frozen_aliases": array_string,
             }, ["name", "description", "entry", "bindings", "edges",
                 "initial_fields", "max_visits", "base_graph_id", "frozen_aliases"]),
             self.create_controller_graph),
            ("compile_controller_graph", "Compile without launching a robot.",
             _object({"graph_id": string}, ["graph_id"]), self.compile_controller_graph),
            ("execute_controller_graph", "Execute one compiled graph in this round.",
             _object({"graph_id": string}, ["graph_id"]), self.execute_controller_graph),
            ("inspect_controller_graph", "Inspect a controller graph.",
             _object({"graph_id": string}, ["graph_id"]), self.inspect_controller_graph),
            ("create_capability_tool", "Create a versioned reusable capability Tool.",
             _object({
                 "name": string, "description": string, "source": string,
                 "input_schema": free_object, "output_schema": free_object,
                 "source_urls": array_string,
             }, ["name", "description", "source", "input_schema",
                 "output_schema", "source_urls"]), self.create_capability_tool),
            ("test_capability_tool", "Test a Tool before it can be reused.",
             _object({"tool_id": string, "cases": {"type": "array", "items": free_object}},
                     ["tool_id", "cases"]), self.test_capability_tool),
            ("list_tested_capability_tools", "List reusable tested Tools.",
             _object({}, []), self.list_tested_capability_tools),
            ("inspect_capability_tool", "Inspect Tool source and provenance.",
             _object({"tool_id": string}, ["tool_id"]), self.inspect_capability_tool),
            ("search_public_web", "Search public internet resources.",
             _object({"query": string, "limit": {"type": "integer"}}, ["query", "limit"]),
             self.search_public_web),
            ("register_research_lead", "Save a relevant public resource.",
             _object({"title": string, "url": string, "relevance": string},
                     ["title", "url", "relevance"]), self.register_research_lead),
            ("record_experience", "Save a reusable sensor-grounded lesson.",
             _object({"observation": string, "diagnosis": string,
                      "reusable_lesson": string},
                     ["observation", "diagnosis", "reusable_lesson"]),
             self.record_experience),
        ]
        for name, description, parameters, function in definitions:
            registry.register(name=name, description=description,
                              parameters=parameters, function=function)
        return registry


__all__ = ["AuthoringAPI"]
