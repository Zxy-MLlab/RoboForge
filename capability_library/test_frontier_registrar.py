from frontier_registrar import make_frontier_registrar


def test_registrar_registers_capability_market(tmp_path):
    class Registry:
        def __init__(self):
            self.names = []

        def tool(self, **kwargs):
            def decorate(fn):
                self.names.append(kwargs.get("name") or fn.__name__)
                return fn

            return decorate

        def get(self, name):
            return object() if name == "self_evolve_from_failure" else None

    registrar = make_frontier_registrar(
        ["libero_object:task_3"],
        ledger_path=str(tmp_path / "events.jsonl"),
        state_path=str(tmp_path / "state.json"),
    )
    registry = Registry()
    registrar(registry)
    assert set(registry.names) == {
        "search_public_embodied_resources",
        "record_capability_acquisition_event",
        "check_asset_provenance",
        "self_evolve_from_failure",
        "consult_external_model",
            "register_capability_asset",
            "register_public_research_lead",
        "find_capability_assets",
            "record_capability_reuse",
            "create_controller_script",
            "execute_controller_script",
            "inspect_controller_script",
            "inspect_controller_run",
        }
    assert registrar.controller.current_tasks == ("libero_object:task_3",)


def test_acquisition_registrar_does_not_expose_controller_execution(tmp_path):
    class Registry:
        def __init__(self):
            self.names = []

        def tool(self, **kwargs):
            def decorate(fn):
                self.names.append(kwargs.get("name") or fn.__name__)
                return fn
            return decorate

        def get(self, name):
            return object() if name == "self_evolve_from_failure" else None

    registrar = make_frontier_registrar(
        ["libero_spatial:task_5"],
        ledger_path=str(tmp_path / "events.jsonl"),
        state_path=str(tmp_path / "state.json"),
        capability_workspace=tmp_path / "capabilities",
        include_controller_tools=False,
    )
    registry = Registry()
    registrar(registry)
    assert "create_capability_tool" in registry.names
    assert "test_capability_hook" in registry.names
    assert "create_controller_script" not in registry.names
    assert "execute_controller_script" not in registry.names


def test_graph_registrar_exposes_nodes_and_graph_not_monolithic_program(tmp_path):
    class Registry:
        def __init__(self):
            self.names = []

        def tool(self, **kwargs):
            def decorate(fn):
                self.names.append(kwargs.get("name") or fn.__name__)
                return fn
            return decorate

        def get(self, name):
            return object() if name == "self_evolve_from_failure" else None

    registrar = make_frontier_registrar(
        ["libero_spatial:task_4"],
        ledger_path=str(tmp_path / "events.jsonl"),
        state_path=str(tmp_path / "state.json"),
        capability_workspace=tmp_path / "capabilities",
        stage_node_workspace=tmp_path / "nodes",
        controller_graph_workspace=tmp_path / "graphs",
        controller_graph_executor=lambda graph_id: {
            "graph_id": graph_id, "sensor_evidence": {},
        },
    )
    registry = Registry()
    registrar(registry)
    assert {
        "create_stage_node", "inspect_stage_node",
        "create_controller_graph", "inspect_controller_graph",
        "preflight_controller_graph", "execute_controller_graph",
    }.issubset(registry.names)
    assert "create_controller_program" not in registry.names
    assert "create_controller_script" not in registry.names


def test_graph_registrar_schemas_are_accepted_by_real_thea_registry(tmp_path):
    from harness.tools.registry import ToolRegistry

    registrar = make_frontier_registrar(
        ["libero_spatial:task_4"],
        ledger_path=str(tmp_path / "events.jsonl"),
        state_path=str(tmp_path / "state.json"),
        capability_workspace=tmp_path / "capabilities",
        stage_node_workspace=tmp_path / "nodes",
        controller_graph_workspace=tmp_path / "graphs",
        controller_graph_executor=lambda graph_id: {
            "graph_id": graph_id, "sensor_evidence": {},
        },
    )
    registry = ToolRegistry()
    registrar(registry)
    assert registry.get("create_stage_node") is not None
    assert registry.get("create_controller_graph") is not None
    assert registry.get("preflight_controller_graph") is not None
    assert registry.get("execute_controller_graph") is not None
