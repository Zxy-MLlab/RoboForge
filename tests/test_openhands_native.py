from pathlib import Path
from roboforge.assets import AssetLibrary
from roboforge.fakes import FakeAdapter
from roboforge.service import ExperimentService, IndeterminateExperiment
from roboforge.capability import CapabilityAcquirer

def test_physical_trial_is_immutable_idempotent_and_diagnostic_is_separate(tmp_path):
    controller = tmp_path / "controller.py"; controller.write_text("def run(robot): return {}\n")
    adapter = FakeAdapter(); service = ExperimentService(tmp_path / "run", adapter, max_trials=2)
    diagnostic = service.observe(request_id="observe-1")
    first = service.run_controller(request_id="trial-1", controller_path=controller, intent="baseline")
    again = service.run_controller(request_id="trial-1", controller_path=controller, intent="baseline")
    assert diagnostic.execution_kind == "diagnostic"
    assert first.evidence_sha256 == again.evidence_sha256
    assert adapter.controller_runs == 1 and service.status()["physical_trials"] == 1

def test_crash_after_reservation_never_replays_physical_action(tmp_path):
    controller = tmp_path / "controller.py"; controller.write_text("def run(robot): return {}\n")
    def crash(point):
        if point == "physical_reserved": raise RuntimeError("power loss")
    service = ExperimentService(tmp_path / "run", FakeAdapter(), crash_hook=crash)
    try: service.run_controller(request_id="trial", controller_path=controller, intent="test")
    except RuntimeError: pass
    else: assert False
    restored = ExperimentService(tmp_path / "run", FakeAdapter())
    try: restored.run_controller(request_id="trial", controller_path=controller, intent="test")
    except IndeterminateExperiment: pass
    else: assert False
    try: restored.run_controller(request_id="different-tool-call", controller_path=controller, intent="test")
    except IndeterminateExperiment: pass
    else: assert False

def test_assets_are_progressively_disclosed_and_evidence_backed(tmp_path):
    lib = AssetLibrary(tmp_path / "assets")
    summary = lib.register("experiences", name="stable approach", purpose="reuse approach",
        description="Evidence-backed approach behavior", evidence=["experiment://physical-000001"],
        provenance={"task": "A"}, implementation={"secret_detail": 1})
    assert "implementation" not in summary
    found = lib.search("approach")
    assert len(found) == 1 and "implementation" not in found[0]
    assert lib.read(summary["asset_id"])["implementation"] == {"secret_detail": 1}
    assert lib.search("robot stable RGB-D approach", kind="experiences")[0]["asset_id"] == summary["asset_id"]
    capability = lib.register("capabilities", name="point reference extractor",
        purpose="parse nested perception output", description="public JSON helper")
    assert lib.search("object detection handles", kind="capabilities")[0]["asset_id"] == capability["asset_id"]

def test_new_runtime_has_no_generic_agent_loop_implementation():
    root = Path(__file__).parents[1] / "roboforge"
    text = "\n".join(p.read_text() for p in root.glob("*.py"))
    assert "class AgentLoop" not in text
    assert "embodied_codex.kernel.agent_loop" not in text
    cli = (root / "cli.py").read_text()
    for forbidden in ("embodied_codex.providers", "kernel.agent_loop", "kernel.workspace", "context_builder"):
        assert forbidden not in cli
    assert "evaluation" not in text
    assert "class Workspace" not in text
    assert "class AgentLoop" not in text

def test_openhands_editor_allows_capability_modules_but_remains_workspace_confined():
    runtime = (Path(__file__).parents[1] / "roboforge" / "runtime.py").read_text()
    assert "allowed_edits_files" not in runtime
    assert "class ConfinedFileEditorExecutor" in runtime
    assert "relative_to(self.workspace_root)" in runtime

def test_formal_cli_passes_explicit_provider_to_adapter_worker():
    cli = (Path(__file__).parents[1] / "roboforge" / "cli.py").read_text()
    assert 'p.add_argument("--provider"' in cli
    assert '"verifier_provider": provider' in cli
    assert '"--configuration-json"' in cli
    assert "physical_verification.verified=true" in cli
    assert "Decide whether this is useful yourself" in cli

def test_libero_reset_captures_authentic_before_frame_after_reset():
    source = (Path(__file__).parents[1] / "embodied_codex" / "deployments" / "libero.py").read_text()
    reset = source[source.index("    def _reset_to_initial_condition(self):"):
                   source.index("    def resume_protocol(self):")]
    assert 'self._outcome_before = self._capture_outcome_rgb("before")' in reset
    assert reset.index("self.obs = self.env.reset()") < reset.index(
        'self._outcome_before = self._capture_outcome_rgb("before")')

def test_capability_acquisition_validates_and_pins_source(tmp_path):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    source = workspace / "depth.py"
    source.write_text("import sys\nassert '--self-test' in sys.argv\nprint('ok')\n")
    library = AssetLibrary(tmp_path / "assets")
    saved = CapabilityAcquirer(workspace, library).acquire(source_path=str(source),
        name="depth utility", purpose="depth processing", description="validated utility",
        validation_command=["python", "--self-test"], evidence=[], provenance={"url": "local-test"})
    try:
        CapabilityAcquirer(workspace, library).materialize(saved["asset_id"], "capability.py")
    except ValueError as exc: assert "read" in str(exc)
    else: assert False
    detail = library.read(saved["asset_id"])
    assert detail["implementation"]["sha256"]
    assert detail["implementation"]["validation_stdout"] == "ok\n"
    materialized = CapabilityAcquirer(workspace, library).materialize(saved["asset_id"], "capability.py")
    assert materialized["source_sha256"] == detail["implementation"]["sha256"]

def test_capability_acquisition_error_routes_existing_assets_to_materialization(tmp_path):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    outside = tmp_path / "asset.json"; outside.write_text("{}")
    library = AssetLibrary(tmp_path / "assets")
    try:
        CapabilityAcquirer(workspace, library).acquire(source_path=str(outside),
            name="wrong source", purpose="test", description="test",
            validation_command=["python"], evidence=[], provenance={})
    except ValueError as exc:
        assert "read_asset then materialize_capability" in str(exc)
    else: assert False

def test_compare_trials_includes_controller_diff(tmp_path):
    controller = tmp_path / "controller.py"; adapter = FakeAdapter()
    service = ExperimentService(tmp_path / "run", adapter)
    controller.write_text("value = 1\n")
    a = service.run_controller(request_id="a", controller_path=controller, intent="baseline")
    controller.write_text("value = 2\n")
    b = service.run_controller(request_id="b", controller_path=controller, intent="change")
    comparison = service.compare_trials(a.ref, b.ref)
    assert "-value = 1" in comparison["controller_diff"]
    assert "+value = 2" in comparison["controller_diff"]

def test_physical_evidence_binds_declared_asset_use(tmp_path):
    controller = tmp_path / "controller.py"; controller.write_text("value = 1\n")
    service = ExperimentService(tmp_path / "run", FakeAdapter())
    evidence = service.run_controller(request_id="asset-use", controller_path=controller,
        intent="reuse validated perception", assets_used=["experience://abc"])
    assert evidence.assets_used == ("experience://abc",)
    assert service.inspect_trial(evidence.ref).assets_used == ("experience://abc",)

def test_openhands_run_controller_requires_asset_to_be_read(tmp_path):
    import pytest
    pytest.importorskip("openhands.sdk")
    from roboforge.openhands_tools import RunControllerAction, RunControllerExecutor
    workspace = tmp_path / "workspace"; workspace.mkdir()
    controller = workspace / "controller.py"; controller.write_text("value = 1\n")
    library = AssetLibrary(tmp_path / "assets")
    asset = library.register("experiences", name="perception", purpose="locate object",
        description="RGB-D partial", evidence=["experiment://physical-1"])
    service = ExperimentService(tmp_path / "run", FakeAdapter())
    executor = RunControllerExecutor(service, controller, library)
    denied = executor(RunControllerAction(intent="reuse", assets_used=[asset["asset_id"]]))
    assert denied.is_error and service.status()["physical_trials"] == 0
    library.read(asset["asset_id"])
    accepted = executor(RunControllerAction(intent="reuse", assets_used=[asset["asset_id"]]))
    assert not accepted.is_error
    assert accepted.result["assets_used"] == [asset["asset_id"]]

def test_asset_read_provenance_is_conversation_scoped(tmp_path):
    import pytest
    pytest.importorskip("openhands.sdk")
    from types import SimpleNamespace
    from roboforge.asset_tools import ReadAssetAction, ReadAssetExecutor
    from roboforge.openhands_tools import RunControllerAction, RunControllerExecutor
    workspace = tmp_path / "workspace"; workspace.mkdir()
    controller = workspace / "controller.py"; controller.write_text("value = 1\n")
    library = AssetLibrary(tmp_path / "assets")
    asset = library.register("experiences", name="partial", purpose="perception",
        description="RGB-D", evidence=["experiment://physical-1"])
    service = ExperimentService(tmp_path / "run", FakeAdapter())
    class State:
        def active_branch(self): return []
    first = SimpleNamespace(id="conversation-a", state=State())
    second = SimpleNamespace(id="conversation-b", state=State())
    ReadAssetExecutor(library)(ReadAssetAction(asset_id=asset["asset_id"]), first)
    executor = RunControllerExecutor(service, controller, library)
    denied = executor(RunControllerAction(intent="reuse", assets_used=[asset["asset_id"]]), second)
    assert denied.is_error and service.status()["physical_trials"] == 0
    accepted = executor(RunControllerAction(intent="reuse", assets_used=[asset["asset_id"]]), first)
    assert not accepted.is_error
