from pathlib import Path
import json
import pytest
from roboforge.assets import AssetLibrary
from roboforge.fakes import FakeAdapter
from roboforge.service import ExperimentService, IndeterminateExperiment
from roboforge.capability import CapabilityAcquirer
from roboforge.control_plane import compare, replay, submit
from roboforge.trust import sign_receipt, verify_receipt

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

def test_capability_promotion_is_external_and_requires_verified_evidence(tmp_path):
    import time
    library = AssetLibrary(tmp_path / "assets")
    candidate = library.register("capabilities", name="depth filter", purpose="perception",
        description="Reject invalid depth", implementation={"source": "value = 1"})
    assert candidate["verification_status"] == "candidate"
    controller = tmp_path / "controller.py"; controller.write_text("value = 1\n")
    run = tmp_path / "run"
    adapter = FakeAdapter(); adapter.receipt_verified = True
    evidence = ExperimentService(run, adapter).run_controller(
        request_id="verified", controller_path=controller, intent="validate candidate",
        assets_used=[candidate["asset_id"]])
    evidence_path = next((run / "evidence").glob("*.json"))
    body = json.loads(evidence_path.read_text()); body["sealed_receipt"] = sign_receipt(
        {"trial_id": evidence.ref, "issued_at": time.time(), "success": True}, b"evaluator")
    body.pop("evidence_sha256", None)
    from roboforge.store import canonical_json
    body["evidence_sha256"] = __import__("hashlib").sha256(canonical_json({k:v for k,v in body.items() if k != "evidence_sha256"})).hexdigest()
    evidence_path.write_text(json.dumps(body, indent=2, sort_keys=True))
    promoted = submit(library.root, candidate["asset_id"], [str(evidence_path)],
        note="external contract and physical validation passed", evaluator_key=b"evaluator")
    assert promoted["verification_status"] == "promoted"
    assert promoted["verification_decision"]["evidence"] == [evidence.ref]

    assert library.decide_capability(candidate["asset_id"], decision="promoted",
        evidence=[evidence.ref], note="external contract and physical validation passed") == promoted
    with pytest.raises(ValueError, match="immutable"):
        library.decide_capability(candidate["asset_id"], decision="rejected",
            evidence=[evidence.ref], note="conflicting terminal decision")

def test_capability_promotion_rejects_negative_physical_evidence(tmp_path):
    library = AssetLibrary(tmp_path / "assets")
    candidate = library.register("capabilities", name="unsafe filter", purpose="perception",
        description="Unverified candidate", implementation={"source": "value = 1"})
    controller = tmp_path / "controller.py"; controller.write_text("value = 1\n")
    run = tmp_path / "run"
    evidence = ExperimentService(run, FakeAdapter()).run_controller(
        request_id="negative", controller_path=controller, intent="negative validation")
    evidence_path = next((run / "evidence").glob("*.json"))
    with pytest.raises(ValueError, match="independently verified physical evidence"):
        submit(library.root, candidate["asset_id"], [str(evidence_path)], note="must fail")
    assert library.read(candidate["asset_id"], session_id="test")["verification_status"] == "candidate"

def test_signed_receipt_rejects_tamper_and_replay(tmp_path):
    import time
    key = b"evaluator-only-test-key"
    receipt = sign_receipt({"trial_id": "trial-1", "task": "8", "success": True,
                            "issued_at": time.time()}, key)
    assert verify_receipt(receipt, key)
    altered = dict(receipt); altered["success"] = False
    assert not verify_receipt(altered, key)
    expired = sign_receipt({"trial_id": "trial-1", "issued_at": 1.0}, key)
    assert not verify_receipt(expired, key, now=5000.0)

def test_capability_decision_does_not_modify_content_addressed_object(tmp_path):
    library = AssetLibrary(tmp_path / "assets")
    candidate = library.register("capabilities", name="immutable", purpose="test", description="CAS")
    path = tmp_path / "assets" / "capabilities" / (candidate["asset_id"].split("://", 1)[1] + ".json")
    before = path.read_bytes()
    library.decide_capability(candidate["asset_id"], decision="rejected", evidence=["trial://x"], note="failed gate")
    assert path.read_bytes() == before

def test_capability_registration_cannot_bypass_external_gate(tmp_path):
    library = AssetLibrary(tmp_path / "assets")
    with pytest.raises(TypeError, match="verification_status"):
        library.register("capabilities", name="bypass", purpose="test",
            description="must remain a candidate", verification_status="promoted")

def test_lifecycle_cli_parses_frozen_run_request(monkeypatch, tmp_path, capsys):
    from roboforge import cli
    controller = tmp_path / "controller.py"; controller.write_text("value = 1\n")
    captured = {}
    def fake_run(args):
        captured.update(vars(args)); return {"ref": "evidence://frozen"}
    monkeypatch.setattr(cli, "_run_frozen_candidate", fake_run)
    assert cli.main(["run", str(controller), "--runtime", "libero", "--task", "8",
                     "--seed", "51", "--run-dir", str(tmp_path / "run")]) == 0
    assert captured["entrypoint"] == controller
    assert (captured["runtime"], captured["task"], captured["seed"]) == ("libero", "8", 51)
    assert json.loads(capsys.readouterr().out)["ref"] == "evidence://frozen"

def test_replay_never_reexecutes_and_compare_reads_verified_records(tmp_path):
    service = ExperimentService(tmp_path / "run", FakeAdapter())
    controller = tmp_path / "controller.py"; controller.write_text("value = 1\n")
    first = service.run_controller(request_id="first", controller_path=controller, intent="baseline")
    controller.write_text("value = 2\n")
    second = service.run_controller(request_id="second", controller_path=controller, intent="candidate")
    files = sorted((tmp_path / "run" / "evidence").glob("*.json"))
    replayed = replay(files[0])
    assert replayed["physical_action_replayed"] is False
    result = compare(files[0], files[1])
    assert result["baseline"] == first.ref and result["candidate"] == second.ref
    assert any(change["field"] == "controller_sha256" for change in result["changes"])

def test_new_runtime_has_no_generic_agent_loop_implementation():
    root = Path(__file__).parents[1] / "roboforge"
    text = "\n".join(p.read_text() for p in root.glob("*.py"))
    assert "class AgentLoop" not in text
    assert "_handle_content_response" not in text
    assert "embodied_codex.kernel.agent_loop" not in text
    cli = (root / "cli.py").read_text()
    for forbidden in ("embodied_codex.providers", "kernel.agent_loop", "kernel.workspace", "context_builder"):
        assert forbidden not in cli
    assert "evaluation.sealed" not in text
    assert "class Workspace" not in text
    assert "class AgentLoop" not in text

def test_openhands_editor_allows_capability_modules_but_remains_workspace_confined():
    runtime = (Path(__file__).parents[1] / "roboforge" / "runtime.py").read_text()
    assert "allowed_edits_files" not in runtime
    assert "class ConfinedFileEditorExecutor" in runtime
    assert "relative_to(self.workspace_root)" in runtime

def test_formal_cli_passes_explicit_provider_to_adapter_worker():
    cli = (Path(__file__).parents[1] / "roboforge" / "cli.py").read_text()
    assert 'f"--token={token}"' not in cli
    assert 'worker_env["ROBOFORGE_RPC_TOKEN"] = token' in cli
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
