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
    with pytest.raises(RuntimeError, match="external promotion service"):
        submit(library.root, candidate["asset_id"], [str(evidence_path)],
            note="external contract and physical validation passed", evaluator_key=b"evaluator")
    assert library.read(candidate["asset_id"], session_id="test")["verification_status"] == "candidate"

def test_agent_cannot_enable_trusted_promotion(monkeypatch, tmp_path):
    library = AssetLibrary(tmp_path / "assets")
    candidate = library.register("capabilities", name="attack", purpose="test", description="attack")
    monkeypatch.setenv("ROBOFORGE_TRUSTED_MODE", "1")
    for kwargs in ({}, {"require_trusted_mode": False},
                   {"require_trusted_mode": True, "evaluator_key": b"attacker-key"}):
        with pytest.raises(RuntimeError, match="external promotion service"):
            submit(library.root, candidate["asset_id"], [], note="forged", **kwargs)
    assert library.read(candidate["asset_id"], session_id="attack")["verification_status"] == "candidate"

def test_capability_promotion_rejects_negative_physical_evidence(tmp_path):
    library = AssetLibrary(tmp_path / "assets")
    candidate = library.register("capabilities", name="unsafe filter", purpose="perception",
        description="Unverified candidate", implementation={"source": "value = 1"})
    controller = tmp_path / "controller.py"; controller.write_text("value = 1\n")
    run = tmp_path / "run"
    evidence = ExperimentService(run, FakeAdapter()).run_controller(
        request_id="negative", controller_path=controller, intent="negative validation")
    evidence_path = next((run / "evidence").glob("*.json"))
    with pytest.raises(RuntimeError, match="external promotion service"):
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


def test_distribution_exposes_only_openhands_native_agent_cli():
    root = Path(__file__).parents[1]
    project = (root / "pyproject.toml").read_text()
    assert 'roboforge = "roboforge.cli:main"' in project
    assert 'roboforge-openhands = "roboforge.cli:main"' in project
    assert 'embodied_codex = "embodied_codex.cli:main"' not in project
    public_api = (root / "embodied_codex" / "__init__.py").read_text()
    assert "AgentLoop" not in public_api
    compatibility_main = (root / "embodied_codex" / "__main__.py").read_text()
    assert "Deprecated source-checkout compatibility entry point" in compatibility_main
    assert "from .cli import main" in compatibility_main
    assert '"openhands-sdk==1.44.1"' in project
    assert '"openhands-tools==1.44.1"' in project

def test_openhands_editor_allows_capability_modules_but_remains_workspace_confined():
    runtime = (Path(__file__).parents[1] / "roboforge" / "runtime.py").read_text()
    assert "allowed_edits_files" not in runtime
    assert "class ConfinedFileEditorExecutor" in runtime
    assert "relative_to(self.workspace_root)" in runtime


def test_openhands_editor_views_workspace_images_as_multimodal_content(tmp_path):
    pytest.importorskip("openhands.sdk")
    import base64

    from openhands.sdk.llm import ImageContent
    from openhands.tools.file_editor import FileEditorAction
    from roboforge.runtime import ConfinedFileEditorExecutor

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "keyframe.png"
    # Valid 1x1 transparent PNG; the upstream editor forwards image bytes
    # without requiring Pillow or another image decoder in the OH venv.
    image.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4"
        "z8DwHwAFgAI/ScLwWQAAAABJRU5ErkJggg=="
    ))

    executor = ConfinedFileEditorExecutor(workspace_root=str(workspace))
    observation = executor(FileEditorAction(command="view", path=str(image)))

    assert not observation.is_error
    assert any(isinstance(item, ImageContent) for item in observation.content)
    assert any(
        url.startswith("data:image/png;base64,")
        for item in observation.content
        if isinstance(item, ImageContent)
        for url in item.image_urls
    )


def test_openhands_conversation_uses_official_default_condenser(tmp_path):
    pytest.importorskip("openhands.sdk")
    from openhands.sdk import LLM
    from openhands.sdk.context.condenser import LLMSummarizingCondenser
    from roboforge import create_openhands_conversation

    workspace = tmp_path / "workspace"
    persistence = tmp_path / "openhands"
    workspace.mkdir()
    controller = workspace / "controller.py"
    controller.write_text("def run(robot): return robot.observe()\n")
    llm = LLM(model="gpt-5", api_key="test-only", usage_id="campaign")

    conversation = create_openhands_conversation(
        llm=llm,
        workspace=workspace,
        persistence_dir=persistence,
        service=ExperimentService(tmp_path / "run", FakeAdapter()),
        controller_path=controller,
    )

    condenser = conversation.agent.condenser
    assert isinstance(condenser, LLMSummarizingCondenser)
    assert condenser.max_size == 80
    assert condenser.keep_first == 4
    assert condenser.llm.usage_id == "condenser"
    assert conversation.agent.llm.usage_id == "campaign"


def test_formal_tools_use_public_planning_and_subagent_extensions(tmp_path):
    pytest.importorskip("openhands.sdk")
    from roboforge.runtime import register_spike_tools

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    controller = workspace / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    tools = register_spike_tools(
        ExperimentService(tmp_path / "run", FakeAdapter()),
        workspace=workspace,
        controller_path=controller,
    )
    by_name = {tool.name: tool for tool in tools}
    assert "planning_file_editor" in by_name
    assert by_name["planning_file_editor"].params == {
        "plan_path": str(workspace / "PLAN.md")
    }
    assert "task_tool_set" in by_name
    assert "browser_tool_set" not in by_name
    for embodied_name in ("observe", "run_controller", "inspect_trial", "compare_trials"):
        assert embodied_name not in by_name

def test_formal_cli_passes_explicit_provider_to_adapter_worker():
    cli = (Path(__file__).parents[1] / "roboforge" / "cli.py").read_text()
    assert 'f"--token={token}"' not in cli
    assert "worker_env = _runtime_worker_env(token)" in cli
    assert "os.environ.copy()" not in cli
    assert 'p.add_argument("--provider"' in cli
    assert '"verifier_provider": provider' in cli
    assert '"--configuration-json"' in cli
    assert "physical_verification.verified=true" in cli
    assert "there is no Agent-side" in cli
    assert "acquire_capability" not in cli


def test_runtime_worker_environment_does_not_inherit_unrelated_secrets(monkeypatch):
    from roboforge.cli import _runtime_worker_env

    monkeypatch.setenv("ROBOFORGE_PROMOTION_TOKEN", "must-not-cross")
    monkeypatch.setenv("APEX_API_KEY", "model-secret")
    monkeypatch.setenv("LIBERO_CONFIG_PATH", "/safe/config")
    env = _runtime_worker_env("rpc-token")
    assert env["ROBOFORGE_RPC_TOKEN"] == "rpc-token"
    assert env["LIBERO_CONFIG_PATH"] == "/safe/config"
    assert "ROBOFORGE_PROMOTION_TOKEN" not in env
    assert "APEX_API_KEY" not in env


def test_canonical_workspace_layout(tmp_path):
    from roboforge.cli import _initialize_persistent_workspace

    workspace = tmp_path / "workspace"
    _initialize_persistent_workspace(workspace)
    expected = {
        "controllers", "capabilities/perception", "capabilities/grasping",
        "capabilities/planning", "capabilities/control", "models", "services",
        "robot_sdk", "runtime_adapters", "experiments", "diagnostics", "tests",
        "configs", "requirements", "task_docs",
    }
    assert all((workspace / path).is_dir() for path in expected)

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


def test_public_evidence_artifacts_materialize_inside_workspace(tmp_path):
    pytest.importorskip("openhands.sdk")
    from roboforge.openhands_tools import materialize_public_evidence

    service = ExperimentService(tmp_path / "run", FakeAdapter())
    evidence = service.observe(request_id="materialize")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    first = materialize_public_evidence(service, evidence, workspace)
    artifact = first["artifacts"][0]
    target = Path(artifact["local_path"])
    before_mtime = target.stat().st_mtime_ns
    second = materialize_public_evidence(service, evidence, workspace)

    assert target.is_file()
    assert target.resolve().is_relative_to(workspace.resolve())
    assert target.read_bytes() == b"diagnostic-1"
    assert __import__("hashlib").sha256(target.read_bytes()).hexdigest() == artifact["sha256"]
    assert second["artifacts"][0]["local_path"] == str(target)
    assert target.stat().st_mtime_ns == before_mtime
    assert "local_path" not in json.dumps(evidence.public_dict())


def test_materialized_artifact_name_cannot_escape_workspace(tmp_path):
    pytest.importorskip("openhands.sdk")
    from dataclasses import replace
    from roboforge.openhands_tools import materialize_public_evidence

    service = ExperimentService(tmp_path / "run", FakeAdapter())
    evidence = service.observe(request_id="traversal")
    malicious = replace(
        evidence,
        artifacts=(replace(evidence.artifacts[0], name="../../outside.png"),),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    body = materialize_public_evidence(service, malicious, workspace)

    target = Path(body["artifacts"][0]["local_path"])
    assert target.name == "outside.png"
    assert target.resolve().is_relative_to(workspace.resolve())
    assert not (tmp_path / "outside.png").exists()


def test_openhands_error_observation_is_structured_and_sanitized():
    pytest.importorskip("openhands.sdk")
    from roboforge.openhands_tools import _error

    observation = _error(
        "run_controller",
        RuntimeError("failed at /root/private/controller.py token=attacker-value"),
    )

    assert observation.is_error is True
    assert observation.result["tool_error"]["type"] == "RuntimeError"
    encoded = json.dumps(observation.result)
    assert "/root/private" not in encoded
    assert "attacker-value" not in encoded
    assert "<redacted-path>" in encoded and "token=<redacted>" in encoded


def test_canonical_path_uses_terminal_trial_cli():
    cli = (Path(__file__).parents[1] / "roboforge" / "cli.py").read_text()
    assert "python -m roboforge trial" in cli
    assert "_run_with_failure_feedback" not in cli

def _legacy_feedback_test_removed(tmp_path):
    pytest.importorskip("openhands.sdk")
    from types import SimpleNamespace

    from roboforge.cli import _run_with_failure_feedback

    service = ExperimentService(tmp_path / "run", FakeAdapter(), max_trials=3)
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")

    class Conversation:
        def __init__(self):
            self.run_calls = 0
            self.messages = []
            self.state = SimpleNamespace(events=[])

        def run(self):
            self.run_calls += 1
            if self.run_calls == 1:
                service.run_controller(
                    request_id="failed-trial",
                    controller_path=controller,
                    intent="exercise feedback continuation",
                )
            else:
                self.state.events.extend(["action", "observation"])

        def send_message(self, message, sender=None):
            self.messages.append((message, sender))

    conversation = Conversation()
    _run_with_failure_feedback(conversation, service, tmp_path / "workspace")

    assert conversation.run_calls == 2
    assert len(conversation.messages) == 1
    message, sender = conversation.messages[0]
    assert sender == "roboforge"
    assert "physical_verification" in message
    assert "local_path" in message
    assert "hidden success-evaluator state" in message.lower()


def _legacy_noop_feedback_test_removed(tmp_path):
    pytest.importorskip("openhands.sdk")
    from types import SimpleNamespace

    from roboforge.cli import _run_with_failure_feedback

    service = ExperimentService(tmp_path / "run", FakeAdapter(), max_trials=3)

    class Conversation:
        def __init__(self):
            self.run_calls = 0
            self.messages = []
            self.state = SimpleNamespace(events=[])

        def run(self):
            self.run_calls += 1
            if self.run_calls == 1:
                # OpenHands acknowledged the task in prose but emitted no
                # ActionEvent or physical trial.
                self.state.events.append("assistant-message")
            else:
                # A real tool action is progress; the helper must hand control
                # back instead of prescribing the next development step.
                self.state.events.extend(["action", "observation"])

        def send_message(self, message, sender=None):
            self.messages.append((message, sender))

    conversation = Conversation()
    _run_with_failure_feedback(conversation, service, tmp_path / "workspace")

    assert conversation.run_calls == 2
    assert len(conversation.messages) == 1
    message, sender = conversation.messages[0]
    assert sender == "roboforge"
    assert "did not invoke a tool" in message
    assert "immediately use observe" in message


def test_stop_gate_denies_while_trials_remain(tmp_path):
    from roboforge.stop_gate import stop_decision

    status = tmp_path / "status.json"
    status.write_text(json.dumps({
        "physical_trials": 1,
        "max_physical_trials": 3,
        "latest_physical_evidence": "experiment://physical-000001",
        "latest_verified": False,
    }))
    decision = stop_decision(status)
    assert decision["decision"] == "deny"
    assert "not yet verified" in decision["reason"]
    assert "experiment://physical-000001" in decision["additionalContext"]


def test_stop_gate_allows_verified_or_exhausted_and_denies_missing(tmp_path):
    from roboforge.stop_gate import stop_decision

    missing = stop_decision(tmp_path / "missing.json")
    assert missing["decision"] == "deny"

    verified = tmp_path / "verified.json"
    verified.write_text(json.dumps({
        "physical_trials": 1, "max_physical_trials": 3,
        "latest_verified": True,
    }))
    assert stop_decision(verified)["decision"] == "allow"

    exhausted = tmp_path / "exhausted.json"
    exhausted.write_text(json.dumps({
        "physical_trials": 3, "max_physical_trials": 3,
        "latest_verified": False,
    }))
    assert stop_decision(exhausted)["decision"] == "allow"


def test_execution_task_gate_requires_local_terminal_trial(tmp_path):
    from roboforge.stop_gate import execution_task_stop_decision, write_tool_activity

    activity = write_tool_activity(tmp_path, 0)
    assert execution_task_stop_decision(activity, tmp_path)["decision"] == "deny"
    write_tool_activity(tmp_path, 1)
    assert execution_task_stop_decision(activity, tmp_path)["decision"] == "deny"
    failed = tmp_path / ".roboforge" / "trials" / "preflight-invalid"; failed.mkdir(parents=True)
    (failed / "result.json").write_text(json.dumps({"trial_id": failed.name, "physical_trial_consumed": False}))
    assert "non-physical" in execution_task_stop_decision(activity, tmp_path)["reason"]
    physical = tmp_path / ".roboforge" / "trials" / "physical-000001"; physical.mkdir(parents=True)
    (physical / "result.json").write_text(json.dumps({"trial_id": physical.name, "physical_trial_consumed": True,
                                                       "task_success": False}))
    denied = execution_task_stop_decision(activity, tmp_path)
    assert denied["decision"] == "deny"
    assert "not verified" in denied["reason"]
    status = tmp_path / ".roboforge" / "campaign-status.json"
    status.write_text(json.dumps({"physical_trials": 1, "max_physical_trials": 2,
                                  "latest_verified": True}))
    assert execution_task_stop_decision(activity, tmp_path)["decision"] == "allow"
    status.write_text(json.dumps({"physical_trials": 2, "max_physical_trials": 2,
                                  "latest_verified": False}))
    assert execution_task_stop_decision(activity, tmp_path)["decision"] == "allow"


def _legacy_stop_feedback_test_removed(tmp_path):
    pytest.importorskip("openhands.sdk")
    from roboforge.cli import _run_with_failure_feedback

    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")

    for verified, max_trials in ((True, 2), (False, 1)):
        adapter = FakeAdapter()
        adapter.receipt_verified = verified
        service = ExperimentService(
            tmp_path / f"run-{verified}-{max_trials}",
            adapter,
            max_trials=max_trials,
        )

        class Conversation:
            run_calls = 0
            messages = []

            def run(self):
                self.run_calls += 1
                service.run_controller(
                    request_id="trial",
                    controller_path=controller,
                    intent="terminal condition",
                )

            def send_message(self, message, sender=None):
                self.messages.append((message, sender))

        conversation = Conversation()
        _run_with_failure_feedback(conversation, service, tmp_path / "workspace")
        assert conversation.run_calls == 1
        assert conversation.messages == []


def test_service_sanitizes_embedded_paths_and_secrets_without_mutating_adapter_data(
    tmp_path,
):
    from copy import deepcopy
    from roboforge.models import AdapterResult

    class Adapter(FakeAdapter):
        def observe(self):
            return AdapterResult(
                public={
                    "message": (
                        "failed at /root/private/controller.py "
                        "token=attacker-token"
                    ),
                    "nested": {
                        "api_key": "attacker-key",
                        "detail": "log: /tmp/private/runtime.log",
                    },
                }
            )

    adapter = Adapter()
    original = adapter.observe().public
    expected = deepcopy(original)
    adapter.observe = lambda: AdapterResult(public=original)

    evidence = ExperimentService(tmp_path / "run", adapter).observe(
        request_id="sanitize"
    )
    encoded = json.dumps(evidence.public, sort_keys=True)

    assert original == expected
    assert "/root/private" not in encoded and "/tmp/private" not in encoded
    assert "attacker-token" not in encoded
    assert "attacker-key" not in encoded
    assert "<redacted-path>" in encoded
