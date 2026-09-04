import hashlib
import json
from pathlib import Path

import pytest

from embodied_codex import cli
from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.legacy.agent_loop import AgentLoop, ProtocolError
from embodied_codex.kernel.capability_manager import CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.recovery import save_checkpoint
from embodied_codex.kernel.workspace import PersistentWorkspace
from evaluation.frozen_runner import (FrozenDependencyResolver,
    FrozenEvaluationError, FrozenEvaluationRunner)
from evaluation.run_embodied_codex_libero import _sealed_status


class SealedCase:
    sdk_index = {"operations": ["observe", "act", "use", "verify"]}
    instruction = "sealed"

    def __init__(self, success=False, native=None):
        self.success = success; self.native = dict(native or {})
        self.reset_count = 0; self.sealed = False; self.bound = []

    def reset_case(self): self.reset_count += 1
    def begin_controller_execution(self): self.sealed = False
    def seal_controller_execution(self): self.sealed = True
    def hidden_evaluator(self, execution):
        assert self.sealed
        return self.success
    def verification_receipt(self, execution):
        return {"verified": not self.success,
                "controller_sha256": execution["program_sha256"]}
    def native_capability_manifest(self): return self.native
    def register_capability(self, tool_id, function, manifest):
        self.bound.append((tool_id, function, manifest))


class Runtime:
    def execute(self, controller, adapter):
        return {"completed": True, "program_sha256": sha(controller)}


class ToolLibrary:
    def __init__(self, manifest): self.manifest = manifest
    def inspect(self, tool_id):
        if tool_id != self.manifest["tool_id"]: raise FileNotFoundError(tool_id)
        return {"manifest": dict(self.manifest)}
    def runtime_function(self, tool_id, *, artifact_resolver=None): return lambda payload: payload


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_bundle(controller, sealed_cases):
    partition = {"protocol": "test-partition-v1", "seed": 1,
        "development_cases": ["development"],
        "sealed_cases": [str(value) for value in sealed_cases]}
    partition["digest"] = hashlib.sha256(json.dumps(partition, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    payload = {"protocol": "roboforge-evaluation-bundle-v1",
        "controller_sha256": sha(controller), "observed_tool_dependencies": [],
        "dependency_closure": [], "adapter_requirements": {"capabilities": []},
        "development_provenance": [{"sha256": "a" * 64}],
        "development_partition": partition, "source_skill_id": None}
    payload["bundle_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return payload


@pytest.mark.parametrize("successes", range(4))
def test_frozen_runner_aggregates_each_hidden_evaluator_case(tmp_path, successes):
    controller = tmp_path / "controller.py"; controller.write_text("def run(robot): return {}\n")
    cases = [(str(i), SealedCase(i < successes)) for i in range(3)]
    bundle = evaluation_bundle(controller, range(3))
    result = FrozenEvaluationRunner(cases=cases, runtime=Runtime(), controller=controller,
        expected_sha256=sha(controller), resolver=FrozenDependencyResolver(
            bundle=bundle, tool_library=None), bundle=bundle).run()
    assert result["episodes"] == 3
    assert result["evaluator_successes"] == successes
    assert result["success_rate"] == successes / 3
    assert result["controller_sha256"] == sha(controller)
    assert [row["evaluator_success"] for row in result["sealed_evaluation_cases"]] == [
        i < successes for i in range(3)]
    assert [row["name"] for row in result["evaluation_policies"]] == [
        "frozen_controller", "provenance", "anti_cheating", "generalization",
        "sealed_evaluation"]
    # The receipt deliberately reports the opposite verdict: hidden truth wins.
    assert all(case.reset_count == 1 for _, case in cases)


def test_frozen_dependency_restores_exact_promoted_tool_and_native_capability():
    runtime = {"runtime_id": "sha256-runtime"}
    manifest = {"tool_id": "shared:v003", "version": 3, "status": "promoted",
        "source_sha256": "a" * 64, "test_receipt_sha256": "b" * 64,
        "runtime_environment": runtime}
    native = {"libero.seed:v001": {"capability_id": "libero.seed:v001",
        "version": "v001", "contract_sha256": "c" * 64}}
    skill = {"dependency_closure": [{key: manifest[key] for key in
        ("tool_id", "version", "source_sha256", "test_receipt_sha256", "runtime_environment")}],
        "adapter_requirements": {"capabilities": [native["libero.seed:v001"]]}}
    case = SealedCase(native=native)
    result = FrozenDependencyResolver(skill=skill,
        tool_library=ToolLibrary(manifest)).restore(case)
    assert result == {"shared_tools": ["shared:v003"],
                      "adapter_capabilities": ["libero.seed:v001"]}
    assert [row[0] for row in case.bound] == ["shared:v003"]


def test_frozen_dependency_mismatch_and_missing_native_fail_closed():
    manifest = {"tool_id": "shared:v001", "version": 1, "status": "promoted",
        "source_sha256": "a", "test_receipt_sha256": "b", "runtime_environment": None}
    skill = {"dependency_closure": [{**manifest, "source_sha256": "wrong"}],
             "adapter_requirements": {"capabilities": [{"capability_id": "seed:v001"}]}}
    with pytest.raises(FrozenEvaluationError, match="dependency mismatch"):
        FrozenDependencyResolver(skill=skill, tool_library=ToolLibrary(manifest)).restore(SealedCase())
    skill["dependency_closure"] = []
    with pytest.raises(FrozenEvaluationError, match="missing Adapter-native"):
        FrozenDependencyResolver(skill=skill, tool_library=None).restore(SealedCase())


def test_sealed_campaign_status_consumes_canonical_case_results(tmp_path):
    rows = [{"case": str(state), "evaluator_success": index < 2,
             "controller_sha256": "f" * 64}
            for index, state in enumerate((4, 5, 6))]
    partition = {"protocol": "partition-v1", "seed": 1,
        "development_cases": ["0"], "sealed_cases": ["4", "5", "6"]}
    partition["digest"] = hashlib.sha256(json.dumps(partition, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    bundle = {"protocol": "roboforge-evaluation-bundle-v1",
        "controller_sha256": "f" * 64, "observed_tool_dependencies": [],
        "dependency_closure": [], "adapter_requirements": {"capabilities": []},
        "development_provenance": [{"sha256": "a" * 64,
                                     "controller_sha256": "f" * 64}],
        "development_partition": partition, "source_skill_id": "skill:v001"}
    bundle["bundle_sha256"] = hashlib.sha256(json.dumps(bundle, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps(bundle))
    (tmp_path / "result.json").write_text(json.dumps({
        "skill_id": "skill:v001",
        "sealed_evaluation_cases": rows, "episodes": 3,
        "evaluator_successes": 2, "success_rate": 2 / 3,
        "controller_sha256": "f" * 64,
        "evaluation_passed": False,
        "evaluation_bundle_sha256": bundle["bundle_sha256"],
        "evaluation_policies": [{"name": name, "passed": name != "sealed_evaluation"}
            for name in ("frozen_controller", "provenance", "anti_cheating",
                          "generalization", "sealed_evaluation")]}))
    result = _sealed_status(tmp_path, skill_id="skill:v001", states=[4, 5, 6])
    assert result["episodes"] == 3 and result["evaluator_successes"] == 2
    assert result["success_rate"] == 2 / 3 and result["cases"] == rows
    assert result["controller_sha256"] == "f" * 64


def test_skill_dependency_classifies_adapter_native_without_shared_library(tmp_path):
    controller = tmp_path / "controller.py"; controller.write_text("def run(robot): pass\n")
    controller_sha = sha(controller)
    identity = {"episode_id": "e", "environment_generation": "g"}
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"controller_sha256": controller_sha,
        "environment_identity": identity,
        "verification_receipt": {"verified": True, "controller_sha256": controller_sha,
            "environment_identity": identity, "episode_id": "e", "environment_generation": "g"},
        "execution": {"rpc_events": [{"method": "use",
            "arguments": {"tool_id": "libero.seed:v001"}}]}}))
    Workspace = type("Workspace", (), {"root": tmp_path, "controller": controller})
    class Adapter:
        def native_capability_manifest(self):
            return {"libero.seed:v001": {"capability_id": "libero.seed:v001",
                "version": "v001", "contract_sha256": "d" * 64}}
    class Skills:
        def freeze(self, **values): return values
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=Workspace(),
        adapter=Adapter(), tool_library=None, skill_library=Skills())
    result = manager.register_skill(name="seed", task="x", controller=str(controller),
        tool_ids=["libero.seed:v001"], evidence_paths=[str(evidence)])
    assert result["tool_ids"] == []
    assert result["adapter_requirements"]["capabilities"][0]["capability_id"] == "libero.seed:v001"


def test_frozen_cli_branch_never_constructs_coding_model_or_requires_key(tmp_path, monkeypatch):
    controller = tmp_path / "controller.py"; controller.write_text("def run(robot): return {}\n")
    class Sandbox:
        safe = True
        def require(self): pass
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("APEX_API_KEY", raising=False)
    monkeypatch.setattr(cli, "select_sandbox", lambda value: Sandbox())
    monkeypatch.setattr(cli, "adapter_preflight", lambda value: None)
    monkeypatch.setattr(cli, "_model", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("coding model must not be constructed")))
    monkeypatch.setattr(cli, "_run_frozen_benchmark", lambda *a, **k: {
        "evaluation_passed": True, "finished": True, "evaluation_policies": []})
    args = type("Args", (), {"sandbox": "auto", "profile": "benchmark",
        "adapter": "libero", "task": "0", "states": [1],
        "run_dir": str(tmp_path / "run"), "asset_root": str(tmp_path / "assets"),
        "controller_source": str(controller), "frozen_controller": True})()
    assert cli.run_command(args) == 0


def test_nonresumable_unknown_pending_execution_survives_stale_evidence(tmp_path):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    workspace.write_file("controller.py", "def run(robot): return {}\n")
    snapshot = workspace.snapshot()
    evidence_dir = tmp_path / "evidence"; evidence_dir.mkdir()
    stale_identity = {"adapter": "fake", "episode_id": "old",
                      "environment_generation": "old"}
    evidence = {"controller_sha256": sha(workspace.controller),
        "environment_identity": stale_identity,
        "verification_receipt": {"verified": True,
            "controller_sha256": sha(workspace.controller),
            "environment_identity": stale_identity, "episode_id": "old",
            "environment_generation": "old"}, "execution": {"completed": True}}
    evidence_path = evidence_dir / "old.json"
    evidence_path.write_text(json.dumps(evidence))
    save_checkpoint(tmp_path, {"snapshot_id": snapshot.snapshot_id,
        "latest_evidence": {"artifact_uri": "run://evidence/old.json",
            "artifact_sha256": sha(evidence_path)},
        "state": {"pending_execution": {"execution_key": "unknown",
            "call_id": "physical-call"}}, "active_tool_groups": [],
        "active_shared_tools": [], "session": {"index": 1}, "cumulative": {}})
    class NonResumable(FakeAdapter):
        def resume_protocol(self):
            return {"supports_resume": False,
                    "environment_generation": self.generation,
                    "replay_allowed": False, "actions_idempotent": False}
    adapter = NonResumable("x", tmp_path / "adapter-root")
    manager = CapabilityManager(asset_root=tmp_path / "assets",
        workspace=workspace, adapter=adapter)
    loop = AgentLoop(model=object(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=None, workspace=workspace), capability_manager=manager,
        runtime=object(), event_store=EventStore(tmp_path / "events"), root=tmp_path)
    assert loop.state["pending_execution"]["execution_key"] == "unknown"
    with pytest.raises(ProtocolError, match="outcome is unknown"):
        loop._run_controller()
    assert adapter.actions == []


def test_libero_reset_uses_same_warmup_and_changes_generation(tmp_path):
    import numpy as np
    from embodied_codex.deployments.libero import LiberoDeployment, LiberoEpisode
    class Env:
        def __init__(self): self.resets = 0; self.actions = []
        def reset(self): self.resets += 1; return {"robot0_eef_pos": np.zeros(3),
                                                   "agentview_image": np.zeros((2, 2, 3))}
        def set_init_state(self, state):
            return {"robot0_eef_pos": np.zeros(3), "state": state,
                    "agentview_image": np.zeros((2, 2, 3))}
        def step(self, action):
            self.actions.append(action)
            return ({"robot0_eef_pos": np.zeros(3), "state": "same",
                     "agentview_image": np.zeros((2, 2, 3))}, 0, False, {})
    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False; deployment.env = Env()
    deployment.episode = LiberoEpisode("suite", 0, 0, warmup_steps=3)
    deployment._init_states = ["initial"]; deployment._warmup_steps = 3
    deployment.artifact_dir = tmp_path; deployment._execution_index = 0
    deployment._capture_outcome_rgb = lambda name: {"name": name}
    deployment.verified_attachments = set(); deployment.environment_generation = ""
    deployment._reset_to_initial_condition()
    first_generation = deployment.environment_generation
    assert deployment.step == 3 and len(deployment.env.actions) == 3
    deployment._reset_to_initial_condition()
    assert deployment.step == 3 and len(deployment.env.actions) == 6
    assert deployment.environment_generation != first_generation


def test_campaign_preserves_episodic_trial_and_control_step_contract():
    from embodied_codex.legacy.campaign import CampaignAdapter

    class Episodic:
        episodic_trials = True
        trial_horizon_exhausted = False
        step = 17
        instruction = "task"
        sdk_index = {}

        def inspect_native_capability(self, capability_id):
            return {"asset_id": capability_id, "manual": "complete"}

        def reset_case(self):
            self.step = 3

    case = Episodic()
    campaign = CampaignAdapter([("opaque", case)])
    assert campaign.episodic_trials is True
    assert campaign.step == 17
    assert campaign.trial_horizon_exhausted is False
    campaign.reset_case()
    assert campaign.step == 3
    assert campaign.inspect_native_capability("native:v1") == {
        "asset_id": "native:v1", "manual": "complete"}


def test_explicit_trial_budget_is_independent_of_legacy_execution_limit():
    from embodied_codex.legacy.agent_loop import LoopBudget

    budget = LoopBudget(max_steps=60, max_executions=2, max_trials=12)
    budget.executions = 2
    assert budget.exhausted() is False
    budget.executions = 12
    assert budget.exhausted() is True


def test_max_trials_means_cumulative_physical_trials_across_resume():
    from embodied_codex.legacy.agent_loop import LoopBudget

    budget = LoopBudget(max_steps=60, max_executions=20, max_trials=12,
                        trials_before=8)
    budget.executions = 3
    assert budget.exhausted() is False
    budget.executions = 4
    assert budget.exhausted() is True


def test_libero_horizon_exhaustion_is_trial_local():
    import numpy as np
    from embodied_codex.deployments.libero import LiberoDeployment, LiberoDeploymentError

    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.episode = type("Episode", (), {"horizon": 1})()
    deployment.step = 1
    deployment.trial_horizon_exhausted = False
    with pytest.raises(LiberoDeploymentError, match="horizon exhausted"):
        deployment._sim_step(np.zeros(7))
    assert deployment.trial_horizon_exhausted is True


def test_native_tool_inspection_includes_complete_manual_and_nested_schema():
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment = LiberoDeployment.__new__(LiberoDeployment)
    tool_id = "libero.rgbd_perception:v001"
    deployment._native_capability_ids = frozenset({tool_id})
    deployment.capability_contracts = {tool_id: {
        "input_schema": {"type": "object", "properties": {
            "frame": {"type": "object", "properties": {
                "cameras": {"type": "object"}}}}},
        "output_schema": {"type": "object", "properties": {
            "detections": {"type": "object"}}},
    }}
    detail = deployment.inspect_native_capability(tool_id)
    manifest = detail["manifest"]
    assert manifest["input_schema"]["properties"]["frame"]["properties"]["cameras"]
    assert manifest["output_schema"]["properties"]["detections"]
    assert {"purpose", "examples", "failure_modes", "limitations", "provenance"} <= set(
        manifest["manual"])
    assert detail["source"] is None


def test_libero_hidden_evaluator_reads_env_success_only_after_seal():
    from embodied_codex.deployments.libero import LiberoDeployment, LiberoDeploymentError
    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False; deployment._controller_execution_sealed = False
    deployment._evaluator_calls = 0
    deployment.env = type("Env", (), {"check_success": lambda self: True})()
    with pytest.raises(LiberoDeploymentError, match="not sealed"):
        deployment.hidden_evaluator()
    deployment._controller_execution_sealed = True
    assert deployment.hidden_evaluator() is True


def test_libero_verify_exception_diagnostic_is_projectable_and_blind(tmp_path):
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.references = {"artifact://source": {"world_xyz": [0.0, 0.0, 0.0]}}
    deployment.verifiers = {
        "visual_attachment": lambda payload: (_ for _ in ()).throw(
            RuntimeError("synthetic verifier failure"))
    }
    deployment.verified_attachments = set()
    deployment.trace = []
    deployment.step = 3
    deployment.last_verify = False

    raw = deployment.dispatch("verify", {
        "verifier": "visual_attachment",
        "payload": {
            "frame": {"frame_id": "frame-1", "cameras": {}},
            "object_query": "object",
            "source_ref": "artifact://source",
        },
    })
    projected = deployment.project_rpc_output("verify", {}, raw)

    assert projected["verified"] is False
    assert projected["sensor_only"] is True
    assert projected["verifier_error"] == {
        "type": "RuntimeError", "message": "synthetic verifier failure"}
    assert not {"reward", "done", "hidden_evaluator", "check_success"} & set(projected)


def test_libero_verify_projection_rejects_undeclared_fields_and_never_evaluates_hidden_state():
    from embodied_codex.deployments.libero import LiberoDeployment, LiberoDeploymentError

    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.references = {"artifact://source": {"world_xyz": [0.0, 0.0, 0.0]}}
    deployment.verifiers = {"visual_attachment": lambda payload: {"verified": True}}
    deployment.verified_attachments = set()
    deployment.trace = []
    deployment.step = 3
    deployment.last_verify = False
    class Env:
        def check_success(self):
            raise AssertionError("hidden evaluator must not run during verify")
    deployment.env = Env()

    raw = deployment.dispatch("verify", {
        "verifier": "visual_attachment",
        "payload": {
            "frame": {"frame_id": "frame-1", "cameras": {}},
            "object_query": "object",
            "source_ref": "artifact://source",
        },
    })
    projected = deployment.project_rpc_output("verify", {}, raw)
    assert projected == {"verified": True}
    with pytest.raises(LiberoDeploymentError, match="undeclared verify output fields"):
        deployment.project_rpc_output("verify", {}, {"verified": False, "unexpected": True})


def test_reset_observation_is_registered_as_opaque_agent_artifact(tmp_path):
    image = tmp_path / "adapter" / "fresh.png"; image.parent.mkdir(); image.write_bytes(b"png")
    class Adapter:
        artifact_dir = image.parent
        generation = "before"
        def execution_identity(self):
            return {"episode_id": "opaque", "environment_generation": self.generation}
        def canonical_observation(self, observation):
            return dict(observation)
        def reset_case(self):
            self.generation = "after"
            return {"rgb_path": str(image)}
    loop = AgentLoop.__new__(AgentLoop)
    loop.adapter = Adapter(); loop.workspace = type("W", (), {"root": tmp_path / "workspace"})()
    loop.root = tmp_path; loop._artifact_handles = {}; loop.latest_evidence = {"old": True}
    loop._agent_latest_evidence = {"old": True}; loop.state = {}
    loop.context_builder = ContextBuilder(adapter_index={}, asset_registry=None,
        workspace=None)
    result = loop._reset_case()
    handle = loop.context_builder.initial_observation["rgb_path"]
    assert handle.startswith("artifact://agent/") and str(tmp_path) not in json.dumps(result)


def test_libero_verifier_configuration_reaches_vlm_constructor(tmp_path, monkeypatch):
    import embodied_codex.adapters.libero as plugin
    captured = {}
    class Perception:
        def __init__(self, **kwargs): pass
        def detect(self, payload): return {}
        def verify_attachment(self, payload): return {"verified": False}
        def verify_support_relation(self, payload): return {"verified": False}
    class Grasp:
        def __init__(self, **kwargs): pass
        def infer(self, payload): return {}
    class Verifier:
        def __init__(self, **kwargs): captured.update(kwargs)
        def verify(self, payload): return {"verified": False}
    class Deployment:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr(plugin, "OpenVocabularyRGBD", Perception)
    monkeypatch.setattr(plugin, "GraspNetRGBD", Grasp)
    monkeypatch.setattr(plugin, "VLMVisualTaskOutcomeVerifier", Verifier)
    monkeypatch.setattr(plugin, "LiberoDeployment", Deployment)
    monkeypatch.setattr(plugin, "_paths", lambda: {
        "package_root": tmp_path, "groundingdino_root": tmp_path,
        "groundingdino_config": tmp_path / "config", "groundingdino_checkpoint": tmp_path / "dino",
        "groundingdino_text_encoder": tmp_path, "sam_root": tmp_path,
        "sam_checkpoint": tmp_path / "sam", "graspnet_checkpoint": tmp_path / "grasp",
        "graspnet_root": tmp_path})
    provider = type("Provider", (), {"api_key": "secret", "endpoint": "https://official.example/v1"})()
    monkeypatch.setattr(plugin, "resolve_provider", lambda **kwargs: provider)
    monkeypatch.setenv("OPENAI_API_KEY", "not-logged")
    plugin.create(task="0", state=1, root=tmp_path / "run", configuration={
        "verifier_provider": "openai", "verifier_base_url": "https://official.example/v1",
        "verifier_model": "verifier-model", "verifier_reasoning_effort": "medium"})
    assert captured == {"api_key": "secret", "base_url": "https://official.example/v1",
                        "model": "verifier-model", "reasoning_effort": "medium"}
