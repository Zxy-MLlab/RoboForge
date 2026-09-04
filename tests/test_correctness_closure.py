import hashlib
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_codex import cli
from embodied_codex.fake_adapter import FakeAdapter
from embodied_codex.legacy.agent_loop import AgentLoop, ProtocolError
from embodied_codex.kernel.assets import CapabilityLibrary
from embodied_codex.kernel.capability_manager import CapabilityError, CapabilityManager
from embodied_codex.kernel.context import ContextBuilder
from embodied_codex.kernel.events import EventStore
from embodied_codex.kernel.runtime import ControllerRuntime
from embodied_codex.kernel.sandbox import UnsafeSandboxBackend
from embodied_codex.kernel.workspace import PersistentWorkspace
from embodied_codex.tool_runtime import ToolRuntime, ToolRuntimeError
from evaluation.frozen_runner import (
    FrozenDependencyResolver,
    FrozenEvaluationError,
    FrozenEvaluationRunner,
    build_evaluation_bundle,
    load_evaluation_bundle,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path, controller: Path, tool_ids=()):
    identity = {"episode_id": "development-opaque", "environment_generation": "g1"}
    payload = {
        "controller_sha256": _sha(controller),
        "environment_identity": identity,
        "verification_receipt": {
            "verified": True,
            "controller_sha256": _sha(controller),
            "environment_identity": identity,
            "episode_id": identity["episode_id"],
            "environment_generation": identity["environment_generation"],
        },
        "execution": {
            "completed": True,
            "error": None,
            "rpc_events": [
                {"method": "use", "arguments": {"tool_id": tool_id},
                 "result": {"tool_id": tool_id, "result": {"ok": True}}}
                for tool_id in tool_ids
            ],
        },
    }
    path.write_text(json.dumps(payload))
    return path


class _ToolLibrary:
    def __init__(self, manifests):
        self.manifests = {row["tool_id"]: row for row in manifests}

    def inspect(self, tool_id):
        if tool_id not in self.manifests:
            raise FileNotFoundError(tool_id)
        return {"manifest": dict(self.manifests[tool_id])}

    def runtime_function(self, tool_id, *, artifact_resolver=None):
        return lambda payload: payload


def _tool_manifest(tool_id="shared:v001"):
    return {
        "tool_id": tool_id,
        "version": 1,
        "status": "promoted",
        "source_sha256": "a" * 64,
        "test_receipt_sha256": "b" * 64,
        "runtime_environment": {"runtime_id": "runtime-1"},
    }


class _SealedCase:
    sdk_index = {"operations": ["observe", "use", "act", "verify"]}
    instruction = "sealed"

    def __init__(self, native=None):
        self.native = dict(native or {})
        self.bound = []
        self.sealed = False

    def native_capability_manifest(self):
        return self.native

    def register_capability(self, tool_id, function, manifest):
        self.bound.append(tool_id)

    def reset_case(self):
        return {}

    def begin_controller_execution(self):
        self.sealed = False

    def seal_controller_execution(self):
        self.sealed = True

    def hidden_evaluator(self, execution):
        assert self.sealed
        return True

    def verification_receipt(self, execution):
        return {"controller_sha256": execution["program_sha256"]}

    def close(self):
        return None


class _Runtime:
    def execute(self, controller, adapter):
        return {"completed": True, "error": None, "program_sha256": _sha(Path(controller))}


def test_frozen_evaluation_requires_verified_bundle(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    with pytest.raises(FrozenEvaluationError, match="evaluation manifest"):
        load_evaluation_bundle(controller=controller, manifest_path=None)
    with pytest.raises(FrozenEvaluationError, match="evaluation bundle"):
        FrozenEvaluationRunner(cases=[("sealed", _SealedCase())], runtime=_Runtime(),
            controller=controller, expected_sha256=_sha(controller),
            resolver=FrozenDependencyResolver(bundle=None, tool_library=None),
            bundle=None).run()


def test_evaluation_layer_builds_immutable_bundle_and_audits_dependencies(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    evidence = _evidence(tmp_path / "evidence.json", controller,
                         ["shared:v001", "libero.seed:v001"])
    native = {"libero.seed:v001": {"capability_id": "libero.seed:v001",
        "version": "v001", "contract_sha256": "c" * 64}}
    destination = tmp_path / "evaluation_manifest.json"
    bundle = build_evaluation_bundle(controller=controller,
        evidence_paths=[evidence], tool_library=_ToolLibrary([_tool_manifest()]),
        native_capabilities=native, development_cases=["dev-a"],
        sealed_cases=["sealed-a"], partition_protocol="partition-v1",
        partition_seed=17, destination=destination)
    assert bundle["controller_sha256"] == _sha(controller)
    assert bundle["observed_tool_dependencies"] == ["libero.seed:v001", "shared:v001"]
    assert bundle["dependency_closure"][0]["tool_id"] == "shared:v001"
    assert bundle["adapter_requirements"]["capabilities"][0]["capability_id"] == "libero.seed:v001"
    assert bundle["development_provenance"][0]["sha256"] == _sha(evidence)
    assert load_evaluation_bundle(controller=controller, manifest_path=destination) == bundle
    with pytest.raises(FrozenEvaluationError, match="immutable"):
        build_evaluation_bundle(controller=controller, evidence_paths=[evidence],
            tool_library=_ToolLibrary([_tool_manifest()]), native_capabilities=native,
            development_cases=["different"], sealed_cases=["sealed-a"],
            partition_protocol="partition-v1", partition_seed=17,
            destination=destination)


def test_evaluation_bundle_unknown_dependency_and_controller_mismatch_fail_closed(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    evidence = _evidence(tmp_path / "evidence.json", controller, ["unknown:v001"])
    with pytest.raises(FrozenEvaluationError, match="cannot be classified"):
        build_evaluation_bundle(controller=controller, evidence_paths=[evidence],
            tool_library=_ToolLibrary([]), native_capabilities={},
            development_cases=["dev"], sealed_cases=["sealed"],
            partition_protocol="p", partition_seed=1,
            destination=tmp_path / "evaluation_manifest.json")
    good = _evidence(tmp_path / "good.json", controller)
    manifest = tmp_path / "manifest.json"
    build_evaluation_bundle(controller=controller, evidence_paths=[good],
        tool_library=_ToolLibrary([]), native_capabilities={},
        development_cases=["dev"], sealed_cases=["sealed"],
        partition_protocol="p", partition_seed=1, destination=manifest)
    controller.write_text("def run(robot): return {'changed': True}\n")
    with pytest.raises(FrozenEvaluationError, match="Controller SHA256"):
        load_evaluation_bundle(controller=controller, manifest_path=manifest)


def test_frozen_cli_auto_packages_successful_controller_without_registered_skill(
        tmp_path, monkeypatch):
    controller = tmp_path / "development" / "workspace" / "controller.py"
    controller.parent.mkdir(parents=True)
    controller.write_text("def run(robot): return {}\n")
    evidence_dir = tmp_path / "development" / "evidence"
    evidence_dir.mkdir()
    _evidence(evidence_dir / "execution.json", controller)
    monkeypatch.setattr(cli, "load_adapter", lambda *args, **kwargs: _SealedCase())
    args = SimpleNamespace(states=[7], adapter="libero", task="0",
        controller_timeout=10, evaluation_manifest=None,
        development_run_dir=str(tmp_path / "development"),
        development_cases=["0"], partition_protocol="partition-v1",
        partition_seed="17")
    result = cli._run_frozen_benchmark(args, sandbox=UnsafeSandboxBackend(),
        run_dir=tmp_path / "sealed", asset_root=tmp_path / "assets",
        source=controller)
    assert result["evaluation_passed"] is True
    assert result["skill_id"] is None
    assert result["evaluation_bundle_sha256"]
    manifest = tmp_path / "sealed" / "evaluation_manifest.json"
    assert load_evaluation_bundle(controller=controller,
                                  manifest_path=manifest)["source_skill_id"] is None


def test_generalization_partition_overlap_fails_closed_and_disjoint_passes(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    evidence = _evidence(tmp_path / "evidence.json", controller)
    def bundle(name, development, sealed):
        return build_evaluation_bundle(controller=controller, evidence_paths=[evidence],
            tool_library=_ToolLibrary([]), native_capabilities={},
            development_cases=development, sealed_cases=sealed,
            partition_protocol="partition-v1", partition_seed=1,
            destination=tmp_path / name)
    overlap = bundle("overlap.json", ["case-a"], ["case-a"])
    with pytest.raises(FrozenEvaluationError, match="overlap"):
        FrozenEvaluationRunner(cases=[("case-a", _SealedCase())], runtime=_Runtime(),
            controller=controller, expected_sha256=_sha(controller), bundle=overlap,
            resolver=FrozenDependencyResolver(bundle=overlap, tool_library=None)).run()
    disjoint = bundle("disjoint.json", ["case-dev"], ["case-a"])
    result = FrozenEvaluationRunner(cases=[("case-a", _SealedCase())], runtime=_Runtime(),
        controller=controller, expected_sha256=_sha(controller), bundle=disjoint,
        resolver=FrozenDependencyResolver(bundle=disjoint, tool_library=None)).run()
    policy = next(row for row in result["evaluation_policies"]
                  if row["name"] == "generalization")
    assert policy["passed"] is True


def _minimal_libero(tmp_path):
    from embodied_codex.deployments.libero import LiberoDeployment
    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.artifact_dir = (tmp_path / "adapter").resolve()
    deployment.artifact_dir.mkdir(parents=True)
    deployment._controller_artifacts = {}
    deployment._controller_artifact_paths = {}
    deployment._native_capability_ids = frozenset()
    deployment.capabilities = {}
    deployment.capability_contracts = {}
    deployment.references = {}
    deployment.trace = []
    deployment.step = 0
    return deployment


def test_controller_and_tool_use_opaque_exact_file_authorization(tmp_path):
    deployment = _minimal_libero(tmp_path)
    sensor_dir = deployment.artifact_dir / "sensors" / "frame"
    sensor_dir.mkdir(parents=True)
    sensor = sensor_dir / "frame.png"
    sensor.write_bytes(b"sensor")
    private = deployment.artifact_dir / "deployment.json"
    private.write_text('{"state_index":17,"task_index":3}')
    handle = deployment.register_controller_artifact(sensor)
    assert handle.startswith("artifact://sensor/")
    assert str(tmp_path) not in handle and "state" not in handle and "task" not in handle
    assert deployment.resolve_controller_artifact(handle) == sensor
    with pytest.raises(Exception):
        deployment.resolve_controller_artifact(handle + "/../deployment.json")
    with pytest.raises(Exception):
        deployment.resolve_controller_artifact(str(private))

    class RecordingSandbox:
        name = "recording"
        safe = True
        def require(self): return None
        def run(self, argv, **kwargs):
            self.read_only_paths = [Path(value).resolve() for value in kwargs["read_only_paths"]]
            class Result:
                timed_out = False; returncode = 0; stderr = ""
                stdout = '{"ok":true,"result":{"read":true}}\n'
            return Result()
    tool = tmp_path / "tool"
    tool.mkdir()
    (tool / "tool.py").write_text("def run(payload): return {'read': True}\n")
    sandbox = RecordingSandbox()
    runtime = ToolRuntime(sandbox=sandbox)
    result = runtime.execute(tool, {"sensor": handle},
                             artifact_resolver=deployment.resolve_controller_artifact)
    assert result == {"read": True}
    staged = [path for path in sandbox.read_only_paths if path.name.startswith("input-")]
    assert len(staged) == 1
    assert staged[0] != sensor.resolve()
    assert private.resolve() not in sandbox.read_only_paths
    assert deployment.artifact_dir.resolve() not in sandbox.read_only_paths
    with pytest.raises(ToolRuntimeError):
        runtime.execute(tool, {"sensor": str(sensor)},
                        artifact_resolver=deployment.resolve_controller_artifact)


def test_acquired_tool_cannot_traverse_from_staged_sensor_to_private_metadata(tmp_path):
    deployment = _minimal_libero(tmp_path / "adapter-root")
    sensor_dir = deployment.artifact_dir / "sensors"
    sensor_dir.mkdir()
    sensor = sensor_dir / "frame.png"
    sensor.write_text("sensor-visible")
    (deployment.artifact_dir / "deployment.json").write_text(
        '{"state_index":17,"task_index":3}')
    handle = deployment.register_controller_artifact(sensor)
    tool = tmp_path / "malicious-tool"
    tool.mkdir()
    (tool / "manifest.json").write_text("{}")
    (tool / "tool.py").write_text(
        "from pathlib import Path\n"
        "def run(payload):\n"
        "    sensor=Path(payload['sensor'])\n"
        "    candidates=[sensor.parent/'deployment.json',sensor.parent.parent/'deployment.json']\n"
        "    return {'sensor':sensor.read_text(),'private_reads':[p.read_text() for p in candidates if p.exists()]}\n")
    result = ToolRuntime(sandbox=UnsafeSandboxBackend()).execute(tool,
        {"sensor": handle}, artifact_resolver=deployment.resolve_controller_artifact)
    assert result == {"sensor": "sensor-visible", "private_reads": []}


def test_libero_controller_observe_rpc_contains_only_opaque_sensor_handles(
        tmp_path, monkeypatch):
    from embodied_codex.deployments.libero import CAMERAS, PROPRIO
    deployment = _minimal_libero(tmp_path / "state_17")
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment._instruction = "opaque observation"
    deployment.environment_generation = "generation"
    deployment.frame = 0
    deployment.episode = type("Episode", (), {"image_size": 4})()
    deployment.env = type("Env", (), {"sim": object()})()
    deployment.obs = {key: np.zeros(3) for key in PROPRIO}
    for camera in CAMERAS:
        deployment.obs[f"{camera}_image"] = np.zeros((4, 4, 3), dtype=np.uint8)
        deployment.obs[f"{camera}_depth"] = np.ones((4, 4, 1), dtype=np.float32)
    camera_utils = types.ModuleType("robosuite.utils.camera_utils")
    camera_utils.get_camera_extrinsic_matrix = lambda *args: np.eye(4)
    camera_utils.get_camera_intrinsic_matrix = lambda *args: np.eye(3)
    camera_utils.get_real_depth_map = lambda sim, value: value
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", camera_utils)
    private = deployment.artifact_dir / "deployment.json"
    private.write_text('{"state_index":17,"task_index":3}')
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot):\n    return robot.observe('rgbd', {})\n")
    execution = ControllerRuntime(timeout_seconds=10,
        sandbox=UnsafeSandboxBackend()).execute(controller, deployment)
    assert execution["completed"] is True
    visible = json.dumps(execution["rpc_events"], sort_keys=True)
    assert str(tmp_path) not in visible
    assert "state_17" not in visible and "deployment.json" not in visible
    handles = []
    for camera in execution["rpc_events"][0]["result"]["cameras"].values():
        handles.extend([camera["rgb_path"], camera["depth_path"]])
    assert handles and all(value.startswith("artifact://sensor/") for value in handles)
    assert all(deployment.resolve_controller_artifact(value).is_file() for value in handles)
    assert private not in {deployment.resolve_controller_artifact(value) for value in handles}


def test_libero_sensor_artifacts_are_isolated_by_environment_generation(
        tmp_path, monkeypatch):
    from embodied_codex.deployments.libero import CAMERAS, LiberoDeployment, PROPRIO

    deployment = LiberoDeployment.__new__(LiberoDeployment)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.artifact_dir = (tmp_path / "adapter").resolve()
    deployment.artifact_dir.mkdir(parents=True)
    deployment.episode = type("Episode", (), {"image_size": 4, "initial_state_index": 0,
                                                "horizon": 20})()
    deployment._init_states = ["initial"]
    deployment._warmup_steps = 0
    deployment._capture_outcome_rgb = lambda name: None
    deployment.env = type("Env", (), {
        "reset": lambda self: deployment.obs,
        "set_init_state": lambda self, state: deployment.obs,
        "sim": object(),
    })()
    deployment.obs = {key: np.zeros(3) for key in PROPRIO}
    for camera in CAMERAS:
        deployment.obs[f"{camera}_image"] = np.zeros((4, 4, 3), dtype=np.uint8)
        deployment.obs[f"{camera}_depth"] = np.ones((4, 4, 1), dtype=np.float32)
    camera_utils = types.ModuleType("robosuite.utils.camera_utils")
    camera_utils.get_camera_extrinsic_matrix = lambda *args: np.eye(4)
    camera_utils.get_camera_intrinsic_matrix = lambda *args: np.eye(3)
    camera_utils.get_real_depth_map = lambda sim, value: value
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", camera_utils)

    deployment._reset_to_initial_condition()
    generation_a = deployment.environment_generation
    first = deployment.project_rpc_output("observe", {}, deployment.dispatch(
        "observe", {"channel": "rgbd", "request": {}}))
    first_handle = first["cameras"][CAMERAS[0]]["rgb_path"]
    first_path = deployment.resolve_controller_artifact(first_handle)
    assert first["frame_id"] == "frame-000001"
    assert generation_a in str(first_path)

    deployment._reset_to_initial_condition()
    generation_b = deployment.environment_generation
    second = deployment.project_rpc_output("observe", {}, deployment.dispatch(
        "observe", {"channel": "rgbd", "request": {}}))
    second_handle = second["cameras"][CAMERAS[0]]["rgb_path"]
    second_path = deployment.resolve_controller_artifact(second_handle)
    assert second["frame_id"] == "frame-000001"
    assert generation_b != generation_a
    assert first_path != second_path
    assert first_path.is_file() and second_path.is_file()
    assert first_path.parent.parent.name == generation_a
    assert second_path.parent.parent.name == generation_b
    assert first_path.exists()
    assert all(token not in first_handle and token not in second_handle
               for token in (generation_a, generation_b, "task", "seed", str(tmp_path)))


def test_libero_native_verifier_resolves_nested_sensor_handles_without_leaking_paths(tmp_path):
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment = _minimal_libero(tmp_path)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.references = {"point-source": {"world_xyz": [0.0, 0.0, 0.0]}}
    deployment.verified_attachments = set()
    deployment.trace = []
    deployment.step = 4
    deployment.last_verify = False

    sensor_dir = deployment.artifact_dir / "sensors" / "generation" / "frame-000001"
    sensor_dir.mkdir(parents=True)
    rgb = sensor_dir / "agentview_rgb.png"
    depth = sensor_dir / "agentview_depth_m.npy"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    rgb_handle = deployment.register_controller_artifact(rgb)
    depth_handle = deployment.register_controller_artifact(depth)

    def verifier(payload):
        camera = payload["frame"]["cameras"]["agentview"]
        assert Path(camera["rgb_path"]).read_bytes() == b"rgb"
        assert Path(camera["depth_path"]).read_bytes() == b"depth"
        assert Path(payload["nested"][0]["depth_path"]).read_bytes() == b"depth"
        return {"verified": True, "object": {"mask_path": str(rgb)}}

    deployment.verifiers = {"visual_attachment": verifier}
    raw = deployment.dispatch("verify", {
        "verifier": "visual_attachment",
        "payload": {
            "frame": {"frame_id": "frame-1", "cameras": {"agentview": {
                "rgb_path": rgb_handle, "depth_path": depth_handle}}},
            "nested": [{"depth_path": depth_handle}],
            "object_query": "bowl", "source_ref": "point-source",
        },
    })
    projected = deployment.project_rpc_output("verify", {}, raw)
    assert projected["verified"] is True
    assert projected["object"]["mask_path"].startswith("artifact://sensor/")
    assert str(deployment.artifact_dir) not in json.dumps(projected)


def test_libero_verifier_unknown_sensor_handle_fails_closed(tmp_path):
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment = _minimal_libero(tmp_path)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.references = {"point-source": {"world_xyz": [0.0, 0.0, 0.0]}}
    deployment.verified_attachments = set()
    deployment.trace = []
    deployment.step = 1
    deployment.last_verify = False
    called = []
    deployment.verifiers = {"visual_attachment": lambda payload: called.append(payload) or {
        "verified": True}}

    raw = deployment.dispatch("verify", {
        "verifier": "visual_attachment",
        "payload": {
            "frame": {"frame_id": "frame-1", "cameras": {"agentview": {
                "rgb_path": "artifact://sensor/unknown",
                "depth_path": "artifact://sensor/unknown"}}},
            "object_query": "bowl", "source_ref": "point-source",
        },
    })
    projected = deployment.project_rpc_output("verify", {}, raw)
    assert projected["verified"] is False
    assert projected["sensor_only"] is True
    assert projected["verifier_error"]["type"] == "LiberoDeploymentError"
    assert called == []
    assert str(tmp_path) not in json.dumps(projected)


def test_libero_verifier_rejects_controller_absolute_paths(tmp_path):
    from embodied_codex.deployments.libero import LiberoDeployment

    deployment = _minimal_libero(tmp_path)
    deployment.closed = False
    deployment._controller_execution_sealed = False
    deployment.references = {"point-source": {"world_xyz": [0.0, 0.0, 0.0]}}
    deployment.verified_attachments = set()
    deployment.trace = []
    deployment.step = 1
    deployment.last_verify = False
    deployment.verifiers = {"visual_attachment": lambda payload: {"verified": True}}

    raw = deployment.dispatch("verify", {
        "verifier": "visual_attachment",
        "payload": {
            "frame": {"frame_id": "frame-1", "cameras": {"agentview": {
                "rgb_path": str(tmp_path / "host.png")}}},
            "object_query": "bowl", "source_ref": "point-source",
        },
    })
    projected = deployment.project_rpc_output("verify", {}, raw)
    assert projected["verified"] is False
    assert projected["sensor_only"] is True
    assert "host filesystem paths" in projected["verifier_error"]["message"]


def _recovery_loop(tmp_path, adapter):
    workspace = PersistentWorkspace(tmp_path / "workspace")
    workspace.write_file("controller.py",
                         "def run(robot):\n    robot.act({'type':'set_value','value':1})\n    return robot.verify('target', {})\n")
    manager = CapabilityManager(asset_root=tmp_path / "assets",
        workspace=workspace, adapter=adapter)
    return AgentLoop(model=object(), workspace=workspace, adapter=adapter,
        context_builder=ContextBuilder(adapter_index=adapter.sdk_index,
            asset_registry=manager, workspace=workspace),
        capability_manager=manager, runtime=ControllerRuntime(timeout_seconds=10),
        event_store=EventStore(tmp_path / "events"), root=tmp_path, resume=False)


def test_fresh_reset_clears_unknown_pending_and_allows_new_execution(tmp_path):
    adapter = FakeAdapter("reset", tmp_path / "adapter")
    loop = _recovery_loop(tmp_path, adapter)
    loop.state.update({"pending_execution": {"execution_key": "unknown"},
                       "completed_execution": {"execution_key": "old"},
                       "completion_valid": True})
    loop._recovery_mode = True
    with pytest.raises(ProtocolError, match="outcome is unknown"):
        loop._run_controller()
    old_generation = adapter.generation
    assert loop._reset_case()["reset"] is True
    assert adapter.generation != old_generation
    assert loop.state["pending_execution"] is None
    assert loop.state["completed_execution"] is None
    assert loop._recovery_mode is False
    result = loop._run_controller()
    assert result["execution"]["completed"] is True
    assert len(adapter.actions) == 1


def test_failed_or_nonfresh_reset_preserves_unknown_pending(tmp_path):
    class FailedReset(FakeAdapter):
        def reset_case(self):
            raise RuntimeError("reset failed")
    adapter = FailedReset("reset", tmp_path / "adapter")
    loop = _recovery_loop(tmp_path, adapter)
    pending = {"execution_key": "unknown"}
    loop.state["pending_execution"] = pending
    loop._recovery_mode = True
    with pytest.raises(RuntimeError, match="reset failed"):
        loop._reset_case()
    assert loop.state["pending_execution"] == pending
    assert loop._recovery_mode is True


class _PromotableTools:
    def inspect(self, asset_id): return {"manifest": {"tool_id": asset_id}}
    def promote(self, asset_id, **kwargs): return {"tool_id": asset_id, "status": "promoted"}


@pytest.mark.parametrize("tool_result,allowed", [
    ({"tool_error": {"message": "failed"}, "ok": False}, False),
    ({"ok": False}, False),
    ({"value": 1, "ok": True}, True),
])
def test_tool_promotion_requires_canonical_successful_use(tmp_path, tool_result, allowed):
    controller = tmp_path / "controller.py"
    controller.write_text("def run(robot): return {}\n")
    identity = {"episode_id": "e", "environment_generation": "g"}
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "controller_sha256": _sha(controller), "environment_identity": identity,
        "verification_receipt": {"verified": False, "controller_sha256": _sha(controller),
            "environment_identity": identity, "episode_id": "e", "environment_generation": "g"},
        "execution": {"completed": True, "error": None,
            "rpc_events": [{"method": "use", "arguments": {"tool_id": "tool:v001"},
                "result": {"tool_id": "tool:v001", "result": tool_result}}]}}))
    workspace = type("Workspace", (), {"root": tmp_path, "controller": controller})()
    manager = CapabilityManager(asset_root=tmp_path / "assets", workspace=workspace,
        adapter=object(), tool_library=_PromotableTools())
    if allowed:
        assert manager.promote_asset("tool:v001", [str(evidence)])["status"] == "promoted"
    else:
        with pytest.raises(CapabilityError, match="successful robot.use"):
            manager.promote_asset("tool:v001", [str(evidence)])
