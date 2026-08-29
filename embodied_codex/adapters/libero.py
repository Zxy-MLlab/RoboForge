"""LIBERO Adapter plugin. All task execution still runs through kernel.AgentLoop."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

from ..capabilities import (GraspNetRGBD, OpenVocabularyRGBD,
                            VLMVisualRelationGrounder, VLMVisualTaskOutcomeVerifier)
from ..deployments.libero import LiberoDeployment, LiberoEpisode
from ..providers import resolve_provider
from .libero_sdk import LIBERO_ROBOT_SDK_CONTRACT


DOCTOR_TASK = "0"


_CHECKPOINT_SHA256 = {
    "groundingdino": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    "sam": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    "graspnet": "60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868",
}

_TEXT_ENCODER_SHA256 = {
    "config.json": "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20",
    "pytorch_model.bin": "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a",
    "tokenizer.json": "ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98",
    "tokenizer_config.json": "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


def _sdk_index(capabilities, verifiers, contracts=None):
    """Return the bounded SDK index needed to form valid RPC requests.

    This is contract metadata, not a task policy: it describes the accepted
    request shapes while leaving object selection and motion strategy to the
    controller.
    """
    def compact_schema(schema):
        schema = dict(schema or {})
        properties = dict(schema.get("properties") or {})
        return {"type": schema.get("type", "object"),
                "required": list(schema.get("required") or []),
                "fields": sorted(str(key) for key in properties)}

    def action_contract(name, contract):
        return {key: value for key, value in contract.items()
                if key in {"required", "any_of", "enum", "optional", "field_semantics",
                           "rule", "example", "examples"}}

    purposes = {
        "libero.rgbd_perception:v001":
            "Detect language-named objects from public calibrated RGB-D observations.",
        "libero.grasp_proposals:v001":
            "Generate grasp candidates from a public RGB-D frame and detection.",
    }
    def tool_index(capability_id):
        purpose = purposes.get(str(capability_id), "Adapter-native capability.")
        return {"id": str(capability_id), "description": purpose, "purpose": purpose}

    return {
        "protocol": "embodied-codex-libero-robot-sdk-v1",
        "operations": ["observe", "use", "act", "verify", "record"],
        "methods": LIBERO_ROBOT_SDK_CONTRACT["methods"],
        "action_contracts": {
            name: action_contract(name, contract)
            for name, contract in LIBERO_ROBOT_SDK_CONTRACT["actions"].items()
        },
        "verifier_contracts": {
            name: {key: value for key, value in contract.items()
                   if key in {"required", "optional", "rule", "example"}}
            for name, contract in LIBERO_ROBOT_SDK_CONTRACT["verifiers"].items()
        },
        "seed_tool_contracts": {
            name: {"input": compact_schema(contract.get("input_schema")),
                   "output": compact_schema(contract.get("output_schema"))}
            for name, contract in dict(contracts or {}).items()
        },
        "seed_tools": sorted(capabilities),
        "seed_tool_index": [tool_index(name) for name in sorted(capabilities)],
        "verifiers": sorted(verifiers),
    }


def _path(env_name: str, default: str) -> Path:
    return Path(os.environ.get(env_name, default)).expanduser().resolve()


def _vendor_configuration() -> dict[str, str]:
    configured = os.environ.get("ROBOFORGE_LIBERO_VENDOR_CONFIG")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        config_home = Path(os.environ.get("XDG_CONFIG_HOME",
            str(Path.home() / ".config"))).expanduser().resolve()
        path = config_home / "roboforge" / "libero_vendor.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid LIBERO vendor configuration: {path}") from exc
    if not isinstance(value, dict) or value.get("protocol") != "roboforge-libero-vendor-v1":
        raise RuntimeError(f"unsupported LIBERO vendor configuration: {path}")
    sources = value.get("sources")
    if not isinstance(sources, dict) or not all(isinstance(key, str)
            and isinstance(item, str) for key, item in sources.items()):
        raise RuntimeError(f"invalid LIBERO vendor source map: {path}")
    return dict(sources)


def _configured_path(environment: str, key: str, default: Path,
                     configuration: dict[str, str]) -> Path:
    value = os.environ.get(environment) or configuration.get(key) or str(default)
    return Path(value).expanduser().resolve()


def _array(item, length: int | None = None):
    value = {"type": "array", "items": item}
    if length is not None:
        value.update(minItems=length, maxItems=length)
    return value


def _frame_schema():
    number = {"type": "number"}
    vector3 = _array(number, 3)
    quaternion = _array(number, 4)
    eef_pose = {"type": "object", "properties": {
        "frame": {"type": "string"}, "position_m": vector3,
        "orientation_xyzw": quaternion},
        "required": ["frame", "position_m", "orientation_xyzw"],
        "additionalProperties": False}
    gripper = {"type": "object", "properties": {"width_m": number},
               "required": ["width_m"], "additionalProperties": False}
    joint_state = {"type": "object", "properties": {
        "position": {"type": "array", "items": number},
        "velocity": {"type": "array", "items": number},
        "gripper_velocity": {"type": "array", "items": number}},
        "additionalProperties": False}
    proprioception = {"type": "object", "properties": {
        "eef_pose": eef_pose, "gripper": gripper,
        "joint_state": joint_state,
        "proprioception": {"type": "object", "properties": {
            "joint_position": {"type": "array", "items": number},
            "joint_velocity": {"type": "array", "items": number}},
            "additionalProperties": False}},
        "additionalProperties": False}
    camera = {"type": "object", "properties": {
        "rgb_path": {"type": "string"}, "rgb_sha256": {"type": "string"},
        "depth_path": {"type": "string"}, "depth_sha256": {"type": "string"},
        "shape": _array({"type": "integer"}), "depth_range_m": _array(number, 2),
        "intrinsic": _array(_array(number, 3), 3),
        "camera_to_world": _array(_array(number, 4), 4)},
        "required": ["rgb_path", "depth_path", "intrinsic", "camera_to_world"],
        "additionalProperties": False}
    return {"type": "object", "properties": {
        "frame_id": {"type": "string"}, "step": {"type": "integer"},
        "cameras": {"type": "object", "additionalProperties": camera},
        "proprioception": proprioception,
    }, "required": ["frame_id", "cameras"], "additionalProperties": False}


def _detection_schema():
    number = {"type": "number"}
    return {"type": "object", "properties": {
        "query": {"type": "string"}, "label": {"type": "string"},
        "score": number, "box_xyxy": _array(number, 4), "sam_score": number,
        "box_containment": number, "mask_pixels": {"type": "integer"},
        "mask_path": {"type": "string"}, "world_xyz": _array(number, 3),
        "world_bounds_10_90": _array(_array(number, 3), 2),
        "projection_error": {"type": "string"}, "point_ref": {"type": "string"}},
        "required": ["query", "label", "score", "box_xyxy"],
        "additionalProperties": False}


def _perception_contract():
    number = {"type": "number"}; string = {"type": "string"}
    detection = _detection_schema()
    issue = {"type": "object", "properties": {
        "kind": string, "query": string, "detail": string,
        "candidate_index": {"type": "integer"},
        "candidate_indices": _array({"type": "integer"}),
        "queries": _array(string), "score_margin": number,
        "box_iou": number, "world_distance_m": {"type": ["number", "null"]}},
        "required": ["kind"], "additionalProperties": False}
    reliability = {"type": "object", "properties": {
        "protocol": string, "frame_id": string,
        "status": {"type": "string", "enum": ["supported", "uncertain", "unusable"]},
        "requires_independent_confirmation": {"type": "boolean"},
        "issues": _array(issue), "query_candidate_counts": {
            "type": "object", "additionalProperties": {"type": "integer"}},
        "decision_boundary": string},
        "required": ["protocol", "status", "requires_independent_confirmation",
                     "issues", "query_candidate_counts", "decision_boundary"],
        "additionalProperties": False}
    return {"input_schema": {"type": "object", "properties": {
            "frame": _frame_schema(), "queries": _array(string), "camera": string,
            "box_threshold": number, "text_threshold": number,
            "max_detections_per_query": {"type": "integer", "minimum": 1, "maximum": 12},
            "distinct_query_pairs": _array(_array(string, 2))},
            "required": ["frame", "queries"], "additionalProperties": False},
        "output_schema": {"type": "object", "properties": {
            "frame_id": string, "camera": string,
            "detections": {"type": "object", "additionalProperties": _array(detection)},
            "reliability": reliability},
            "required": ["frame_id", "camera", "detections", "reliability"],
            "additionalProperties": False}}


def _grasp_contract():
    number = {"type": "number"}; string = {"type": "string"}
    vector = _array(number, 3); rotation = _array(_array(number, 3), 3)
    candidate = {"type": "object", "properties": {
        "rank_score": number, "model_score": number, "distance_to_target_m": number,
        "width_m": number, "height_m": number, "depth_m": number,
        "translation_camera": vector, "rotation_camera": rotation,
        "translation_world": vector, "rotation_world": rotation,
        "downward_score": number, "collision_iou": number, "inner_occupancy": number,
        "orientation_override_required": {"type": "boolean"}, "world_xyz": vector,
        "approach_world": vector, "eef_rotation_world": rotation,
        "pose_kind": {"type": "string", "enum": ["full_6dof", "calibrated_topdown"]}},
        "required": ["rank_score", "model_score", "distance_to_target_m", "width_m",
                     "translation_world", "rotation_world", "world_xyz",
                     "eef_rotation_world", "pose_kind"], "additionalProperties": False}
    scalar_map = {"type": "object", "additionalProperties": {
        "oneOf": [number, string, {"type": "boolean"}, _array(number),
                  {"type": "object"}, _array({"type": "object"})]}}
    return {"input_schema": {"type": "object", "properties": {
            "frame": _frame_schema(), "detection": _detection_schema(), "camera": string,
            "downward_min": number, "preferred_downward_min": number},
            "required": ["frame", "detection"], "additionalProperties": False},
        "output_schema": {"type": "object", "properties": {
            "frame_id": string, "target_center_world": vector,
            "full_6dof_grasps": _array(candidate),
            "calibrated_topdown_grasps": _array(candidate),
            "filter_thresholds": {"oneOf": [scalar_map, {"type": "null"}]},
            "filter_diagnostics": {"oneOf": [scalar_map, {"type": "null"}]},
            "artifact_path": string},
            "required": ["target_center_world", "full_6dof_grasps",
                         "calibrated_topdown_grasps", "filter_thresholds",
                         "filter_diagnostics", "artifact_path"],
            "additionalProperties": False}}


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paths():
    package_root = Path(__file__).resolve().parents[2]
    configuration = _vendor_configuration()
    dino_root = _configured_path("ROBOFORGE_GROUNDINGDINO_ROOT", "groundingdino",
                                 package_root / "third_party/GroundingDINO", configuration)
    sam_root = _configured_path("ROBOFORGE_SAM_ROOT", "segment_anything",
                                package_root / "third_party/segment-anything", configuration)
    grasp_root = _configured_path("ROBOFORGE_GRASPNET_ROOT", "graspnet",
                                  package_root / "third_party/graspnet-baseline", configuration)
    text_encoder = _configured_path("ROBOFORGE_GROUNDINGDINO_TEXT_ENCODER",
        "groundingdino_text_encoder", dino_root.parent / "bert-base-uncased",
        configuration)
    return {"package_root": package_root, "groundingdino_root": dino_root,
        "groundingdino_config": _path("ROBOFORGE_GROUNDINGDINO_CONFIG",
            str(dino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py")),
        "groundingdino_checkpoint": _path("ROBOFORGE_GROUNDINGDINO_CHECKPOINT",
            str(package_root / "checkpoints/groundingdino_swint_ogc.pth")),
        "groundingdino_text_encoder": text_encoder,
        "sam_root": sam_root, "sam_checkpoint": _path("ROBOFORGE_SAM_CHECKPOINT",
            str(package_root / "checkpoints/sam_vit_b_01ec64.pth")),
        "graspnet_root": grasp_root, "graspnet_checkpoint": _path(
            "ROBOFORGE_GRASPNET_CHECKPOINT",
            str(package_root / "checkpoints/graspnet-checkpoint-rs.tar"))}


def doctor_checks():
    paths = _paths(); modules = {}
    for name in ("torch", "torchvision", "transformers", "PIL", "scipy", "open3d",
                 "libero", "robosuite", "cv2", "numpy", "timm", "addict", "yapf",
                 "supervision", "pycocotools", "tensorboard"):
        modules[name] = importlib.util.find_spec(name) is not None
    sources = {"groundingdino": (paths["groundingdino_root"] / "groundingdino").is_dir(),
               "segment_anything": (paths["sam_root"] / "segment_anything").is_dir(),
               "graspnet": (paths["graspnet_root"] / "models/graspnet.py").is_file(),
               "graspnet_pointnet2_extension": any(
                   (paths["graspnet_root"] / "pointnet2/pointnet2").glob("_ext*.so")),
               "graspnet_knn_extension": any(
                   (paths["graspnet_root"] / "knn/knn_pytorch").glob("knn_pytorch*.so"))}
    checkpoints = {}
    for name, key in (("groundingdino", "groundingdino_checkpoint"),
                      ("sam", "sam_checkpoint"), ("graspnet", "graspnet_checkpoint")):
        path = paths[key]; actual = _sha256(path) if path.is_file() else None
        checkpoints[name] = {"path": str(path), "available": path.is_file(),
            "sha256": actual, "expected_sha256": _CHECKPOINT_SHA256[name],
            "valid": actual == _CHECKPOINT_SHA256[name]}
    text_encoder = paths["groundingdino_text_encoder"]
    encoder_files = {}
    for name, expected in _TEXT_ENCODER_SHA256.items():
        path = text_encoder / name
        actual = _sha256(path) if path.is_file() else None
        encoder_files[name] = {"available": path.is_file(), "sha256": actual,
                               "expected_sha256": expected, "valid": actual == expected}
    checkpoints["groundingdino_text_encoder"] = {
        "path": str(text_encoder),
        "available": all(item["available"] for item in encoder_files.values()),
        "valid": all(item["valid"] for item in encoder_files.values()),
        "files": encoder_files,
    }
    device = os.environ.get("ROBOFORGE_DEVICE", "cuda")
    accelerator = {"requested": device, "available": True}
    if device.startswith("cuda") and modules["torch"]:
        import torch
        accelerator["available"] = bool(torch.cuda.is_available())
        accelerator["device_count"] = int(torch.cuda.device_count())
    ok = (all(modules.values()) and all(sources.values())
          and all(item["valid"] for item in checkpoints.values())
          and accelerator["available"] is True)
    return {"ok": ok, "modules": modules, "sources": sources,
            "checkpoints": checkpoints, "accelerator": accelerator}


def create(*, task: str, state: int = 0, root: str | Path,
           configuration: dict | None = None):
    paths = _paths(); package_root = paths["package_root"]
    config = os.environ.get("LIBERO_CONFIG_PATH")
    episode = LiberoEpisode("libero_spatial", int(task), int(state), config_path=config,
                            case_handle=f"libero-task-{task}-state-{state}")
    adapter_configuration = dict(configuration or {})
    model_name = str(adapter_configuration.get("verifier_model")
                     or os.environ.get("ROBOFORGE_MODEL", "gpt-5.6-sol"))
    perception = OpenVocabularyRGBD(groundingdino_root=paths["groundingdino_root"],
        groundingdino_config=paths["groundingdino_config"],
        groundingdino_checkpoint=paths["groundingdino_checkpoint"],
        groundingdino_text_encoder=paths["groundingdino_text_encoder"],
        sam_root=paths["sam_root"], sam_checkpoint=paths["sam_checkpoint"],
        device=os.environ.get("ROBOFORGE_DEVICE", "cuda"))
    grasp = GraspNetRGBD(backend_script=package_root / "embodied_codex/capabilities/graspnet_backend.py",
                         checkpoint=paths["graspnet_checkpoint"],
                         source_root=paths["graspnet_root"], python=sys.executable)
    verifiers = {"visual_attachment": perception.verify_attachment,
                 "visual_support_relation": perception.verify_support_relation}
    outcome = None
    if (not adapter_configuration.get("disable_agent_verifier")
            and (os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY"))):
        provider = resolve_provider(provider=adapter_configuration.get("verifier_provider")
            or adapter_configuration.get("model_provider")
            or os.environ.get("ROBOFORGE_MODEL_PROVIDER"),
            base_url=adapter_configuration.get("verifier_base_url")
            or adapter_configuration.get("model_base_url")
            or os.environ.get("ROBOFORGE_MODEL_BASE_URL"))
        outcome = VLMVisualTaskOutcomeVerifier(api_key=provider.api_key,
            base_url=provider.endpoint, model=model_name,
            reasoning_effort=str(adapter_configuration.get("verifier_reasoning_effort")
                                 or "low")).verify
    capabilities = {"libero.rgbd_perception:v001": perception.detect,
                    "libero.grasp_proposals:v001": grasp.infer}
    contracts = {"libero.rgbd_perception:v001": _perception_contract(),
                 "libero.grasp_proposals:v001": _grasp_contract()}
    for contract in contracts.values():
        contract["consequence"] = "READ_ONLY"
    deployment = LiberoDeployment(episode=episode, artifact_dir=Path(root) / "adapter",
        capabilities=capabilities, capability_contracts=contracts, verifiers=verifiers,
        outcome_verifier=outcome)
    deployment.sdk_index = _sdk_index(capabilities, verifiers, contracts)
    return deployment
