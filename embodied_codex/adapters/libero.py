"""LIBERO Adapter plugin. All task execution still runs through kernel.AgentLoop."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from ..capabilities import (GraspNetRGBD, OpenVocabularyRGBD,
                            VLMVisualRelationGrounder, VLMVisualTaskOutcomeVerifier)
from ..deployments.libero import LiberoDeployment, LiberoEpisode


def _path(env_name: str, default: str) -> Path:
    return Path(os.environ.get(env_name, default)).expanduser().resolve()


def _contract():
    any_object = {"type": "object", "additionalProperties": True}
    return {"input_schema": any_object, "output_schema": any_object}


def create(*, task: str, state: int = 0, root: str | Path):
    package_root = Path(__file__).resolve().parents[2]
    config = os.environ.get("LIBERO_CONFIG_PATH")
    episode = LiberoEpisode("libero_spatial", int(task), int(state), config_path=config,
                            case_handle=f"libero-task-{task}-state-{state}")
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("APEX_API_KEY")
    base_url = os.environ.get("APEX_BASE_URL", "https://api.apexin.ai/v1")
    model_name = os.environ.get("ROBOFORGE_MODEL", "gpt-5.6-sol")
    dino_root = _path("ROBOFORGE_GROUNDINGDINO_ROOT", str(package_root / "third_party/GroundingDINO"))
    dino_config = _path("ROBOFORGE_GROUNDINGDINO_CONFIG", str(dino_root / "groundingdino/config/GroundingDINO_SwinT_OGC.py"))
    dino_checkpoint = _path("ROBOFORGE_GROUNDINGDINO_CHECKPOINT", str(package_root / "checkpoints/groundingdino_swint_ogc.pth"))
    sam_root = _path("ROBOFORGE_SAM_ROOT", str(package_root / "third_party/segment-anything"))
    sam_checkpoint = _path("ROBOFORGE_SAM_CHECKPOINT", str(package_root / "checkpoints/sam_vit_b_01ec64.pth"))
    grasp_checkpoint = _path("ROBOFORGE_GRASPNET_CHECKPOINT", str(package_root / "checkpoints/graspnet-checkpoint-rs.tar"))
    perception = OpenVocabularyRGBD(groundingdino_root=dino_root, groundingdino_config=dino_config,
        groundingdino_checkpoint=dino_checkpoint, sam_root=sam_root, sam_checkpoint=sam_checkpoint,
        device=os.environ.get("ROBOFORGE_DEVICE", "cuda"))
    grasp = GraspNetRGBD(backend_script=package_root / "embodied_codex/capabilities/graspnet_backend.py",
                         checkpoint=grasp_checkpoint, python=sys.executable)
    verifiers = {"visual_attachment": perception.verify_attachment,
                 "visual_support_relation": perception.verify_support_relation}
    outcome = None
    if key:
        outcome = VLMVisualTaskOutcomeVerifier(api_key=key, base_url=base_url, model=model_name).verify
    capabilities = {"libero.rgbd_perception:v001": perception.detect,
                    "libero.grasp_proposals:v001": grasp.infer}
    contracts = {tool_id: _contract() for tool_id in capabilities}
    deployment = LiberoDeployment(episode=episode, artifact_dir=Path(root) / "adapter",
        capabilities=capabilities, capability_contracts=contracts, verifiers=verifiers,
        outcome_verifier=outcome)
    deployment.sdk_index = {
        "protocol": "embodied-codex-libero-robot-sdk-v1",
        "operations": ["observe", "use", "act", "verify", "record"],
        "seed_tools": sorted(capabilities),
        "verifiers": sorted(verifiers),
    }
    return deployment
