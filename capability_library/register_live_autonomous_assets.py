"""Register the live, no-simulator-state task 0/1/2 capability assets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from asset_registry import register_asset

LIB = str(ROOT / "library.json")
RUN = Path("/data/zxy/embodied_frontier/runs/privilege_ladder/final_live_autonomous_strict_v2")
TASKS = ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"]
REPORT = str(RUN / "FINAL_REPORT.md")

assets = [
    {
        "asset_id": "tool.live-groundingdino-rgb-detector.v1",
        "kind": "tool", "name": "Live Grounding DINO RGB detector", "version": "Swin-T OGC",
        "status": "cross_task_reused", "source_urls": ["https://github.com/IDEA-Research/GroundingDINO"],
        "sensors": ["RGB"], "input_schema": ["current_RGB", "text_queries"],
        "output_schema": ["normalized_boxes", "confidence"], "tested_tasks": TASKS, "reused_tasks": TASKS,
        "known_failures": ["semantic aliases across visually similar containers"],
        "checkpoint_sha256": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
        "source_revision": "856dde20aee659246248e20734ef9ba5214f5e44",
        "implementation": str(ROOT / "tools/groundingdino_detector.py"), "evidence": REPORT,
        "current_task_data_used": False, "privileged_state_used": False,
    },
    {
        "asset_id": "tool.sam-box-prompt-rgbd-segmenter.v1",
        "kind": "tool", "name": "SAM box-prompt RGB-D instance segmenter", "version": "ViT-B",
        "status": "cross_task_reused", "source_urls": ["https://github.com/facebookresearch/segment-anything"],
        "sensors": ["RGB", "RGB-D"], "input_schema": ["current_RGB-D", "detector_box", "camera_calibration"],
        "output_schema": ["instance_mask", "mask_center_world"], "tested_tasks": TASKS, "reused_tasks": TASKS,
        "known_failures": ["box prompt must isolate the intended instance"],
        "checkpoint_sha256": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
        "source_revision": "dca509fe793f601edb92606367a655c15ac00fdf",
        "implementation": str(ROOT / "tools/sam_box_segment.py"), "evidence": REPORT,
        "current_task_data_used": False, "privileged_state_used": False,
    },
    {
        "asset_id": "tool.graspnet-rgbd-adaptive-grasp.v1",
        "kind": "tool", "name": "GraspNet RGB-D adaptive grasp generator", "version": "checkpoint-rs epoch 18",
        "status": "cross_task_reused", "source_urls": ["https://github.com/graspnet/graspnet-baseline"],
        "sensors": ["RGB-D"], "input_schema": ["target_RGB-D_point_cloud", "SAM_mask_center", "camera_calibration"],
        "output_schema": ["ranked_6DoF_grasps", "filter_diagnostics"], "tested_tasks": TASKS, "reused_tasks": TASKS,
        "known_failures": ["oblique high-confidence candidates can be unstable for top-access execution"],
        "checkpoint_sha256": "60680087c61cba2b6791614fef1519071e294f6dcaf99b3f581bb95f7c51a868",
        "source_revision": "280c215129f759ed8649cb4e89fc5dfee55f4f80",
        "implementation": str(ROOT / "tools/graspnet_rgbd_grasp.py"), "evidence": REPORT,
        "current_task_data_used": False, "privileged_state_used": False,
    },
    {
        "asset_id": "skill.vlm-rgbd-physical-region-selection.v1",
        "kind": "skill", "name": "VLM RGB-D physical-region relation selection", "version": "1",
        "status": "cross_task_reused", "source_urls": ["external-model:gpt-5.6-sol"],
        "sensors": ["RGB", "RGB-D", "language"],
        "input_schema": ["numbered_current_RGB", "cross-query_regions", "region_XYZ", "instruction"],
        "output_schema": ["source_region", "reference_region", "target_region", "geometry_audit"],
        "tested_tasks": TASKS, "reused_tasks": TASKS,
        "known_failures": ["depends on detector recall and external VLM availability"],
        "implementation": "/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py",
        "evidence": REPORT, "current_task_data_used": False, "privileged_state_used": False,
    },
    {
        "asset_id": "skill.autonomous-rgbd-graspnet-pick-place.v1",
        "kind": "skill", "name": "Autonomous RGB-D GraspNet pick and place", "version": "1",
        "status": "cross_task_reused", "source_urls": ["local:capability_library/skills/autonomous-rgbd-graspnet-pick-place/SKILL.md"],
        "sensors": ["RGB", "RGB-D", "proprioception", "language"],
        "input_schema": ["instruction", "live_RGB-D", "camera_calibration", "proprioception"],
        "output_schema": ["success_only_trajectory", "model_audits", "controller_trace"],
        "tested_tasks": TASKS, "reused_tasks": TASKS,
        "known_failures": ["untested under real depth and calibration noise", "validated on one initial state per task"],
        "implementation": str(ROOT / "skills/autonomous-rgbd-graspnet-pick-place/SKILL.md"),
        "evidence": REPORT, "current_task_data_used": False, "privileged_state_used": False,
    },
]

for task_id, steps in ((0, 345), (1, 348), (2, 307)):
    task_dir = RUN / f"task{task_id}_state0_rankauto_graspnet_candidate0_robot-topdown"
    assets.append({
        "asset_id": f"experience.live-autonomous-libero-spatial-task-{task_id}-success.v1",
        "kind": "experience", "name": f"Live autonomous LIBERO Spatial task {task_id} success", "version": "1",
        "status": "development_validated", "source_urls": [],
        "sensors": ["RGB", "RGB-D", "proprioception", "language"],
        "interface": "live perception -> autonomous instance selection -> GraspNet pick/place -> post-run success",
        "tested_tasks": [f"libero_spatial:{task_id}"], "reused_tasks": [], "known_failures": [],
        "success": True, "steps": steps, "state": 0, "tcp_offset": None,
        "trajectory": str(task_dir / "trajectory.hdf5"), "result": str(task_dir / "result.json"),
        "visual_evidence": [str(task_dir / "candidate_regions.png"), str(task_dir / "sam_mask.png"), str(task_dir / "final_rgb.png")],
        "evidence": REPORT, "current_task_data_used": False, "privileged_state_used": False,
    })

for asset in assets:
    result = register_asset(asset, library_path=LIB)
    if not result["success"]:
        raise RuntimeError(result)
    print(result["asset_id"])
