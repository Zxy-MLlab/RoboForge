"""Register the frozen detector and validated RGB-D skills after evaluation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from asset_registry import register_asset, asset_id_for

LIB = str(Path(__file__).parent / "library.json")
assets = [
    {
        "asset_id": asset_id_for("tool", "groundingdino open vocabulary detector", "1"),
        "kind": "tool", "name": "GroundingDINO open-vocabulary detector", "version": "Swin-T OGC frozen checkpoint",
        "status": "cross_task_reused", "source_urls": ["https://github.com/IDEA-Research/GroundingDINO"],
        "sensors": ["RGB"], "interface": "RGB image + text query -> candidate boxes and confidence",
        "tested_tasks": ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"],
        "reused_tasks": ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"],
        "known_failures": ["multiple instances can share the same semantic label", "highest confidence is not always the referred instance"],
        "current_task_data_used": False, "privileged_state_used": False,
        "evidence": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino/final_task012_report.json",
    },
    {
        "asset_id": asset_id_for("skill", "rgbd instance relation selection", "1"),
        "kind": "skill", "name": "RGB-D instance and spatial-relation selection", "version": "1",
        "status": "cross_task_reused", "source_urls": [], "sensors": ["RGB", "RGB-D", "language"],
        "interface": "candidate boxes + deprojected XYZ + referring expression -> selected source/target",
        "tested_tasks": ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"],
        "reused_tasks": ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"],
        "known_failures": ["requires multiple candidate hypotheses when references are not detected"],
        "current_task_data_used": False, "privileged_state_used": False,
        "evidence": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/final_task012_report.json",
    },
    {
        "asset_id": asset_id_for("skill", "rgbd tcp contact calibration", "1"),
        "kind": "skill", "name": "RGB-D TCP contact-plane calibration", "version": "1",
        "status": "cross_task_reused", "source_urls": [], "sensors": ["RGB-D", "proprioception"],
        "interface": "visible surface XYZ -> TCP offset and staged Cartesian waypoints",
        "tested_tasks": ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"],
        "reused_tasks": ["libero_spatial:0", "libero_spatial:1", "libero_spatial:2"],
        "known_failures": ["contact offset is object-geometry and setup dependent"],
        "current_task_data_used": False, "privileged_state_used": False,
        "evidence": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/final_task012_report.json",
    },
    {
        "asset_id": asset_id_for("experience", "groundingdino libero spatial task 0 success", "1"),
        "kind": "experience", "name": "GroundingDINO LIBERO Spatial task 0 successful rollout", "version": "1",
        "status": "cross_task_reused", "source_urls": [], "sensors": ["RGB", "RGB-D", "proprioception"],
        "interface": "language + RGB-D candidate geometry -> verified pick/place trajectory",
        "tested_tasks": ["libero_spatial:0"], "reused_tasks": ["libero_spatial:0"],
        "known_failures": [], "current_task_data_used": False, "privileged_state_used": False,
        "tcp_offset": [0.0, -0.05, 0.008],
        "trajectory": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/task0_state0_rank1_y-0.050_z+0.008/trajectory.hdf5",
        "evidence": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/final_task012_report.json",
    },
    {
        "asset_id": asset_id_for("experience", "groundingdino libero spatial task 1 success", "1"),
        "kind": "experience", "name": "GroundingDINO LIBERO Spatial task 1 successful rollout", "version": "1",
        "status": "cross_task_reused", "source_urls": [], "sensors": ["RGB", "RGB-D", "proprioception"],
        "interface": "language + RGB-D candidate geometry -> verified pick/place trajectory",
        "tested_tasks": ["libero_spatial:1"], "reused_tasks": ["libero_spatial:1"],
        "known_failures": [], "current_task_data_used": False, "privileged_state_used": False,
        "tcp_offset": [0.0, -0.05, 0.008],
        "trajectory": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/task1_state0_rank1_y-0.050_z+0.008/trajectory.hdf5",
        "evidence": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/final_task012_report.json",
    },
    {
        "asset_id": asset_id_for("experience", "groundingdino libero spatial task 2 success", "1"),
        "kind": "experience", "name": "GroundingDINO LIBERO Spatial task 2 successful rollout", "version": "1",
        "status": "cross_task_reused", "source_urls": [], "sensors": ["RGB", "RGB-D", "proprioception"],
        "interface": "language + RGB-D candidate geometry -> verified pick/place trajectory",
        "tested_tasks": ["libero_spatial:2"], "reused_tasks": ["libero_spatial:2"],
        "known_failures": [], "current_task_data_used": False, "privileged_state_used": False,
        "tcp_offset": [0.0, -0.05, -0.02],
        "trajectory": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/task2_state0_rank1_y-0.050_z-0.020/trajectory.hdf5",
        "evidence": "/data/zxy/embodied_frontier/runs/privilege_ladder/p2_groundingdino_controller/final_task012_report.json",
    },
]
for a in assets:
    print(register_asset(a, library_path=LIB))
