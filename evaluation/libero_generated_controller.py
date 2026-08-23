"""LIBERO backend for immutable Harness-generated controller programs.

This first backend reuses the audited live GroundingDINO/SAM/GraspNet
controller while keeping its evaluator result outside the authoring-agent
observation.  Unsupported closed-loop stages are reported explicitly instead
of being silently claimed as implemented.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from capability_library.runtime_capabilities import FrozenCapabilityRuntime

import cv2
import numpy as np


LEGACY_CONTROLLER = Path(
    "/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py"
)


def _load_backend():
    spec = importlib.util.spec_from_file_location(
        "embodied_frontier_live_controller", LEGACY_CONTROLLER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load controller backend: {LEGACY_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _region_change(initial: np.ndarray, final: np.ndarray, bbox: list[int]) -> float:
    height, width = initial.shape[:2]
    x0, y0, x1, y1 = (int(value) for value in bbox)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    before = initial[y0:y1, x0:x1].astype(np.float32)
    after = final[y0:y1, x0:x1].astype(np.float32)
    return float(np.mean(np.abs(before - after)) / 255.0)


def _conservative_support_recheck(evidence: Mapping[str, Any]) -> bool | None:
    """Recheck recorded legal mask/RGB-D evidence against Harness safety floors.

    Older controllers used permissive overlap thresholds. Returning ``None``
    preserves compatibility for old records that did not save mask metrics;
    whenever the required evidence exists, a rim-contact result cannot remain
    verified merely because the old controller labelled it so.
    """
    first = evidence.get("first") or {}
    second = evidence.get("second") or {}
    first_metrics = first.get("mask_metrics") or {}
    second_metrics = second.get("mask_metrics") or {}
    required = ("containment", "clearance_ratio", "centroid_x", "centroid_y")
    if not all(key in first_metrics and key in second_metrics for key in required):
        return None
    profile = evidence.get("support_relation_profile") or {}
    min_containment = max(0.70, float(profile.get("min_containment", 0.75)))
    min_clearance = max(0.60, float(profile.get("min_clearance_ratio", 0.75)))
    max_pixel_motion = min(8.0, float(profile.get("max_centroid_motion_px", 6.0)))
    max_world_motion = min(0.015, float(profile.get("max_world_motion_m", 0.010)))
    max_xy_error = min(0.025, float(profile.get("max_xy_center_error_m", 0.020)))
    centers = np.asarray([
        [first_metrics["centroid_x"], first_metrics["centroid_y"]],
        [second_metrics["centroid_x"], second_metrics["centroid_y"]],
    ], dtype=float)
    values = np.asarray([
        first_metrics["containment"], second_metrics["containment"],
        first_metrics["clearance_ratio"], second_metrics["clearance_ratio"],
        evidence.get("world_motion_m"),
        evidence.get("first_xy_center_error_m"),
        evidence.get("second_xy_center_error_m"),
    ], dtype=float)
    if not np.isfinite(centers).all() or not np.isfinite(values).all():
        return False
    return bool(
        bool(first.get("height_ok"))
        and bool(second.get("height_ok"))
        and min(float(first_metrics["containment"]), float(second_metrics["containment"]))
        >= min_containment
        and min(float(first_metrics["clearance_ratio"]), float(second_metrics["clearance_ratio"]))
        >= min_clearance
        and float(np.linalg.norm(centers[1] - centers[0])) <= max_pixel_motion
        and float(evidence["world_motion_m"]) <= max_world_motion
        and max(
            float(evidence["first_xy_center_error_m"]),
            float(evidence["second_xy_center_error_m"]),
        ) <= max_xy_error
    )


def _sensor_summary(run_dir: Path, result: Mapping[str, Any], controller_spec: Mapping[str, Any]) -> dict[str, Any]:
    initial = cv2.imread(str(run_dir / "current_rgb.png"), cv2.IMREAD_COLOR)
    final = cv2.imread(str(run_dir / "final_rgb.png"), cv2.IMREAD_COLOR)
    source = dict(result.get("selected_source") or {})
    target = dict(result.get("selected_target") or {})
    source_change = target_change = None
    if initial is not None and final is not None and initial.shape == final.shape:
        if source.get("bbox_xyxy"):
            source_change = _region_change(initial, final, source["bbox_xyxy"])
        if target.get("bbox_xyxy"):
            target_change = _region_change(initial, final, target["bbox_xyxy"])

    trace_path = run_dir / "trace.json"
    trace = json.loads(trace_path.read_text()) if trace_path.is_file() else []
    phases = []
    for row in trace:
        phase = row.get("phase") if isinstance(row, dict) else None
        if phase and phase not in phases:
            phases.append(phase)
    close_rows = [row for row in trace if isinstance(row, dict) and row.get("phase") == "close"]
    closed_width = None
    if close_rows and close_rows[-1].get("gripper_qpos"):
        values = np.asarray(close_rows[-1]["gripper_qpos"], dtype=float)
        closed_width = float(np.sum(np.abs(values)))

    phase_control_error = {}
    for phase in phases:
        errors = [float(row["error"]) for row in trace
                  if isinstance(row, dict) and row.get("phase") == phase
                  and isinstance(row.get("error"), (int, float))]
        if errors:
            phase_control_error[phase] = {
                "minimum_m": min(errors),
                "final_m": errors[-1],
                "maximum_m": max(errors),
                "samples": len(errors),
            }

    requested = set(controller_spec.get("stages") or ())
    closed_loop_backend = result.get("protocol") == "groundingdino-rgbd-closed-loop-v1"
    unsupported = []
    if not closed_loop_backend:
        unsupported = sorted(requested & {"verify_attachment", "verify_placement", "correct_or_regrasp"})
    elif "correct_or_regrasp" in requested and result.get("correction_status") == "not_yet_implemented":
        unsupported = ["correct_or_regrasp"]
    placement = result.get("placement_verification")
    transport_verified = False
    placement_verified = False
    support_relation_recheck = None
    if isinstance(placement, dict):
        transport_verified = bool((placement.get("post_transfer_attachment") or {}).get("verified"))
        support_relation_recheck = _conservative_support_recheck(placement)
        placement_verified = (
            bool(placement.get("verified"))
            and support_relation_recheck is not False
            and transport_verified
        )
        for correction in placement.get("correction_attempts") or ():
            correction_recheck = _conservative_support_recheck(
                correction.get("placement_evidence") or {}
            )
            if correction.get("verified_placement") and correction_recheck is not False:
                correction_transport = bool(
                    (correction.get("post_transfer_attachment") or {}).get("verified")
                )
                transport_verified = transport_verified or correction_transport
                placement_verified = placement_verified or correction_transport
    evidence = {
        "execution_completed": bool(trace),
        "backend": "live-groundingdino-sam-graspnet-v1",
        "implemented_stages": phases,
        "requested_stages_not_yet_closed_loop": unsupported,
        "language": result.get("language"),
        "selection_reason": result.get("selection_reason"),
        "selection_audit": result.get("selection_audit"),
        "source_region_initial_change": source_change,
        "target_region_initial_change": target_change,
        "gripper_width_after_close": closed_width,
        "phase_control_error": phase_control_error,
        "grasp_candidate": result.get("grasp_candidate"),
        "graspnet_selected": result.get("graspnet_selected"),
        "grasp_attempts": result.get("grasp_attempts"),
        "articulation": result.get("articulation"),
        "attachment_verified": result.get("attachment_verified"),
        "placement_verification": result.get("placement_verification"),
        "transport_verified": transport_verified,
        "placement_verified": placement_verified,
        "support_relation_recheck": support_relation_recheck,
        "correction_status": result.get("correction_status"),
        "capability_hook_invocations": result.get("capability_hook_invocations") or [],
        "artifacts": {
            key: str(run_dir / filename)
            for key, filename in {
                "initial_rgb": "current_rgb.png",
                "final_rgb": "final_rgb.png",
                "rollout": "rollout.mp4",
                "trace": "trace.json",
                "detections": "groundingdino_raw.json",
                "language_queries": "language_detector_queries.json",
                "relation_candidate_crops": "relation_candidate_crops.png",
                "sam_mask": "sam_mask.png",
                "grasp_candidates": "graspnet_output.json",
            }.items()
            if (run_dir / filename).is_file()
        },
        "sensor_only_conclusion": (
            "closed_loop_stage_unavailable" if unsupported else
            "attachment_not_verified" if result.get("attachment_verified") is False else
            "transport_not_verified" if isinstance(placement, dict) and not transport_verified else
            "placement_not_verified" if isinstance(placement,dict) and not placement_verified else
            "sensor_verification_passed" if result.get("attachment_verified") is True else
            "fixed_program_completed"
        ),
    }
    return evidence


def execute_controller(
    controller_spec: Mapping[str, Any],
    *,
    suite: str,
    task: int,
    state: int,
    seed: int,
    output: str | Path,
    controller_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a generated controller and publish separate agent/scorer views."""
    if suite != "libero_spatial":
        raise ValueError("only libero_spatial is supported by this backend")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    backend_root = destination / "backend"
    backend = _load_backend()
    runtime_bindings = controller_spec.get("runtime_capability_hooks") or {}
    capability_runtime = None
    if runtime_bindings:
        if controller_root is None:
            raise ValueError("controller_root is required for frozen capability hooks")
        capability_runtime = FrozenCapabilityRuntime(controller_root, runtime_bindings)

    def invoke_capability(hook: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if capability_runtime is None:
            return {"hook": hook, "applied": False, "error": "no_capability_bound"}
        return capability_runtime.invoke(hook, payload)

    requested_stages=set(controller_spec.get("stages") or ())
    if requested_stages & {"verify_attachment","verify_placement","correct_or_regrasp"}:
        result=backend.run_one_closed_loop(
            int(task),init_state=int(state),seed=int(seed),
            grasp_orientation=str(controller_spec.get("grasp_orientation","robot-topdown")),
            downward_min=float(controller_spec.get("fallback_downward_score",.55)),
            output_root=backend_root,
            max_grasp_attempts=int(controller_spec.get("max_grasp_attempts",3)),
            max_place_corrections=int(controller_spec.get("max_place_corrections",0)),
            detector_queries=tuple(controller_spec.get("detector_queries") or ()),
            enable_articulation=bool(requested_stages & {
                "detect_articulated_handle","open_drawer",
                "verify_articulation","reobserve_after_articulation"}),
            capability_hook=invoke_capability if capability_runtime is not None else None,
            capability_hooks=tuple(runtime_bindings),
        )
    else:
        result = backend.run_one(
            int(task),init_state=int(state),seed=int(seed),grasp_mode="graspnet",
            grasp_candidate=0,
            grasp_orientation=str(controller_spec.get("grasp_orientation", "robot-topdown")),
            downward_min=float(controller_spec.get("fallback_downward_score", 0.55)),
            output_root=backend_root,
        )
    produced = next(path for path in backend_root.iterdir() if path.is_dir())
    for artifact in produced.iterdir():
        target = destination / artifact.name
        if artifact.is_file():
            shutil.move(str(artifact), str(target))
    shutil.rmtree(backend_root)

    evaluator_success = bool(result.get("success"))
    scorer_report = {
        "protocol": "generated-controller-v1",
        "suite": suite,
        "task": int(task),
        "state": int(state),
        "seed": int(seed),
        "success": evaluator_success,
        "evaluator_calls": 1,
        "evaluator_used_for_action_selection": False,
    }
    evaluator_dir = destination / "_evaluator_only"
    evaluator_dir.mkdir(exist_ok=True)
    (evaluator_dir / "result.json").write_text(json.dumps(scorer_report, indent=2) + "\n")

    # The legacy result is rewritten without success before the authoring agent
    # can inspect it.  The scorer-only copy above is the authoritative score.
    sanitized_result = {key: value for key, value in result.items() if key != "success"}
    (destination / "result.json").write_text(json.dumps(sanitized_result, indent=2) + "\n")
    observation = _sensor_summary(destination, sanitized_result, controller_spec)
    (destination / "agent_observation.json").write_text(json.dumps(observation, indent=2) + "\n")
    return {
        "execution_completed": observation["execution_completed"],
        "agent_observation": str(destination / "agent_observation.json"),
        "evaluator_hidden": True,
    }


__all__ = ["execute_controller"]
