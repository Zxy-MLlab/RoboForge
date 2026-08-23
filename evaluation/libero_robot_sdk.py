"""Sensor-only Robot SDK boundary for agent-authored LIBERO programs.

The adapter owns benchmark initialization and the simulator lifetime. Controller
programs receive opaque RGB-D frame handles, legal proprioception, public
perception/grasp tools, and bounded OSC actions. Reward, done, task files,
simulator identities, and evaluator state never cross the dispatch boundary.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import numpy as np
from jsonschema import Draft202012Validator, ValidationError

# Environment adaptation is deployment-owned. Pin the same complete LIBERO
# source tree (including init_files) used by the existing live controller.
LIBERO_SOURCE = Path(
    "/data/zxy/vla_agentic_harness_pi0_libero/code/openpi/third_party/libero"
)
os.environ.setdefault(
    "LIBERO_CONFIG_PATH",
    "/data/zxy/vla_agentic_harness_pi0_libero/configs/libero",
)
if str(LIBERO_SOURCE) not in sys.path:
    sys.path.insert(0, str(LIBERO_SOURCE))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import get_orientation_error

from controller_program_workspace import ControllerProgramWorkspace
from controller_graph_workspace import (
    ControllerGraphValidationError,
    ControllerGraphWorkspace,
)
from controller_program_runtime import sensor_only
from capability_workspace import CapabilityWorkspace
from runtime_capabilities import HOOK_CONTRACTS, validate_hook_output


LEGACY_CONTROLLER = Path(
    "/data/zxy/vla_agentic_harness_pi0_libero/scripts/run_groundingdino_controller.py"
)
LIBERO_HELPERS = LEGACY_CONTROLLER.parent
RESOLUTION = 512


class LiberoRobotSDKError(RuntimeError):
    pass


def libero_task_instruction(suite: str, task: int) -> str:
    """Return only the legal public language for an adapter-selected task."""
    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"unknown LIBERO benchmark suite: {suite}")
    instruction = str(suites[suite]().get_task(int(task)).language).strip()
    if not instruction:
        raise ValueError(f"LIBERO task has an empty instruction: {suite}:{task}")
    return instruction


def libero_task_state_count(suite: str, task: int) -> int:
    """Return adapter-owned initial-state cardinality without exposing state data."""
    suites = benchmark.get_benchmark_dict()
    if suite not in suites:
        raise ValueError(f"unknown LIBERO benchmark suite: {suite}")
    return len(suites[suite]().get_task_init_states(int(task)))


def locked_attachment_verification(
    object_xyz: Any,
    eef_xyz: Any,
    source_baseline_xyz: Any,
    gripper_width: float,
    visible_object_xyz: Any,
    *,
    max_eef_distance: float = 0.16,
    min_object_motion: float = 0.025,
    source_vacated_distance: float = 0.055,
) -> dict[str, Any]:
    """Verify attachment against an adapter-owned source baseline.

    The controller may suggest a prior coordinate for diagnostics, but it
    cannot choose the baseline used by this decision.
    """
    obj = np.asarray(object_xyz, dtype=float)
    eef = np.asarray(eef_xyz, dtype=float)
    baseline = np.asarray(source_baseline_xyz, dtype=float)
    visible_distances = []
    for value in visible_object_xyz or ():
        point = np.asarray(value, dtype=float)
        if point.shape == (3,) and np.isfinite(point).all():
            visible_distances.append(float(np.linalg.norm(point - baseline)))
    source_vacated = not visible_distances or min(visible_distances) > source_vacated_distance
    finite = (
        obj.shape == (3,) and eef.shape == (3,) and baseline.shape == (3,)
        and np.isfinite(obj).all() and np.isfinite(eef).all()
        and np.isfinite(baseline).all() and np.isfinite(float(gripper_width))
    )
    verified = bool(
        finite and source_vacated
        and np.linalg.norm(obj - eef) <= max_eef_distance
        and np.linalg.norm(obj - baseline) >= min_object_motion
        and float(gripper_width) <= 0.075
    )
    return {
        "verified": verified,
        "source_vacated": source_vacated,
        "nearest_visible_to_source_m": min(visible_distances) if visible_distances else None,
    }


def summarize_motion_outcome(
    target_xyz: Any,
    eef_before_xyz: Any,
    eef_after_xyz: Any,
    *,
    requested_repeat: int,
    tolerance_m: float = 0.025,
) -> dict[str, Any]:
    """Summarize legal proprioceptive evidence for one Cartesian command."""
    target = np.asarray(target_xyz, dtype=float)
    before = np.asarray(eef_before_xyz, dtype=float)
    after = np.asarray(eef_after_xyz, dtype=float)
    finite = all(
        value.shape == (3,) and np.isfinite(value).all()
        for value in (target, before, after)
    )
    initial_error = float(np.linalg.norm(before - target)) if finite else None
    final_error = float(np.linalg.norm(after - target)) if finite else None
    progress = (
        max(0.0, initial_error - final_error)
        if initial_error is not None and final_error is not None else None
    )
    reached = bool(final_error is not None and final_error <= float(tolerance_m))
    return {
        "target_eef_xyz": target.tolist() if target.shape == (3,) else None,
        "eef_before_xyz": before.tolist() if before.shape == (3,) else None,
        "eef_after_xyz": after.tolist() if after.shape == (3,) else None,
        "initial_error_m": initial_error,
        "final_error_m": final_error,
        "progress_m": progress,
        "requested_repeat": int(requested_repeat),
        "tolerance_m": float(tolerance_m),
        "reached_target": reached,
        "stalled": bool(
            not reached and progress is not None
            and progress < 0.0005
        ),
    }


def summarize_phase_motion(
    action_outcomes: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate legal proprioceptive motion evidence by controller phase."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in action_outcomes:
        phase = str(outcome.get("phase") or "unlabeled").strip().casefold()
        grouped.setdefault(phase, []).append(outcome)
    phases: dict[str, dict[str, Any]] = {}
    for phase, outcomes in grouped.items():
        final_errors = [
            float(item["final_error_m"])
            for item in outcomes
            if item.get("final_error_m") is not None
            and np.isfinite(float(item["final_error_m"]))
        ]
        reached = sum(bool(item.get("reached_target")) for item in outcomes)
        stalled = sum(bool(item.get("stalled")) for item in outcomes)
        commands = len(outcomes)
        phases[phase] = {
            "commands": commands,
            "reached": reached,
            "reach_rate": reached / commands if commands else 0.0,
            "stalled": stalled,
            "min_final_error_m": min(final_errors, default=None),
            "max_final_error_m": max(final_errors, default=None),
            "last_command_index": max(
                (int(item.get("command_index") or 0) for item in outcomes),
                default=0,
            ),
            "unreachable": bool(
                commands >= 2 and reached == 0
                and stalled >= max(1, commands // 2)
            ),
            "convergence_failed": bool(
                commands >= 5 and reached / commands < 0.20
            ),
        }
    unreachable = [
        phase for phase, summary in phases.items() if summary["unreachable"]
    ]
    dominant = max(
        unreachable,
        key=lambda phase: phases[phase]["last_command_index"],
        default=None,
    )
    failed_convergence = [
        phase for phase, summary in phases.items()
        if summary["convergence_failed"]
    ]
    dominant_convergence = max(
        failed_convergence,
        key=lambda phase: phases[phase]["last_command_index"],
        default=None,
    )
    return {
        "phases": phases,
        "dominant_unreachable_phase": dominant,
        "dominant_convergence_failure_phase": dominant_convergence,
    }


def diagnose_sensor_failure(
    *,
    instruction: str,
    execution_completed: bool,
    attachment_verified: bool,
    placement_verified: bool,
    verifications: list[Mapping[str, Any]],
    action_outcomes: list[Mapping[str, Any]],
    latest_declared: str = "",
) -> tuple[str, dict[str, Any]]:
    """Return a mechanism-level, evaluator-blind failure diagnosis."""
    phase_motion = summarize_phase_motion(action_outcomes)
    if not execution_completed:
        return "controller_program_error", phase_motion
    if attachment_verified and placement_verified:
        return "sensor_verification_passed", phase_motion
    if attachment_verified:
        if any(item.get("kind") == "support_relation" for item in verifications) \
                or "place" in latest_declared or "release" in latest_declared:
            return "placement_not_verified", phase_motion
        return "transport_not_verified", phase_motion

    language = " ".join(str(instruction).casefold().split())
    articulated_task = "drawer" in language or "cabinet" in language
    articulation_verifications = [
        item for item in verifications if item.get("kind") == "articulation"
    ]
    articulation_phases = {
        phase for phase in phase_motion["phases"]
        if any(token in phase for token in ("drawer", "handle", "articul"))
    }
    if articulated_task:
        if not articulation_phases and not articulation_verifications:
            return "articulation_not_attempted", phase_motion
        if not any(bool(item.get("verified")) for item in articulation_verifications):
            return "drawer_open_not_verified", phase_motion

    if any(item.get("kind") == "attachment" for item in verifications):
        return "attachment_not_verified", phase_motion

    convergence = phase_motion.get("dominant_convergence_failure_phase")
    if convergence:
        normalized = (
            "contact" if "contact" in convergence else
            "approach" if "approach" in convergence else
            "transport" if "transport" in convergence else
            "motion"
        )
        return f"{normalized}_convergence_failed", phase_motion

    unreachable = phase_motion.get("dominant_unreachable_phase")
    if unreachable:
        normalized = (
            "contact" if "contact" in unreachable else
            "approach" if "approach" in unreachable else
            "transport" if "transport" in unreachable else
            "motion"
        )
        return f"{normalized}_unreachable", phase_motion
    return "development_run_completed_without_verification", phase_motion


def mask_support_decision(
    metrics: Mapping[str, Any], height_error_m: float | None, *,
    min_containment: float = 0.55, min_clearance_ratio: float = 0.25,
    max_height_error_m: float = 0.08,
) -> bool:
    """Conservative sensor-only support decision from SAM footprint metrics."""
    try:
        containment = float(metrics["containment"])
        clearance = float(metrics["clearance_ratio"])
        height = float(height_error_m)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        np.isfinite([containment, clearance, height]).all()
        and containment >= float(min_containment)
        and clearance >= float(min_clearance_ratio)
        and height <= float(max_height_error_m)
    )


def _load_backend():
    if str(LIBERO_HELPERS) not in sys.path:
        sys.path.insert(0, str(LIBERO_HELPERS))
    spec = importlib.util.spec_from_file_location("libero_public_sensor_tools", LEGACY_CONTROLLER)
    if spec is None or spec.loader is None:
        raise LiberoRobotSDKError(f"cannot load public sensor tools: {LEGACY_CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def robot_sdk_contract() -> dict[str, Any]:
    """Return the environment-neutral contract visible to controller authors."""
    return {
        "observe": {
            "returns": {
                "frame_id": "opaque RGB-D frame handle",
                "step": "action step count",
                "eef_xyz": "robot-frame wrist XYZ",
                "eef_quat": "wrist quaternion",
                "gripper_qpos": "measured finger positions",
            }
        },
        "call_tool": {
            "grounded_detect": {
                "arguments": {"frame_id": "frame-XXXX", "queries": ["open vocabulary phrase"]},
                "returns": {
                    "detector": "model name",
                    "candidates": [{
                        "query": "matched phrase", "score": 0.0,
                        "bbox_xyxy": [0, 0, 1, 1], "xyz": [0.0, 0.0, 0.0],
                    }],
                },
            },
            "select_entities": {
                "arguments": {"frame_id": "frame-XXXX", "queries": ["phrases"]},
                "returns": {
                    "source": {"query": "source phrase", "bbox_xyxy": [0, 0, 1, 1], "xyz": [0.0, 0.0, 0.0]},
                    "target": {"query": "target phrase", "bbox_xyxy": [0, 0, 1, 1], "xyz": [0.0, 0.0, 0.0]},
                    "source_query": "source phrase", "target_query": "target phrase",
                },
            },
            "segment_box": {
                "arguments": {"frame_id": "frame-XXXX", "bbox_xyxy": [0, 0, 1, 1]},
                "returns": {
                    "xyz": [0.0, 0.0, 0.0], "bbox_xyxy": [0, 0, 1, 1],
                    "mask_id": "opaque reusable SAM mask handle",
                },
            },
            "capture_landmark_baseline": {
                "arguments": {
                    "frame_id": "frame-XXXX", "query": "visual landmark phrase",
                    "bbox_xyxy": [0, 0, 1, 1],
                },
                "returns": {
                    "baseline_id": "opaque adapter-owned landmark baseline",
                    "query": "visual landmark phrase", "xyz": [0.0, 0.0, 0.0],
                },
            },
            "verify_landmark_displacement": {
                "arguments": {
                    "frame_id": "fresh frame-XXXX",
                    "baseline_id": "opaque baseline from capture_landmark_baseline",
                    "min_horizontal_displacement_m": 0.04,
                },
                "returns": {
                    "kind": "articulation", "verified": False,
                    "baseline_xyz": [0.0, 0.0, 0.0],
                    "current_xyz": [0.0, 0.0, 0.0],
                    "horizontal_displacement_m": 0.0,
                },
            },
            "generate_grasps": {
                "arguments": {
                    "frame_id": "frame-XXXX", "bbox_xyxy": [0, 0, 1, 1],
                    "source_xyz": [0.0, 0.0, 0.0], "downward_min": 0.55,
                },
                "returns": {
                    "pool": "candidate pool kind",
                    "grasps": [{
                        "score": 0.0, "translation_world": [0.0, 0.0, 0.0],
                        "rotation_world": [[0.0, 0.0, 0.0]],
                        "approach_world": [0.0, 0.0, -1.0],
                    }],
                },
            },
            "verify_attachment": {
                "arguments": {
                    "frame_id": "frame-XXXX", "query": "object phrase",
                    "previous_object_xyz": [0.0, 0.0, 0.0],
                },
                "returns": {
                    "kind": "attachment", "verified": False,
                    "object_xyz": [0.0, 0.0, 0.0], "eef_xyz": [0.0, 0.0, 0.0],
                    "source_baseline_xyz": [0.0, 0.0, 0.0],
                    "source_vacated": False,
                    "nearest_visible_to_source_m": 0.0, "gripper_width": 0.0,
                },
            },
            "verify_support_relation": {
                "arguments": {
                    "frame_id": "frame-XXXX", "object_query": "object phrase",
                    "target_xyz": [0.0, 0.0, 0.0], "max_xy_error_m": 0.025,
                    "target_mask_id": "mask handle captured before placement (preferred)",
                },
                "returns": {
                    "kind": "support_relation", "verified": False,
                    "object_xyz": [0.0, 0.0, 0.0], "target_xyz": [0.0, 0.0, 0.0],
                    "xy_error_m": 0.0, "height_error_m": 0.0,
                    "mask_metrics": {"containment": 0.0, "clearance_ratio": 0.0},
                },
            },
        },
        "act": {
            "arguments": {
                "target_eef_xyz": [0.0, 0.0, 0.0],
                "gripper": "-1 open or 1 close",
                "orientation": "topdown, hold, or a quaternion array",
                "position_gain": "0.05..1.0",
                "max_translation_action": "0.05..1.0",
                "repeat": "1..20; controller owns outer loops and termination",
                "position_tolerance_m": (
                    "0.002..0.05; use <=0.01 for grasp/contact and a looser "
                    "tolerance only for free-space motion"
                ),
                "phase": "optional short semantic label such as grasp_contact or transport",
            },
            "returns": {
                "step": "action step count",
                "eef_xyz": "new robot-frame wrist XYZ",
                "gripper_qpos": "measured finger positions",
                "initial_error_m": "Cartesian error before this command",
                "error_m": "Cartesian error after this command",
                "progress_m": "nonnegative error reduction",
                "reached_target": "true only when error_m <= tolerance_m",
                "stalled": "true when an unreached command made negligible progress",
                "tolerance_m": "effective clipped position_tolerance_m",
            },
        },
        "forbidden": [
            "reward", "done", "check_success", "BDDL", "MuJoCo poses",
            "simulator segmentation IDs", "task/state branches",
        ],
    }


class LiberoRobotSDKAdapter:
    """One development episode behind the controller-program RPC boundary."""

    def __init__(
        self,
        *,
        suite: str,
        task: int,
        state: int,
        seed: int,
        output: str | Path,
        capability_workspace: str | Path | None = None,
        horizon: int = 1800,
    ) -> None:
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.backend = _load_backend()
        self.capability_store = (
            CapabilityWorkspace(
                capability_workspace,
                python="/data/zxy/envs/vla-report/bin/python",
            )
            if capability_workspace is not None else None
        )
        benchmark_suite = benchmark.get_benchmark_dict()[suite]()
        selected_task = benchmark_suite.get_task(int(task))
        bddl = Path(get_libero_path("bddl_files")) / selected_task.problem_folder / selected_task.bddl_file
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl), camera_heights=RESOLUTION,
            camera_widths=RESOLUTION, camera_depths=True,
            ignore_done=True, horizon=int(horizon),
        )
        self.instruction_text = str(selected_task.language)
        self.env.seed(int(seed))
        self.env.reset()
        self.obs = self.env.set_init_state(benchmark_suite.get_task_init_states(int(task))[int(state)])
        for _ in range(10):
            self.obs, _, _, _ = self.env.step([0.0] * 6 + [-1.0])
        self.frames: dict[str, Any] = {}
        self.sam_masks: dict[str, np.ndarray] = {}
        self.landmark_baselines: dict[str, dict[str, Any]] = {}
        self.video_frames: list[np.ndarray] = []
        self.trace: list[dict[str, Any]] = []
        self.verifications: list[dict[str, Any]] = []
        self.action_outcomes: list[dict[str, Any]] = []
        self.capability_hook_invocations: list[dict[str, Any]] = []
        self.entity_queries: tuple[str, ...] | None = None
        self.source_query: str | None = None
        self.source_baseline_xyz: np.ndarray | None = None
        self.step = 0
        self.closed = False
        # Adapter-only launch metadata is never returned over RPC.
        (self.output / "adapter_manifest.json").write_text(json.dumps({
            "protocol": "libero-robot-sdk-v1",
            "suite": suite, "task_selector": int(task), "state_selector": int(state),
            "seed": int(seed), "evaluator_called": False,
            "privileged_state_exposed_to_controller": False,
        }, indent=2) + "\n")

    def _frame(self, frame_id: Any):
        key = str(frame_id)
        if key not in self.frames:
            raise LiberoRobotSDKError(f"unknown frame_id: {key}")
        return key, self.frames[key]

    def _observe(self) -> dict[str, Any]:
        rgbd = self.backend.render_rgbd(
            self.env.env.sim, height=RESOLUTION, width=RESOLUTION,
            include_segmentation=False,
        )
        frame_id = f"frame-{len(self.frames) + 1:04d}"
        self.frames[frame_id] = rgbd
        cv2.imwrite(
            str(self.output / f"{frame_id}.png"),
            cv2.cvtColor(np.asarray(rgbd.rgb, np.uint8), cv2.COLOR_RGB2BGR),
        )
        np.savez_compressed(
            self.output / f"{frame_id}-rgbd.npz",
            depth=rgbd.depth, intrinsic=rgbd.intrinsic,
            camera_to_world=rgbd.camera_to_world,
        )
        if not self.video_frames:
            self.video_frames.append(np.ascontiguousarray(rgbd.rgb))
        result = {
            "frame_id": frame_id, "step": self.step,
            "eef_xyz": np.asarray(self.obs["robot0_eef_pos"], float).tolist(),
            "eef_quat": np.asarray(self.obs["robot0_eef_quat"], float).tolist(),
            "gripper_qpos": np.asarray(self.obs["robot0_gripper_qpos"], float).tolist(),
            "rgb_shape": list(np.asarray(rgbd.rgb).shape),
            "depth_units": "meters", "coordinate_frame": "robot_world",
        }
        self.trace.append({"event": "observe", **result})
        return result

    def _detect(self, frame_id: Any, queries: Any, prefix: str) -> dict[str, Any]:
        _, frame = self._frame(frame_id)
        clean_queries = tuple(dict.fromkeys(
            str(item).strip().lower() for item in (queries or ()) if str(item).strip()
        ))
        if not clean_queries or len(clean_queries) > 16:
            raise LiberoRobotSDKError("queries must contain 1..16 non-empty phrases")
        result = self.backend.run_groundingdino(
            frame, self.output, queries=clean_queries, prefix=prefix,
        )
        # Keep the RPC bounded even when an open-vocabulary query is noisy.
        candidates = sorted(
            result.get("candidates") or (), key=lambda item: float(item.get("score", 0)),
            reverse=True,
        )[:80]
        return {"detector": result.get("detector"), "candidates": candidates}

    def _instruction_entity_queries(self) -> tuple[str, ...]:
        if self.entity_queries is None:
            extracted = self.backend.extract_language_detector_queries(
                self.instruction_text, self.output,
            )
            self.entity_queries = tuple(dict.fromkeys(
                str(item).strip().lower() for item in extracted if str(item).strip()
            ))
        return self.entity_queries

    def _call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name == "describe_sdk":
            contract = robot_sdk_contract()
            contract["tested_capability_tools"] = (
                [
                    {
                        "tool_id": item["tool_id"],
                        "description": item.get("description", ""),
                        "compatible_hooks": item.get("compatible_hooks") or [],
                        "contracts": {
                            hook: HOOK_CONTRACTS[hook]
                            for hook in item.get("compatible_hooks") or []
                            if hook in HOOK_CONTRACTS
                        },
                    }
                    for item in self.capability_store.tested_tools()
                ]
                if self.capability_store is not None else []
            )
            return contract
        if name == "list_capability_tools":
            return {
                "tools": [] if self.capability_store is None else [
                    {
                        "tool_id": item["tool_id"],
                        "description": item.get("description", ""),
                        "compatible_hooks": item.get("compatible_hooks") or [],
                    }
                    for item in self.capability_store.tested_tools()
                ]
            }
        if name == "grounded_detect":
            return self._detect(
                arguments.get("frame_id"), arguments.get("queries"),
                f"rpc_detect_{len(self.trace):04d}",
            )
        if name == "select_entities":
            frame_id, frame = self._frame(arguments.get("frame_id"))
            supplied = [
                str(item).strip().lower() for item in arguments.get("queries") or ()
                if str(item).strip() and len(str(item).split()) <= 6
            ]
            detected = self._detect(
                frame_id,
                list(dict.fromkeys([*supplied, *self._instruction_entity_queries()])),
                f"rpc_select_{len(self.trace):04d}",
            )
            center = self.backend.estimate_workspace_center(frame)
            source, target, reason, audit, _, _ = self.backend.choose(
                self.instruction_text, detected, center, frame.rgb,
            )
            self.source_query = str(source.get("query") or "").strip() or None
            if self.source_baseline_xyz is None and source.get("xyz") is not None:
                candidate_baseline = np.asarray(source["xyz"], dtype=float)
                if candidate_baseline.shape == (3,) and np.isfinite(candidate_baseline).all():
                    self.source_baseline_xyz = candidate_baseline
            return {
                "source": source, "target": target,
                "source_query": self.source_query,
                "target_query": str(target.get("query") or "").strip() or None,
                "reason": reason, "audit": audit,
            }
        if name == "segment_box":
            _, frame = self._frame(arguments.get("frame_id"))
            bbox = [int(value) for value in arguments.get("bbox_xyxy") or ()]
            if len(bbox) != 4:
                raise LiberoRobotSDKError("bbox_xyxy must contain four integers")
            detection = {"bbox_xyxy": bbox, "xyz": self.backend.detector_box_xyz(frame, bbox)}
            refined = self.backend.sam_refine_detection(
                frame, detection, self.output, f"rpc_sam_{len(self.trace):04d}",
                self.backend.estimate_workspace_center(frame),
            )
            mask_id = f"mask-{len(self.sam_masks) + 1:04d}"
            self.sam_masks[mask_id] = np.asarray(refined["mask"], dtype=bool).copy()
            return {
                "xyz": np.asarray(refined["xyz"], float).tolist(),
                "bbox_xyxy": bbox, "mask_id": mask_id,
            }
        if name == "capture_landmark_baseline":
            frame_id, frame = self._frame(arguments.get("frame_id"))
            query = str(arguments.get("query") or "").strip().casefold()
            bbox = [int(value) for value in arguments.get("bbox_xyxy") or ()]
            if not query or len(query.split()) > 8:
                raise LiberoRobotSDKError("query must be a concise visual landmark phrase")
            if len(bbox) != 4:
                raise LiberoRobotSDKError("bbox_xyxy must contain four integers")
            detection = {"bbox_xyxy": bbox, "xyz": self.backend.detector_box_xyz(frame, bbox)}
            refined = self.backend.sam_refine_detection(
                frame, detection, self.output,
                f"rpc_landmark_baseline_{len(self.trace):04d}",
                self.backend.estimate_workspace_center(frame),
            )
            xyz = np.asarray(refined["xyz"], dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise LiberoRobotSDKError("landmark baseline has no finite RGB-D position")
            baseline_id = f"landmark-{len(self.landmark_baselines) + 1:04d}"
            self.landmark_baselines[baseline_id] = {
                "query": query, "xyz": xyz.copy(), "frame_id": frame_id,
                "bbox_xyxy": bbox,
            }
            return {"baseline_id": baseline_id, "query": query, "xyz": xyz.tolist()}
        if name == "verify_landmark_displacement":
            frame_id, frame = self._frame(arguments.get("frame_id"))
            baseline_id = str(arguments.get("baseline_id") or "")
            baseline = self.landmark_baselines.get(baseline_id)
            if baseline is None:
                raise LiberoRobotSDKError(f"unknown landmark baseline_id: {baseline_id}")
            minimum = float(np.clip(
                arguments.get("min_horizontal_displacement_m", 0.04), 0.02, 0.15
            ))
            detected = self._detect(
                frame_id, [baseline["query"]],
                f"rpc_landmark_verify_{len(self.trace):04d}",
            )
            prior = np.asarray(baseline["xyz"], dtype=float)
            visible = [
                item for item in detected["candidates"]
                if np.asarray(item.get("xyz") or (), dtype=float).shape == (3,)
                and np.isfinite(np.asarray(item["xyz"], dtype=float)).all()
            ]
            nearest = min(
                visible,
                key=lambda item: float(np.linalg.norm(
                    np.asarray(item["xyz"], dtype=float) - prior
                )),
                default=None,
            )
            current = (
                np.asarray(self.backend.sam_refine_detection_xyz(
                    frame, nearest, self.output,
                    f"rpc_landmark_verify_sam_{len(self.trace):04d}",
                    self.backend.estimate_workspace_center(frame),
                ), dtype=float)
                if nearest is not None else np.full(3, np.nan)
            )
            displacement = (
                float(np.linalg.norm(current[:2] - prior[:2]))
                if current.shape == (3,) and np.isfinite(current).all() else None
            )
            evidence = {
                "kind": "articulation",
                "verified": bool(displacement is not None and displacement >= minimum),
                "baseline_id": baseline_id,
                "query": baseline["query"],
                "baseline_xyz": prior.tolist(),
                "current_xyz": current.tolist(),
                "horizontal_displacement_m": displacement,
                "min_horizontal_displacement_m": minimum,
                "baseline_frame_id": baseline["frame_id"],
                "frame_id": frame_id,
                "step": self.step,
            }
            self.verifications.append(evidence)
            return evidence
        if name == "generate_grasps":
            _, frame = self._frame(arguments.get("frame_id"))
            bbox = np.asarray(arguments.get("bbox_xyxy") or (), dtype=int)
            source_xyz = np.asarray(arguments.get("source_xyz") or (), dtype=float)
            if bbox.shape != (4,) or source_xyz.shape != (3,):
                raise LiberoRobotSDKError("generate_grasps requires bbox_xyxy[4] and source_xyz[3]")
            stem = f"rpc_grasp_{len(self.trace):04d}"
            input_path = self.output / f"{stem}_input.npz"
            output_path = self.output / f"{stem}_graspnet_output.json"
            np.savez_compressed(
                input_path, rgb=frame.rgb, depth=frame.depth,
                intrinsic=frame.intrinsic, camera_to_world=frame.camera_to_world,
                bbox_xyxy=bbox, source_xyz=source_xyz,
                workspace_center=self.backend.estimate_workspace_center(frame),
            )
            _, audit = self.backend.run_graspnet(
                input_path, output_path,
                float(np.clip(arguments.get("downward_min", 0.55), 0.0, 1.0)),
            )
            grasps, pool = self.backend.merge_orientation_compatible_grasp_pools(
                audit, "robot-topdown",
            )
            normalized_grasps = []
            for candidate in grasps[:40]:
                item = dict(candidate)
                rotation = np.asarray(item.get("rotation_world") or (), dtype=float)
                item["approach_world"] = (
                    rotation[:, 0].tolist()
                    if rotation.shape == (3, 3) and np.isfinite(rotation).all()
                    else None
                )
                item["score"] = float(
                    item.get("rank_score", item.get("model_score", 0.0))
                )
                normalized_grasps.append(item)
            return {
                "pool": pool, "grasps": normalized_grasps,
                "filter_thresholds": audit.get("filter_thresholds"),
            }
        if name == "verify_attachment":
            frame_id = arguments.get("frame_id")
            _, frame = self._frame(frame_id)
            query = self.source_query or str(arguments.get("query") or "object")
            requested_previous = np.asarray(
                arguments.get("previous_object_xyz") or (), dtype=float
            )
            if requested_previous.shape != (3,):
                raise LiberoRobotSDKError("previous_object_xyz must contain three values")
            if self.source_baseline_xyz is None:
                raise LiberoRobotSDKError(
                    "verify_attachment requires a prior select_entities source baseline"
                )
            previous = self.source_baseline_xyz.copy()
            detected = self._detect(frame_id, [query], f"rpc_attach_{len(self.trace):04d}")
            visible = [item for item in detected["candidates"] if item.get("xyz")]
            eef = np.asarray(self.obs["robot0_eef_pos"], float)
            nearest = self.backend.detection_nearest_eef_pixel(frame, visible, eef)
            xyz = (
                self.backend.sam_refine_detection_xyz(
                    frame, nearest, self.output, f"rpc_attach_sam_{len(self.trace):04d}",
                    self.backend.estimate_workspace_center(frame),
                )
                if nearest is not None else np.full(3, np.nan)
            )
            width = float(np.sum(np.abs(np.asarray(self.obs["robot0_gripper_qpos"], float))))
            locked = locked_attachment_verification(
                xyz, eef, previous, width,
                [item.get("xyz") for item in visible],
            )
            evidence = {
                "kind": "attachment",
                "verified": locked["verified"],
                "object_xyz": xyz.tolist(), "eef_xyz": eef.tolist(),
                "source_baseline_xyz": previous.tolist(),
                "requested_previous_object_xyz": requested_previous.tolist(),
                "source_vacated": locked["source_vacated"],
                "nearest_visible_to_source_m": locked["nearest_visible_to_source_m"],
                "gripper_width": width,
                "frame_id": str(frame_id),
                "step": self.step,
            }
            self.verifications.append(evidence)
            return evidence
        if name == "verify_support_relation":
            frame_id = arguments.get("frame_id")
            _, frame = self._frame(frame_id)
            detected = self._detect(
                frame_id,
                [self.source_query or str(arguments.get("object_query") or "object")],
                f"rpc_support_{len(self.trace):04d}",
            )
            target = np.asarray(arguments.get("target_xyz") or (), dtype=float)
            if target.shape != (3,):
                raise LiberoRobotSDKError("target_xyz must contain three values")
            candidates = [item for item in detected["candidates"] if item.get("xyz")]
            nearest = min(
                candidates,
                key=lambda item: np.linalg.norm(np.asarray(item["xyz"], float)[:2] - target[:2]),
                default=None,
            )
            refined = (
                self.backend.sam_refine_detection(
                    frame, nearest, self.output, f"rpc_support_sam_{len(self.trace):04d}",
                    self.backend.estimate_workspace_center(frame),
                )
                if nearest is not None else None
            )
            object_xyz = (
                np.asarray(refined["xyz"], float)
                if refined is not None else np.full(3, np.nan)
            )
            xy_error = float(np.linalg.norm(object_xyz[:2] - target[:2])) if np.isfinite(object_xyz).all() else None
            max_xy = float(np.clip(arguments.get("max_xy_error_m", 0.025), 0.005, 0.08))
            height_error = float(abs(object_xyz[2] - target[2])) if np.isfinite(object_xyz).all() else None
            target_mask_id = arguments.get("target_mask_id")
            mask_metrics = None
            if target_mask_id is not None:
                target_mask = self.sam_masks.get(str(target_mask_id))
                if target_mask is None:
                    raise LiberoRobotSDKError(f"unknown target_mask_id: {target_mask_id}")
                if refined is not None:
                    mask_metrics = self.backend.mask_support_metrics(
                        np.asarray(refined["mask"], dtype=bool), target_mask,
                    )
            verified = (
                mask_support_decision(mask_metrics or {}, height_error)
                if target_mask_id is not None else
                bool(
                    xy_error is not None and xy_error <= max_xy
                    and height_error is not None and height_error <= 0.08
                )
            )
            evidence = {
                "kind": "support_relation",
                "verified": bool(verified),
                "object_xyz": object_xyz.tolist(), "target_xyz": target.tolist(),
                "xy_error_m": xy_error, "height_error_m": height_error,
                "max_xy_error_m": max_xy, "frame_id": str(frame_id),
                "target_mask_id": str(target_mask_id) if target_mask_id is not None else None,
                "mask_metrics": mask_metrics,
                "verification_method": (
                    "preplacement_target_sam_mask_overlap"
                    if target_mask_id is not None else "legacy_xyz_center_distance"
                ),
            }
            self.verifications.append(evidence)
            return evidence
        if self.capability_store is not None and ":v" in name:
            destination = self.capability_store.resolve(name)
            manifest = json.loads((destination / "manifest.json").read_text())
            if manifest.get("status") != "unit_tested":
                raise LiberoRobotSDKError(f"capability Tool is not unit-tested: {name}")
            hooks = [
                str(hook) for hook in manifest.get("compatible_hooks") or ()
                if hook in HOOK_CONTRACTS
            ]
            generic_contract = manifest.get("generic_contract") or {}
            if hooks and generic_contract:
                raise LiberoRobotSDKError(
                    f"capability Tool cannot mix predefined and generic contracts: {name}"
                )
            if len(hooks) != 1 and not generic_contract:
                raise LiberoRobotSDKError(
                    "capability Tool must expose one tested hook or generic schema contract: "
                    f"{name}"
                )
            if len(hooks) > 1:
                raise LiberoRobotSDKError(
                    f"capability Tool exposes ambiguous hook contracts: {name}"
                )
            hook = hooks[0] if hooks else "generic_capability"
            event = {"hook": hook, "tool_id": name, "applied": False}
            try:
                if generic_contract:
                    Draft202012Validator(
                        generic_contract["input_schema"]
                    ).validate(dict(arguments))
                invoked = self.capability_store.invoke(name, arguments)
                if not invoked.get("success"):
                    raise LiberoRobotSDKError(
                        str(invoked.get("reason") or "capability Tool failed")
                    )
                if generic_contract:
                    normalized = invoked.get("result")
                    Draft202012Validator(
                        generic_contract["output_schema"]
                    ).validate(normalized)
                    event["stage"] = generic_contract.get("stage", "generic")
                else:
                    normalized = validate_hook_output(
                        hook, invoked.get("result"), arguments
                    )
                event["applied"] = True
                self.capability_hook_invocations.append(event)
                return normalized
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                self.capability_hook_invocations.append(event)
                raise LiberoRobotSDKError(
                    f"capability Tool contract violation for {name}/{hook}: {exc}"
                ) from exc
        raise LiberoRobotSDKError(f"unknown public Tool: {name}")

    def _act(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        target = np.asarray(arguments.get("target_eef_xyz") or (), dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise LiberoRobotSDKError("target_eef_xyz must contain three finite values")
        if np.any(np.abs(target[:2]) > 1.5) or not -0.2 <= target[2] <= 2.0:
            raise LiberoRobotSDKError("target_eef_xyz is outside the generic robot safety envelope")
        gripper = float(arguments.get("gripper", -1.0))
        if gripper not in {-1.0, 1.0}:
            raise LiberoRobotSDKError("gripper must be -1 or 1")
        gain = float(np.clip(arguments.get("position_gain", 0.65), 0.05, 1.0))
        action_limit = float(np.clip(arguments.get("max_translation_action", 1.0), 0.05, 1.0))
        repeat = int(np.clip(arguments.get("repeat", 1), 1, 20))
        tolerance = float(np.clip(
            arguments.get("position_tolerance_m", 0.025), 0.002, 0.05
        ))
        orientation = arguments.get("orientation", "topdown")
        phase = str(arguments.get("phase") or "unlabeled")[:80]
        eef_before = np.asarray(self.obs["robot0_eef_pos"], float).copy()
        final_error = None
        for _ in range(repeat):
            eef = np.asarray(self.obs["robot0_eef_pos"], float)
            action = self.backend.move_to(eef, target, gripper, gain=gain)
            action[:3] = np.clip(action[:3], -action_limit, action_limit)
            if orientation == "topdown":
                desired = self.backend.TOPDOWN_QUAT
            elif orientation == "hold":
                desired = np.asarray(self.obs["robot0_eef_quat"], float)
            else:
                desired = np.asarray(orientation, dtype=float)
                if desired.shape != (4,) or not np.isfinite(desired).all():
                    raise LiberoRobotSDKError("orientation must be topdown, hold, or quaternion[4]")
            action[3:6] = np.clip(
                get_orientation_error(desired, np.asarray(self.obs["robot0_eef_quat"], float)) * 0.35,
                -1.0, 1.0,
            )
            self.obs, _, _, _ = self.env.step(action.tolist())
            self.step += 1
            current = np.asarray(self.obs["robot0_eef_pos"], float)
            final_error = float(np.linalg.norm(current - target))
            row = {
                "event": "act", "step": self.step, "target_eef_xyz": target.tolist(),
                "eef_xyz": current.tolist(), "action": action.tolist(), "error_m": final_error,
                "gripper_qpos": np.asarray(self.obs["robot0_gripper_qpos"], float).tolist(),
                "phase": phase,
            }
            self.trace.append(row)
            if self.step % 4 == 0:
                self.video_frames.append(np.ascontiguousarray(self.obs["agentview_image"][::-1]))
        outcome = summarize_motion_outcome(
            target,
            eef_before,
            np.asarray(self.obs["robot0_eef_pos"], float),
            requested_repeat=repeat, tolerance_m=tolerance,
        )
        outcome.update({
            "command_index": len(self.action_outcomes) + 1,
            "step_after": self.step,
            "gripper": gripper,
            "orientation": orientation if isinstance(orientation, str) else "quaternion",
            "position_gain": gain,
            "max_translation_action": action_limit,
            "position_tolerance_m": tolerance,
            "phase": phase,
        })
        self.action_outcomes.append(outcome)
        return {
            "step": self.step,
            "eef_xyz": np.asarray(self.obs["robot0_eef_pos"], float).tolist(),
            "eef_quat": np.asarray(self.obs["robot0_eef_quat"], float).tolist(),
            "gripper_qpos": np.asarray(self.obs["robot0_gripper_qpos"], float).tolist(),
            "error_m": final_error,
            "initial_error_m": outcome["initial_error_m"],
            "progress_m": outcome["progress_m"],
            "reached_target": outcome["reached_target"],
            "stalled": outcome["stalled"],
            "tolerance_m": outcome["tolerance_m"],
            "phase": phase,
        }

    def dispatch(self, method: str, arguments: Mapping[str, Any]) -> Any:
        if self.closed:
            raise LiberoRobotSDKError("adapter is closed")
        if method == "instruction":
            return self.instruction_text
        if method == "observe":
            return self._observe()
        if method == "call_tool":
            return self._call_tool(str(arguments.get("name") or ""), arguments.get("arguments") or {})
        if method == "act":
            return self._act(arguments.get("action") or {})
        if method == "record":
            event = sensor_only(arguments.get("event"))
            self.trace.append({"event": "controller_record", "payload": event})
            return {"recorded": True}
        raise LiberoRobotSDKError(f"unsupported SDK method: {method}")

    def sensor_evidence(
        self,
        runtime_report: Mapping[str, Any],
        *,
        controller_interface: str = "program",
    ) -> dict[str, Any]:
        attachment = [item for item in self.verifications if item["kind"] == "attachment"]
        support = [item for item in self.verifications if item["kind"] == "support_relation"]
        attachment_verified = any(bool(item["verified"]) for item in attachment)
        support_verified = False
        if len(support) >= 2 and all(bool(item["verified"]) for item in support[-2:]):
            first = np.asarray(support[-2]["object_xyz"], float)
            second = np.asarray(support[-1]["object_xyz"], float)
            support_verified = bool(np.isfinite(first).all() and np.isfinite(second).all()
                                    and np.linalg.norm(second - first) <= 0.015)
        completed = bool(runtime_report.get("execution_completed"))
        declared_failures = []
        for event in self.trace:
            if event.get("event") != "controller_record":
                continue
            payload = event.get("payload") or {}
            label = str(
                payload.get("result") or payload.get("event")
                or payload.get("reason") or payload.get("phase") or ""
            ).casefold()
            if label:
                declared_failures.append(label)
        latest_declared = declared_failures[-1] if declared_failures else ""
        controller_error = (
            "controller_graph_error"
            if controller_interface == "graph" else "controller_program_error"
        )
        conclusion = (
            controller_error if not completed else
            "sensor_verification_passed" if attachment_verified and support_verified else
            "placement_not_verified" if attachment_verified and (
                bool(support) or "place" in latest_declared
            ) else
            # A strict successful attachment is stronger evidence than an old
            # controller_record label such as "attachment_check". If no later
            # support evidence exists, the unresolved stage is transport/place.
            "transport_not_verified" if attachment_verified else
            "attachment_not_verified" if attachment else
            "development_run_completed_without_verification"
        )
        diagnostic_failure_class, phase_diagnostics = diagnose_sensor_failure(
            instruction=self.instruction_text,
            execution_completed=completed,
            attachment_verified=attachment_verified,
            placement_verified=support_verified,
            verifications=self.verifications,
            action_outcomes=self.action_outcomes,
            latest_declared=latest_declared,
        )
        return {
            "execution_completed": completed,
            "backend": "libero-robot-sdk-v1",
            "controller_interface": str(controller_interface),
            "graph_owns_control_flow": controller_interface == "graph",
            "program_owns_control_flow": controller_interface == "program",
            "steps": self.step,
            "attachment_verified": attachment_verified,
            "placement_verified": support_verified,
            "sensor_only_conclusion": conclusion,
            "diagnostic_failure_class": diagnostic_failure_class,
            "phase_diagnostics": phase_diagnostics,
            "capability_hook_invocations": self.capability_hook_invocations,
            "program_error": runtime_report.get("error"),
            "failed_rpc": next((
                {
                    "method": item.get("method"), "arguments": item.get("arguments"),
                    "error": item.get("error"),
                }
                for item in reversed(runtime_report.get("rpc_events") or ())
                if item.get("error")
            ), None),
            "verifications": self.verifications,
            "controller_records": [
                sensor_only(event.get("payload") or {})
                for event in self.trace
                if event.get("event") == "controller_record"
            ][-100:],
            "action_outcomes": self.action_outcomes,
            "control_diagnostics": {
                "commands": len(self.action_outcomes),
                "targets_reached": sum(
                    bool(item.get("reached_target")) for item in self.action_outcomes
                ),
                "targets_not_reached": sum(
                    not bool(item.get("reached_target")) for item in self.action_outcomes
                ),
                "stalled_commands": sum(
                    bool(item.get("stalled")) for item in self.action_outcomes
                ),
                "max_final_error_m": max(
                    (float(item["final_error_m"]) for item in self.action_outcomes
                     if item.get("final_error_m") is not None),
                    default=None,
                ),
            },
            "rpc_methods": [event.get("method") for event in runtime_report.get("rpc_events") or ()],
            "artifacts": {
                "trace": str(self.output / "trace.json"),
                "rollout": str(self.output / "rollout.mp4"),
            },
            "evaluator_visible_to_agent": False,
        }

    def close(self) -> None:
        if self.closed:
            return
        try:
            final = self.backend.render_rgbd(
                self.env.env.sim, height=RESOLUTION, width=RESOLUTION,
                include_segmentation=False,
            )
            self.video_frames.append(np.ascontiguousarray(final.rgb))
            cv2.imwrite(
                str(self.output / "final_rgb.png"),
                cv2.cvtColor(np.asarray(final.rgb, np.uint8), cv2.COLOR_RGB2BGR),
            )
            (self.output / "trace.json").write_text(json.dumps(sensor_only(self.trace), indent=2) + "\n")
            if self.video_frames:
                writer = cv2.VideoWriter(
                    str(self.output / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    15.0, (RESOLUTION, RESOLUTION),
                )
                for frame in self.video_frames:
                    writer.write(cv2.cvtColor(np.asarray(frame, np.uint8), cv2.COLOR_RGB2BGR))
                writer.release()
        finally:
            self.env.close()
            self.closed = True


def execute_libero_program(
    workspace: ControllerProgramWorkspace,
    program_id: str,
    *,
    suite: str,
    task: int,
    state: int,
    seed: int,
    output: str | Path,
    capability_workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a frozen program and return only its sensor-side report."""
    adapter = LiberoRobotSDKAdapter(
        suite=suite, task=task, state=state, seed=seed, output=output,
        capability_workspace=capability_workspace,
    )
    runtime_report: dict[str, Any] = {}
    try:
        runtime_report = workspace.execute(program_id, adapter.dispatch)
        evidence = adapter.sensor_evidence(runtime_report)
    finally:
        adapter.close()
    return {
        "success": True,
        "program_id": program_id,
        "controller_id": program_id,
        "execution_completed": bool(runtime_report.get("execution_completed")),
        "sensor_evidence": sensor_only(evidence),
        "runtime_trace": runtime_report.get("trace_path"),
        "evaluator_hidden": True,
    }


def execute_libero_graph(
    workspace: ControllerGraphWorkspace,
    graph_id: str,
    *,
    suite: str,
    task: int,
    state: int,
    seed: int,
    output: str | Path,
    capability_workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Execute a typed graph on one persistent adapter; return sensor evidence only."""
    # Compilation is deliberately outside adapter construction: invalid agent
    # code must not allocate a simulator/GPU episode.
    workspace.preflight(graph_id)
    adapter = LiberoRobotSDKAdapter(
        suite=suite, task=task, state=state, seed=seed, output=output,
        capability_workspace=capability_workspace,
    )
    runtime_report: dict[str, Any] = {}
    try:
        try:
            runtime_report = workspace.execute(
                graph_id, adapter.dispatch,
                initial_context={
                    "task_instruction": str(adapter.instruction_text),
                },
            )
        except ControllerGraphValidationError as exc:
            # Agent-authored graph/context contract failures are experiment
            # evidence, not Harness crashes. Preserve them for the next
            # autonomous revision without exposing evaluator state.
            runtime_report = {
                "execution_completed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "graph_id": graph_id,
                "node_trace": [],
                "rpc_events": [],
                "verified_prefix_aliases": [],
            }
        evidence = adapter.sensor_evidence(runtime_report, controller_interface="graph")
        evidence["controller_graph"] = {
            "graph_id": graph_id,
            "graph_outcome": runtime_report.get("graph_outcome"),
            "node_trace": runtime_report.get("node_trace") or [],
            "verified_prefix_aliases": (
                runtime_report.get("verified_prefix_aliases") or []
            ),
        }
    finally:
        adapter.close()
    return {
        "success": True,
        "graph_id": graph_id,
        "controller_id": graph_id,
        "execution_completed": bool(runtime_report.get("execution_completed")),
        "sensor_evidence": sensor_only(evidence),
        "evaluator_hidden": True,
    }


def execute_libero_program_sealed(
    workspace: ControllerProgramWorkspace,
    program_id: str,
    *,
    suite: str,
    task: int,
    state: int,
    seed: int,
    output: str | Path,
    capability_workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Run a frozen program and score once after its sensor-side execution.

    This entry point is intentionally absent from the Robot SDK registry. The
    evaluator value is written beneath ``_evaluator_only`` and is never part of
    controller RPC responses or sensor evidence.
    """
    output_path = Path(output).resolve()
    adapter = LiberoRobotSDKAdapter(
        suite=suite, task=task, state=state, seed=seed, output=output_path,
        capability_workspace=capability_workspace,
    )
    runtime_report: dict[str, Any] = {}
    evaluator_success = False
    try:
        runtime_report = workspace.execute(program_id, adapter.dispatch)
        evidence = adapter.sensor_evidence(runtime_report)
        # The only evaluator read. It occurs after the controller process has
        # finished and cannot influence any action or program mutation.
        evaluator_success = bool(adapter.env.check_success())
        evaluator_dir = output_path / "_evaluator_only"
        evaluator_dir.mkdir(exist_ok=False)
        (evaluator_dir / "result.json").write_text(json.dumps({
            "success": evaluator_success,
            "evaluator_calls": 1,
            "opened_after_controller_execution": True,
            "visible_to_controller": False,
        }, indent=2) + "\n")
        (output_path / "adapter_manifest.json").write_text(json.dumps({
            "protocol": "libero-robot-sdk-v1",
            "suite": suite, "task_selector": int(task),
            "state_selector": int(state), "seed": int(seed),
            "evaluator_called": True,
            "evaluator_calls": 1,
            "evaluator_opened_after_controller_execution": True,
            "privileged_state_exposed_to_controller": False,
        }, indent=2) + "\n")
    finally:
        adapter.close()
    return {
        "success": True,
        "program_id": program_id,
        "execution_completed": bool(runtime_report.get("execution_completed")),
        "sensor_evidence": sensor_only(evidence),
        "runtime_trace": runtime_report.get("trace_path"),
        "evaluator_result_path": str(output_path / "_evaluator_only" / "result.json"),
        "evaluator_hidden_during_execution": True,
    }


def execute_libero_graph_skill_sealed(
    workspace: Any,
    skill_id: str,
    *,
    suite: str,
    task: int,
    state: int,
    seed: int,
    output: str | Path,
    capability_workspace: str | Path | None = None,
) -> dict[str, Any]:
    """Run one frozen Graph Task Skill, then score behind the sealed barrier."""
    output_path = Path(output).resolve()
    adapter = LiberoRobotSDKAdapter(
        suite=suite, task=task, state=state, seed=seed, output=output_path,
        capability_workspace=capability_workspace,
    )
    runtime_report: dict[str, Any] = {}
    try:
        runtime_report = workspace.execute(skill_id, adapter.dispatch)
        evidence = adapter.sensor_evidence(
            runtime_report, controller_interface="graph"
        )
        evidence["controller_graph"] = {
            "graph_id": runtime_report.get("graph_id"),
            "graph_outcome": runtime_report.get("graph_outcome"),
            "node_trace": runtime_report.get("node_trace") or [],
            "verified_prefix_aliases": (
                runtime_report.get("verified_prefix_aliases") or []
            ),
        }
        evaluator_success = bool(adapter.env.check_success())
        evaluator_dir = output_path / "_evaluator_only"
        evaluator_dir.mkdir(exist_ok=False)
        (evaluator_dir / "result.json").write_text(json.dumps({
            "success": evaluator_success,
            "evaluator_calls": 1,
            "opened_after_controller_execution": True,
            "visible_to_controller": False,
        }, indent=2) + "\n")
        (output_path / "adapter_manifest.json").write_text(json.dumps({
            "protocol": "libero-robot-sdk-v1",
            "suite": suite, "task_selector": int(task),
            "state_selector": int(state), "seed": int(seed),
            "evaluator_called": True, "evaluator_calls": 1,
            "evaluator_opened_after_controller_execution": True,
            "privileged_state_exposed_to_controller": False,
            "controller_interface": "graph",
        }, indent=2) + "\n")
    finally:
        adapter.close()
    return {
        "success": True, "skill_id": skill_id,
        "execution_completed": bool(runtime_report.get("execution_completed")),
        "sensor_evidence": sensor_only(evidence),
        "evaluator_result_path": str(output_path / "_evaluator_only" / "result.json"),
        "evaluator_hidden_during_execution": True,
    }


__all__ = [
    "LiberoRobotSDKAdapter", "LiberoRobotSDKError", "execute_libero_program",
    "execute_libero_graph",
    "execute_libero_graph_skill_sealed",
    "execute_libero_program_sealed",
    "diagnose_sensor_failure", "libero_task_instruction",
    "libero_task_state_count",
    "locked_attachment_verification", "robot_sdk_contract",
    "mask_support_decision", "summarize_motion_outcome", "summarize_phase_motion",
]
