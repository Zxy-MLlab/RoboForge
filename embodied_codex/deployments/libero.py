"""Evaluator-blind LIBERO deployment for Embodied Codex.

The deployment contains robot I/O only. It never exposes reward, done, BDDL,
simulator identities, object state, or task-specific controller logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
import uuid

import cv2
import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from ..adapters.libero_sdk import LIBERO_ROBOT_SDK_CONTRACT, validate_action,validate_verifier_request
from ..kernel.tools import CONSEQUENCE_LEVELS


PROPRIO = ("robot0_joint_pos","robot0_joint_vel","robot0_eef_pos",
           "robot0_eef_quat","robot0_gripper_qpos","robot0_gripper_qvel")
CAMERAS = ("agentview","robot0_eye_in_hand")
Capability = Callable[[Mapping[str,Any]],Mapping[str,Any]]


class LiberoDeploymentError(RuntimeError): pass


_PUBLIC_PATH_RE = re.compile(
    r"(?:file://)?(?:/root|/tmp|/workspace|/home)/[^\s,;]+"
)
_PUBLIC_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[=:]\s*[^\s,;]+"
)
_PRIVATE_EXECUTION_KEYS = {
    "base64",
    "benchmark_state",
    "case_handle",
    "done",
    "environment_identity",
    "hidden_evaluator",
    "image",
    "raw",
    "resume_token",
    "reward",
    "verification_receipt",
}


def _sanitize_public_text(value: Any) -> str:
    """Keep controller-facing diagnostics useful without exposing host data."""
    text = str(value)[:1000]
    text = _PUBLIC_PATH_RE.sub("<redacted-path>", text)
    return _PUBLIC_SECRET_RE.sub(
        lambda match: match.group(0).split("=", 1)[0].split(":", 1)[0]
        + "=<redacted>",
        text,
    )


def _bounded_public(value: Any, *, depth: int = 0) -> Any:
    """Project RPC facts to a bounded JSON shape for the coding Agent."""
    if depth >= 8:
        return "<truncated>"
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_public(item, depth=depth + 1)
            for key, item in list(value.items())[:64]
            if str(key) not in _PRIVATE_EXECUTION_KEYS
            and str(key).casefold() not in {"sim.data", "sim.model"}
            and not str(key).casefold().startswith(("_harness_", "privileged_"))
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_public(item, depth=depth + 1) for item in list(value)[:128]]
    if isinstance(value, str):
        return _sanitize_public_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _sanitize_public_text(value)


def _split_public_error(value: Any) -> tuple[str, str]:
    text = _sanitize_public_text(value)
    kind, separator, message = text.partition(":")
    if separator and kind and " " not in kind:
        return kind, message.strip()
    return "ControllerRuntimeError", text


def _public_execution_diagnostics(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Return only Controller-visible lifecycle, Tool, action, and state facts."""
    raw_error = execution.get("error")
    if execution.get("completed") is True and not raw_error:
        termination = "completed"
    elif raw_error == "controller timed out":
        termination = "timeout"
    elif raw_error == "controller exited":
        termination = "process_exit"
    elif raw_error == "controller process output exceeded the byte limit":
        termination = "output_limit"
    else:
        termination = "controller_error"

    tool_errors: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for index, raw_event in enumerate(execution.get("rpc_events") or []):
        if not isinstance(raw_event, Mapping):
            continue
        method = str(raw_event.get("method") or "")
        arguments = raw_event.get("arguments")
        if method == "use":
            result = raw_event.get("result")
            tool_result = result.get("result") if isinstance(result, Mapping) else None
            tool_error = (
                tool_result.get("tool_error")
                if isinstance(tool_result, Mapping)
                else None
            )
            if isinstance(tool_error, Mapping):
                tool_errors.append(
                    {
                        "index": index,
                        "tool_id": (
                            str((arguments or {}).get("tool_id") or "")
                            if isinstance(arguments, Mapping)
                            else ""
                        ),
                        "step": (
                            result.get("step")
                            if isinstance(result, Mapping)
                            else None
                        ),
                        "type": _sanitize_public_text(
                            tool_error.get("type") or "ToolError"
                        ),
                        "message": _sanitize_public_text(
                            tool_error.get("message") or "Tool failed"
                        ),
                    }
                )
            elif raw_event.get("error"):
                kind, message = _split_public_error(raw_event["error"])
                tool_errors.append(
                    {
                        "index": index,
                        "tool_id": (
                            str((arguments or {}).get("tool_id") or "")
                            if isinstance(arguments, Mapping)
                            else ""
                        ),
                        "step": None,
                        "type": kind,
                        "message": message,
                    }
                )
        if method == "act":
            action = {
                "index": index,
                "requested": _bounded_public(
                    arguments.get("action") if isinstance(arguments, Mapping) else {}
                ),
            }
            if "result" in raw_event:
                action["result"] = _bounded_public(raw_event["result"])
            if "error" in raw_event:
                kind, message = _split_public_error(raw_event["error"])
                action["error"] = {"type": kind, "message": message}
            if "state_before" in raw_event:
                action["state_before"] = _bounded_public(raw_event["state_before"])
            if "state_after" in raw_event:
                action["state_after"] = _bounded_public(raw_event["state_after"])
            actions.append(action)

    result: dict[str, Any] = {
        "controller_termination": termination,
        "tool_errors": tool_errors,
        "action_trace": actions,
        "sanitized_runtime_trace": _bounded_public(execution.get("rpc_events") or []),
    }
    if execution.get("result") is not None:
        result["controller_result"] = _bounded_public(execution["result"])
    if raw_error:
        kind, message = _split_public_error(raw_error)
        result["controller_error"] = {"type": kind, "message": message}
    if execution.get("stderr"):
        result["controller_stderr"] = _sanitize_public_text(execution["stderr"])
    return result


def _validated_rotation_matrix(value: Any) -> np.ndarray:
    """Validate a right-handed world-from-EEF rotation matrix."""
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise LiberoDeploymentError("rotation_matrix must contain 3x3 finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise LiberoDeploymentError("rotation_matrix must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise LiberoDeploymentError("rotation_matrix must be right-handed")
    return rotation


def _validated_quaternion(value: Any) -> np.ndarray:
    """Return a normalized robosuite quaternion in XYZW order."""
    quaternion = np.asarray(value, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise LiberoDeploymentError("quaternion_xyzw must contain 4 finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise LiberoDeploymentError("quaternion_xyzw must be nonzero")
    return quaternion / norm


def _quaternion_angle(first: Any, second: Any) -> float:
    """Shortest unsigned angular distance between XYZW quaternions."""
    a, b = _validated_quaternion(first), _validated_quaternion(second)
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


@dataclass(frozen=True)
class LiberoEpisode:
    suite: str; task_index: int; initial_state_index: int
    seed: int=7; image_size: int=256; horizon: int=1200
    config_path: str|None=None; warmup_steps: int=12
    # Opaque Harness-owned identity.  The benchmark state index remains sealed
    # deployment metadata and never becomes controller/model evidence.
    case_handle: str|None=None
    controller_mode: str = "OSC_POSE"


class LiberoDeployment:
    episodic_trials = True
    robot_sdk_contract = LIBERO_ROBOT_SDK_CONTRACT
    _OUTPUT_FIELDS = {name: set(spec.get("output_fields") or [])
                      for name, spec in LIBERO_ROBOT_SDK_CONTRACT["methods"].items()}
    def __init__(self, *, episode: LiberoEpisode, artifact_dir: str|Path,
                 capabilities: Mapping[str,Capability]|None=None,
                 capability_contracts: Mapping[str,Mapping[str,Any]]|None=None,
                 verifiers: Mapping[str,Capability]|None=None,
                 outcome_verifier: Capability|None=None):
        if episode.config_path: os.environ["LIBERO_CONFIG_PATH"]=str(Path(episode.config_path).resolve())
        os.environ.setdefault("MUJOCO_GL","egl")
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.episode=episode;self.artifact_dir=Path(artifact_dir).resolve()
        # LIBERO does not provide a physical resume protocol.  A resumed
        # Harness run therefore starts a fresh simulator generation and must
        # retain the previous generation's evidence instead of overwriting it.
        if self.artifact_dir.exists():
            self.artifact_dir = self.artifact_dir / f"restart-{uuid.uuid4().hex}"
        self.artifact_dir.mkdir(parents=True,exist_ok=False)
        suite=benchmark.get_benchmark_dict()[episode.suite]();task=suite.get_task(episode.task_index)
        bddl=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file)
        states=suite.get_task_init_states(episode.task_index)
        self._init_states = states
        self._suite = suite
        self.env=OffScreenRenderEnv(bddl_file_name=bddl,camera_names=list(CAMERAS),
            camera_heights=episode.image_size,camera_widths=episode.image_size,
            camera_depths=True,ignore_done=True,horizon=episode.horizon,
            controller=str(episode.controller_mode))
        self.env.seed(episode.seed);self.obs=None
        self._instruction=str(task.language);self.capabilities=dict(capabilities or {})
        self._native_capability_ids = frozenset(str(key) for key in self.capabilities)
        self.capability_contracts={str(key):dict(value) for key,value in
                                   (capability_contracts or {}).items()}
        if set(self.capabilities)!=set(self.capability_contracts):
            raise LiberoDeploymentError("every deployment Tool requires exactly one machine contract")
        for tool_id,contract in self.capability_contracts.items():
            try:
                Draft202012Validator.check_schema(contract["input_schema"])
                Draft202012Validator.check_schema(contract["output_schema"])
            except Exception as exc:
                raise LiberoDeploymentError(f"invalid deployment Tool contract {tool_id}: {exc}") from exc
        self.verifiers=dict(verifiers or {});self.references={};self._retired_references=set();self.trace=[];self.video=[]
        self.verified_attachments=set()
        self.step=0;self.warmup_control_steps=0;self.controller_control_steps=0
        self.closed=False;self.last_verify=False
        self.environment_generation=""
        self._controller_execution_sealed=False;self._evaluator_calls=0
        self.outcome_verifier=outcome_verifier;self._outcome_report=None
        self._outcome_after=None
        self._execution_index = 0
        self._execution_artifacts = {}
        self._execution_sensor_report = None
        self._controller_artifacts: dict[str, Path] = {}
        self._controller_artifact_paths: dict[Path, str] = {}
        # LIBERO init states can leave free objects several centimetres above
        # their support.  A generic no-motion settling period is part of the
        # deployment adapter, not learned task logic.  Reward/done/info remain
        # discarded exactly as during controller execution.
        self._warmup_steps=int(np.clip(episode.warmup_steps,0,60))
        self._gripper_fraction = 1.0
        self._reset_to_initial_condition()
        (self.artifact_dir/"deployment.json").write_text(json.dumps({
            "protocol":"embodied-codex-libero-deployment-v1","suite":episode.suite,
            "task_index":episode.task_index,"state_index":episode.initial_state_index,
            "instruction":self._instruction,"controller_visible":["language","RGB-D",
            "calibration","proprioception","Tool output","action history"],
            "controller_hidden":["reward","done","evaluator","BDDL","object state","sim IDs"],
            "adapter_warmup_steps":self._warmup_steps,
            "created_unix":time.time()},indent=2)+"\n")

    @property
    def instruction(self): return self._instruction

    @property
    def sim(self):
        # Internal adapter convenience.  Controller code receives only the
        # projected observation; the SDK never exposes this property through
        # its public method catalog.
        return self.env.sim

    def get_franka_libero_observation(self):
        """Return the ASPIRE/CaP-X FrankaLiberoEnv observation contract.

        This adapter-side projection is intentionally the only place that
        touches MuJoCo camera/body transforms.  The Controller receives the
        resulting robot-base-frame values, never ``sim.data``/``sim.model``.
        """
        from scipy.spatial.transform import Rotation
        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map
        sim = self.env.sim
        base_id = sim.model.body_name2id("robot0_base")
        base = np.eye(4, dtype=np.float64)
        base[:3, :3] = np.asarray(sim.data.xmat[base_id]).reshape(3, 3)
        base[:3, 3] = np.asarray(sim.data.xpos[base_id])
        base_inv = np.linalg.inv(base)
        out: dict[str, Any] = {}
        for name in CAMERAS:
            rgb = np.ascontiguousarray(self.obs[f"{name}_image"][::-1])
            depth_raw = np.asarray(self.obs[f"{name}_depth"][::-1]).squeeze()
            # Match ASPIRE's FrankaLiberoEnv exactly: use MuJoCo's raw camera
            # pose, then apply the two frame rotations it uses (Ry(pi) and
            # Rz(pi)).  robosuite.get_camera_extrinsic_matrix applies a
            # different Rx(pi) correction and is therefore not interchangeable.
            cam = np.eye(4, dtype=np.float64)
            cam_id = sim.model.camera_name2id(name)
            cam[:3, :3] = np.asarray(sim.data.cam_xmat[cam_id]).reshape(3, 3)
            cam[:3, 3] = np.asarray(sim.data.cam_xpos[cam_id])
            ry = np.diag([-1.0, 1.0, -1.0, 1.0])
            rz = np.diag([-1.0, -1.0, 1.0, 1.0])
            pose = base_inv @ cam @ ry @ rz
            depth = get_real_depth_map(sim, depth_raw)
            K = get_camera_intrinsic_matrix(sim, name, rgb.shape[1], rgb.shape[0])
            out[name] = {"images": {"rgb": rgb, "depth": depth},
                         "intrinsics": np.asarray(K), "pose_mat": pose,
                         "pose": np.r_[pose[:3, 3], self._matrix_to_wxyz(pose[:3, :3])].copy()}
        gripper_id = sim.model.body_name2id("gripper0_eef")
        gripper = np.eye(4, dtype=np.float64)
        gripper[:3, :3] = np.asarray(sim.data.xmat[gripper_id]).reshape(3, 3)
        gripper[:3, 3] = np.asarray(sim.data.xpos[gripper_id])
        # ASPIRE/CaP-X expose the raw ``gripper0_eef`` transform in the robot
        # base frame here. The Controller API applies its panda_hand/TCP
        # offset exactly once when forming an IK target. Applying another tool
        # transform to the observation shifts every Cartesian target by about
        # 10.7 cm and makes current-pose IK fail conformance.
        eef = base_inv @ gripper
        grip = float(np.asarray(self.obs["robot0_gripper_qpos"]).reshape(-1)[0] / 0.04)
        out["robot_joint_pos"] = np.r_[np.asarray(self.obs["robot0_joint_pos"], dtype=np.float64), grip]
        out["robot_cartesian_pos"] = np.r_[eef[:3, 3], self._matrix_to_wxyz(eef[:3, :3]), grip]
        return out

    @staticmethod
    def _matrix_to_wxyz(matrix):
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
        return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)

    @property
    def home_joint_position(self):
        return getattr(self, "_home_joint_position", None)

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def _step_once(self) -> None:
        dof = 7 if str(self.episode.controller_mode) == "JOINT_POSITION" else 6
        self._sim_step(np.r_[np.zeros(dof), 1.0 - 2.0 * self._gripper_fraction])

    def _panda_joint_positions(self) -> np.ndarray:
        """Read the live MuJoCo joint state used by upstream ASPIRE/CaP-X."""
        sim = getattr(self.env, "sim", None)
        if sim is not None:
            addresses = []
            for index in range(1, 8):
                address = sim.model.get_joint_qpos_addr(f"robot0_joint{index}")
                addresses.append(int(address[0] if isinstance(address, tuple) else address))
            return np.asarray(sim.data.qpos[addresses], dtype=np.float64)
        return np.asarray(self.obs["robot0_joint_pos"], dtype=np.float64)[:7]

    def move_to_joints_blocking(self, joints, *, tolerance: float = 0.01,
                                max_steps: int = 120,
                                stable_steps: int = 1) -> None:
        controller = getattr(self.env.robots[0], "controller", None)
        if controller is None or str(getattr(controller, "name", "")) != "JOINT_POSITION":
            raise LiberoDeploymentError(
                "blocking joint control requires controller_mode='JOINT_POSITION'"
            )
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        initial = self._panda_joint_positions()
        control_freq = float(getattr(self.env, "control_freq", 20.0))
        stable_required = max(1, int(stable_steps))
        stable_count = 0
        samples = []
        converged = False
        for step_index in range(int(max_steps)):
            current = self._panda_joint_positions()
            error = target - current
            max_error = float(np.max(np.abs(error)))
            l2_error = float(np.linalg.norm(error))
            samples.append({
                "step": step_index,
                "target_rad": target.tolist(),
                "actual_rad": current.tolist(),
                "max_error_rad": max_error,
                "l2_error_rad": l2_error,
            })
            if l2_error < float(tolerance):
                stable_count += 1
                if stable_count >= stable_required:
                    converged = True
                    break
            else:
                stable_count = 0
            action = np.r_[(target - current) * control_freq,
                           1.0 - 2.0 * self._gripper_fraction]
            self._sim_step(action)
        final = self._panda_joint_positions()
        final_error = target - final
        report = {
            "event": "joint_control",
            "controller": "JOINT_POSITION",
            "action_semantics": "relative_delta_scaled_by_control_frequency",
            "control_frequency_hz": control_freq,
            "target_rad": target.tolist(),
            "initial_rad": initial.tolist(),
            "final_rad": final.tolist(),
            "final_max_error_rad": float(np.max(np.abs(final_error))),
            "final_l2_error_rad": float(np.linalg.norm(final_error)),
            "tolerance_l2_rad": float(tolerance),
            "stable_steps_required": stable_required,
            "stable_steps_observed": stable_count,
            "steps_commanded": max(0, len(samples) - (1 if converged else 0)),
            "max_steps": int(max_steps),
            "status": "converged" if converged else "timeout",
            "samples": samples,
        }
        if hasattr(self, "trace"):
            self.trace.append(report)
        if not converged:
            raise LiberoDeploymentError(
                "blocking joint control did not converge; "
                f"initial_error_rad={np.linalg.norm(target - initial):.6f}; "
                f"final_error_rad={np.linalg.norm(target - final):.6f}; "
                f"steps={int(max_steps)}"
            )

    def execution_identity(self):
        return {"adapter":"libero","episode_id":self.episode.case_handle or
                f"{self.episode.suite}:{self.episode.task_index}:{self.episode.initial_state_index}",
                "environment_generation":self.environment_generation}

    def candidate_runtime_metadata(self):
        """Secret-free Provider/API/model identity bound into Candidate Bundles."""
        return dict(getattr(self, "_candidate_runtime_metadata", {}))

    def initial_observation(self):
        arguments={"channel":"rgbd","request":{}}
        return self.project_rpc_output("observe",arguments,self.dispatch("observe",arguments))

    def begin_controller_execution(self):
        return self.begin_execution("physical_trial")

    def begin_execution(self, kind="physical_trial"):
        """Start fresh execution-local state for either execution kind."""
        if self.closed:
            raise LiberoDeploymentError("deployment closed")
        if kind not in {"physical_trial", "diagnostic"}:
            raise LiberoDeploymentError("unsupported execution kind")
        self._execution_index += 1
        self.execution_kind = kind
        self.trace = []
        self.video = []
        self.references = {}
        self.verified_attachments = set()
        self.last_verify = False
        self._outcome_report = None
        self._outcome_after = None
        self._execution_sensor_report = None
        self._execution_artifacts = {}
        self._controller_artifacts = {}
        self._controller_artifact_paths = {}
        self.controller_control_steps = 0
        self.trial_horizon_exhausted = False
        self._controller_execution_sealed = False
        self._outcome_before = self._capture_outcome_rgb("before") if kind == "physical_trial" else None
        self._evaluator_calls = 0
        self.controller_control_steps = 0
        self.trial_horizon_exhausted = False

    def _finalize_execution_artifacts(self):
        directory = self.artifact_dir / "executions" / f"execution-{self._execution_index:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        trace_path = directory / "trace.json"
        trace_path.write_text(json.dumps(self.trace, indent=2, default=str) + "\n")
        trajectory_path = directory / "trajectory.npz"
        np.savez_compressed(
            trajectory_path,
            events=np.asarray(
                [json.dumps(item, sort_keys=True, default=str) for item in self.trace],
                dtype=str,
            ),
        )
        video_path = directory / "rollout.mp4"
        if self.video:
            h, w = self.video[0].shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            for frame in self.video:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        self._execution_artifacts = {"trace_path": str(trace_path),
                                     "rollout_path": str(video_path),
                                     "trajectory_path": str(trajectory_path)}

    def reset_case(self):
        if self.closed:
            raise LiberoDeploymentError("deployment closed")
        self._reset_to_initial_condition()
        return self.initial_observation()

    def _reset_to_initial_condition(self):
        if not hasattr(self, "_retired_references"):
            self._retired_references = set()
        self._retired_references.update(getattr(self, "references", {}).keys())
        self.obs = self.env.reset()
        self.obs = self.env.set_init_state(self._init_states[self.episode.initial_state_index])
        if isinstance(self.obs, Mapping) and "robot0_joint_pos" in self.obs:
            self._home_joint_position = np.asarray(self.obs["robot0_joint_pos"], dtype=np.float64)[:7].copy()
        else:
            # Lightweight test/development providers may not expose joints;
            # home control will then fail explicitly when requested.
            self._home_joint_position = None
        self.environment_generation = uuid.uuid4().hex
        self.step = 0; self.warmup_control_steps = 0; self.controller_control_steps = 0
        self.frame = 0; self.trace = []; self.video = []
        self.trial_horizon_exhausted = False
        self.references = {}; self.last_verify = False
        self.verified_attachments = set()
        self._controller_execution_sealed = False; self._evaluator_calls = 0
        self._outcome_report = None; self._outcome_after = None
        self._execution_sensor_report = None
        self._controller_artifacts = {}
        self._controller_artifact_paths = {}
        if self._warmup_steps:
            self._in_warmup = True
            for _ in range(self._warmup_steps):
                dof = 7 if str(self.episode.controller_mode) == "JOINT_POSITION" else 6
                self._sim_step(np.r_[np.zeros(dof),-1.0])
            self._in_warmup = False
            self.trace.append({"event":"adapter_warmup","steps":self._warmup_steps,
                               "controller_visible":True})
        # A physical trial is reset immediately after begin_execution().  The
        # reset establishes the actual episode initial state, so capture the
        # verifier's before frame here rather than retaining the stale frame
        # from before reset (or clearing it to None).
        if getattr(self, "execution_kind", "physical_trial") == "physical_trial":
            self._outcome_before = self._capture_outcome_rgb("before")
        else:
            self._outcome_before = None

    def resume_protocol(self):
        # A fresh simulator instance does not preserve physical state across a
        # Harness restart, so committed evidence is diagnostic only.
        return {"supports_resume":False,"environment_generation":self.environment_generation,
                "replay_allowed":True,"actions_idempotent":False}

    def validate_execution_receipt(self, receipt):
        return False

    def register_capability(self,tool_id,function,contract):
        if tool_id in self.capabilities: raise LiberoDeploymentError("duplicate Tool")
        value={key:dict(contract.get(key) or {}) for key in ("input_schema","output_schema")}
        consequence = str(contract.get("consequence", "UNKNOWN")).upper()
        if consequence not in CONSEQUENCE_LEVELS | {"UNKNOWN"}:
            raise LiberoDeploymentError("invalid capability consequence")
        value["consequence"] = consequence
        try:
            for schema in (value["input_schema"], value["output_schema"]):
                Draft202012Validator.check_schema(schema)
        except Exception as exc:
            raise LiberoDeploymentError(f"invalid dynamic Tool contract {tool_id}: {exc}") from exc
        self.capabilities[str(tool_id)]=function
        self.capability_contracts[str(tool_id)]=value

    def validate_capability_registration(self, tool_id, contract):
        if str(tool_id) in self.capabilities:
            raise LiberoDeploymentError("duplicate Tool")
        value = dict(contract or {})
        for key in ("input_schema", "output_schema"):
            Draft202012Validator.check_schema(dict(value.get(key) or {}))
        consequence = str(value.get("consequence", "UNKNOWN")).upper()
        if consequence not in CONSEQUENCE_LEVELS | {"UNKNOWN"}:
            raise LiberoDeploymentError("invalid capability consequence")

    def unregister_capability(self, tool_id):
        self.capabilities.pop(str(tool_id), None)
        self.capability_contracts.pop(str(tool_id), None)

    def capability_consequence(self, tool_id):
        contract = self.capability_contracts.get(str(tool_id), {})
        return str(contract.get("consequence", "UNKNOWN")).upper()

    def native_capability_index(self):
        rows = []
        for item in self.native_capability_manifest().values():
            detail = self.inspect_native_capability(item["capability_id"])
            manifest = detail.get("manifest", {})
            rows.append({**dict(item), "purpose": manifest.get("description", ""),
                         "description": manifest.get("description", ""), "source": "native"})
        return rows

    def native_capability_manifest(self):
        result = {}
        for capability_id in sorted(self._native_capability_ids):
            contract = self.capability_contracts[capability_id]
            encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"))
            version = capability_id.rpartition(":")[2] or None
            result[capability_id] = {"capability_id": capability_id,
                "version": version,
                "consequence": str(contract.get("consequence", "UNKNOWN")).upper(),
                "contract_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        return result

    def inspect_native_capability(self, capability_id: str):
        """Return the complete manual/contract for one Adapter-native Tool."""
        capability_id = str(capability_id)
        if capability_id not in self._native_capability_ids:
            raise LiberoDeploymentError("unknown native capability")
        contract = self.capability_contracts.get(capability_id)
        if not isinstance(contract, Mapping):
            raise LiberoDeploymentError("native capability contract is unavailable")
        manuals = {
            "libero.rgbd_perception:v001": {
                "purpose": "Detect language-named objects from public calibrated RGB-D observations.",
                "examples": [{"frame": "<public observation>", "queries": ["<object description>"]}],
                "failure_modes": ["contract_error", "no_detection", "ambiguous_detection",
                                  "sensor_failure"],
                "limitations": ["Results depend on visible RGB-D evidence and model confidence.",
                                "No simulator object state is available."],
            },
            "libero.grasp_proposals:v001": {
                "purpose": "Generate grasp candidates from a public RGB-D frame and detection.",
                "examples": [{"frame": "<public observation>",
                              "detection": "<perception detection>"}],
                "failure_modes": ["contract_error", "no_grasp_candidate", "sensor_failure"],
                "limitations": ["Candidates are proposals, not task-success guarantees.",
                                "The capability is optional for canonical robot control."],
            },
        }
        manual = dict(manuals.get(capability_id) or {
            "purpose": "Adapter-native embodied capability.", "examples": [],
            "failure_modes": ["contract_error", "sensor_failure"],
            "limitations": ["Behavior is bounded by the published input and output schemas."],
        })
        manual.update({"summary": "Callable through robot.use after explicit inspection.",
                       "provenance": "trusted Adapter implementation"})
        return {"manifest": {"tool_id": capability_id,
                              "capability_id": capability_id,
                              "version": capability_id.rpartition(":")[2] or None,
                              "description": manual["purpose"],
                              "input_schema": dict(contract.get("input_schema") or {}),
                              "output_schema": dict(contract.get("output_schema") or {}),
                              "consequence": str(contract.get("consequence", "UNKNOWN")).upper(),
                              "manual": manual},
                "source": None}

    def register_controller_artifact(self, path: str | Path) -> str:
        """Return an opaque Controller capability for one exact Adapter file."""
        source = Path(path).resolve()
        if not source.is_file():
            raise LiberoDeploymentError("Controller artifact is not an immutable file")
        if not (source == self.artifact_dir or self.artifact_dir in source.parents):
            raise LiberoDeploymentError("Controller artifact is outside the Adapter artifact store")
        existing = self._controller_artifact_paths.get(source)
        if existing is not None:
            return existing
        handle = f"artifact://sensor/{uuid.uuid4().hex}"
        self._controller_artifacts[handle] = source
        self._controller_artifact_paths[source] = handle
        return handle

    def resolve_controller_artifact(self, handle: str) -> Path:
        """Resolve only an exact handle previously issued by this deployment."""
        source = self._controller_artifacts.get(str(handle))
        if source is None or not source.is_file():
            raise LiberoDeploymentError("unknown or unavailable Controller artifact handle")
        return source

    def _tool_input(self, value):
        if isinstance(value, Mapping):
            return {str(key): self._tool_input(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._tool_input(item) for item in value]
        if isinstance(value, str):
            if value.startswith("artifact://"):
                return str(self.resolve_controller_artifact(value))
            if Path(value).is_absolute():
                raise LiberoDeploymentError(
                    "Controller Tool payloads cannot contain host filesystem paths")
        return value

    def _resolve_verifier_payload(self, value):
        """Resolve Controller-visible sensor handles for trusted native verifiers.

        Native verifiers run inside the Adapter process and therefore need the
        exact files behind opaque sensor handles.  Keep this separate from
        Shared ToolRuntime staging: verifier resolution never grants a Tool
        access to a directory or to an unregistered path.
        """
        if isinstance(value, Mapping):
            return {str(key): self._resolve_verifier_payload(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [self._resolve_verifier_payload(item) for item in value]
        if isinstance(value, str):
            if value.startswith("artifact://"):
                # Point references are opaque non-file identities and may be
                # carried alongside the frame.  Only sensor handles are
                # resolved to private files; unknown artifact URIs fail closed.
                if value in self.references:
                    return value
                if not value.startswith("artifact://sensor/"):
                    raise LiberoDeploymentError("verifier payload contains unsupported artifact handle")
                return str(self.resolve_controller_artifact(value))
            if Path(value).is_absolute():
                raise LiberoDeploymentError(
                    "verifier payloads cannot contain host filesystem paths")
        return value

    def _tool_output(self, value):
        if isinstance(value, Mapping):
            return {str(key): self._tool_output(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._tool_output(item) for item in value]
        if isinstance(value, str) and Path(value).is_absolute():
            source = Path(value).resolve()
            if source in self._controller_artifact_paths or (
                    source.is_file() and self.artifact_dir in source.parents):
                return self.register_controller_artifact(source)
            raise LiberoDeploymentError("Tool output attempted to expose a host filesystem path")
        return value

    def _proprio(self):
        return {key:np.asarray(self.obs[key]).tolist() for key in PROPRIO}

    def canonical_embodied_state(self):
        """Translate native observation fields into the Core public contract."""
        raw = self._proprio()
        qpos = raw["robot0_gripper_qpos"]
        return {"frames": {"world": {"name": "world", "parent": None}},
                "eef_frame": "world",
                "robot": {
                    "eef_pose": {"frame": "world", "position_m": raw["robot0_eef_pos"],
                                  "orientation_xyzw": raw["robot0_eef_quat"]},
                    # Width is factual proprioception.  Semantic open/closed
                    # interpretation requires an explicitly calibrated adapter.
                    "gripper": {"width_m": float(abs(qpos[0]) + abs(qpos[1]))},
                    "joint_state": {"position": raw["robot0_joint_pos"],
                                    "velocity": raw["robot0_joint_vel"],
                                    "gripper_velocity": raw["robot0_gripper_qvel"]},
                    "proprioception": {"joint_position": raw["robot0_joint_pos"],
                                       "joint_velocity": raw["robot0_joint_vel"]}},
                "observations": {"step": self.step}}

    def canonical_observation(self, observation):
        """Project a native observation for Core/context without native aliases."""
        if not isinstance(observation, Mapping):
            return observation
        result = dict(observation)
        canonical = self.canonical_embodied_state()
        result = {key: result[key] for key in ("frame_id", "step", "cameras") if key in result}
        result["proprioception"] = canonical["robot"]
        return result

    def dispatch(self,method,arguments):
        if self.closed:raise LiberoDeploymentError("deployment closed")
        if self._controller_execution_sealed:
            raise LiberoDeploymentError("controller execution already sealed")
        if method=="observe":return self._observe(str(arguments.get("channel") or "rgbd"),arguments.get("request") or {})
        if method=="act":return self._act(arguments.get("action") or {})
        if method=="use":return self._use(str(arguments.get("tool_id") or ""),arguments.get("payload") or {})
        if method in {"verify", "check_observable_condition"}:
            return self._verify(str(arguments.get("verifier") or ""),arguments.get("payload") or {})
        if method=="record":
            self.trace.append({"event":"controller_record","payload":arguments.get("event")});return {"recorded":True}
        if method=="sdk":
            name = str(arguments.get("method") or "")
            if not name or name.startswith("_"):
                raise LiberoDeploymentError("invalid Robot SDK method")
            if not hasattr(self, "_controller_sdk"):
                from ..adapters.franka_libero_api import FrankaLiberoApi
                self._controller_sdk = FrankaLiberoApi(self)
            fn = self._controller_sdk.functions().get(name)
            if not callable(fn):
                raise LiberoDeploymentError(f"unknown Robot SDK method: {name}")
            args = arguments.get("args") or []
            kwargs = arguments.get("kwargs") or {}
            if not isinstance(args, list) or not isinstance(kwargs, Mapping):
                raise LiberoDeploymentError("sdk args/kwargs must be JSON containers")
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                self.trace.append({"event": "sdk_call", "method": name,
                                   "status": "error", "error_type": type(exc).__name__,
                                   "error": str(exc)[:1000]})
                raise
            self.trace.append({"event": "sdk_call", "method": name, "status": "ok"})
            return {"method": name, "result": result}
        raise LiberoDeploymentError(f"unsupported method: {method}")

    @staticmethod
    def sdk_consequence(method: str) -> str:
        mutating = {
            "goto_pose", "open_gripper", "close_gripper", "goto_home_joint_position",
            "move_to_joints", "execute_joint_trajectory", "move_to_joints_arm0",
            "move_to_joints_arm1", "open_gripper_arm0", "close_gripper_arm0",
            "open_gripper_arm1", "close_gripper_arm1", "goto_pose_arm0",
            "goto_pose_arm1", "goto_pose_both",
        }
        return "PHYSICAL_CONTROL" if str(method) in mutating else "READ_ONLY"

    def project_rpc_output(self,method,arguments,result):
        """Positive projection of every Adapter response visible to Controller/GPT."""
        if method not in self._OUTPUT_FIELDS or not isinstance(result,Mapping):
            raise LiberoDeploymentError(f"invalid {method} Adapter output")
        unknown=set(str(key) for key in result)-self._OUTPUT_FIELDS[method]
        if unknown:raise LiberoDeploymentError(
            f"undeclared {method} output fields: {sorted(unknown)}")
        projected={str(key):value for key,value in result.items()
                   if str(key) in self._OUTPUT_FIELDS[method]}
        required={"observe":{"step"},"act":{"type","step","reached"},
                  "use":{"tool_id","step","result"},"verify":{"verified"},
                  "check_observable_condition":{"verified"},
                  "record":{"recorded"},"sdk":{"method","result"}}[method]
        missing=required-set(projected)
        if missing:raise LiberoDeploymentError(
            f"missing {method} output fields: {sorted(missing)}")
        if method == "sdk":
            projected["result"] = self._encode_sdk_result(projected["result"])
        return projected

    @classmethod
    def _encode_sdk_result(cls, value):
        """Losslessly encode ndarray type/shape for the isolated Controller."""
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {"__roboforge_ndarray__": True, "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "data_base64": base64.b64encode(array.tobytes()).decode("ascii")}
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Mapping):
            return {str(key): cls._encode_sdk_result(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._encode_sdk_result(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)): return value
        raise LiberoDeploymentError(f"Robot SDK returned unsupported type: {type(value).__name__}")

    def _observe(self,channel,request):
        if channel=="proprioception":
            return {"frame_id": f"proprio-{self.step:06d}", "step": self.step,
                    "proprioception": self.canonical_embodied_state()["robot"]}
        if channel not in ("rgb","rgbd"):raise LiberoDeploymentError("unsupported sensor channel")
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_camera_intrinsic_matrix,get_real_depth_map
        requested=request.get("cameras") or list(CAMERAS)
        if not isinstance(requested,list) or set(requested)-set(CAMERAS):raise LiberoDeploymentError("invalid cameras")
        self.frame+=1;frame_id=f"frame-{self.frame:06d}"
        folder=self.artifact_dir/"sensors"/self.environment_generation/frame_id
        folder.mkdir(parents=True,exist_ok=False);cameras={}
        for name in requested:
            rgb=np.ascontiguousarray(self.obs[f"{name}_image"][::-1]);rgb_path=folder/f"{name}_rgb.png"
            cv2.imwrite(str(rgb_path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            item={"rgb_path":self.register_controller_artifact(rgb_path),
                  "rgb_sha256":hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                  "shape":list(rgb.shape),"intrinsic":get_camera_intrinsic_matrix(self.env.sim,name,
                  self.episode.image_size,self.episode.image_size).tolist(),
                  "camera_to_world":get_camera_extrinsic_matrix(self.env.sim,name).tolist()}
            if channel=="rgbd":
                normalized=np.ascontiguousarray(self.obs[f"{name}_depth"][::-1])
                depth=np.asarray(get_real_depth_map(self.env.sim,normalized),np.float32)
                depth_path=folder/f"{name}_depth_m.npy";np.save(depth_path,depth)
                item.update({"depth_path":self.register_controller_artifact(depth_path),
                             "depth_sha256":hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                             "depth_range_m":[float(np.nanmin(depth)),float(np.nanmax(depth))]})
            cameras[name]=item
        report={"frame_id":frame_id,"step":self.step,"cameras":cameras,
                "proprioception": self.canonical_embodied_state()["robot"]}
        (folder/"observation.json").write_text(json.dumps(report,indent=2)+"\n")
        self.trace.append({"event":"observe","frame_id":frame_id,"step":self.step});return report

    def _sim_step(self,action):
        controller_steps = int(getattr(self, "controller_control_steps", self.step) or 0)
        if not getattr(self, "_in_warmup", False) and controller_steps >= self.episode.horizon:
            self.trial_horizon_exhausted = True
            raise LiberoDeploymentError("action horizon exhausted")
        obs,_reward,_done,_info=self.env.step(np.clip(action,-1,1).tolist())
        self.obs=obs;self.step+=1
        if getattr(self, "_in_warmup", False):
            self.warmup_control_steps += 1
        else:
            self.controller_control_steps += 1
        if self.step%3==0:self.video.append(np.ascontiguousarray(self.obs["agentview_image"][::-1]))

    def _capture_outcome_rgb(self, name):
        folder=self.artifact_dir/"executions"/f"execution-{self._execution_index:06d}" if self._execution_index else self.artifact_dir/"outcome"
        folder.mkdir(parents=True,exist_ok=True)
        external=np.ascontiguousarray(self.obs["agentview_image"][::-1])
        wrist=np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1])
        if wrist.shape[:2]!=external.shape[:2]:
            wrist=cv2.resize(wrist,(external.shape[1],external.shape[0]),
                             interpolation=cv2.INTER_AREA)
        # A single hashed montage keeps the VLM API contract compact while
        # adding the wrist view needed to disambiguate top-view overlap from a
        # real carried object. Labels describe only camera provenance.
        rgb=np.ascontiguousarray(np.concatenate([external,wrist],axis=1))
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        cv2.putText(bgr,"EXTERNAL",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,
                    (0,255,0),1,cv2.LINE_AA)
        cv2.putText(bgr,"WRIST",(external.shape[1]+8,20),
                    cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,0),1,cv2.LINE_AA)
        path=folder/f"{name}_external_wrist_montage.png"
        cv2.imwrite(str(path),bgr)
        return {"rgb_path":str(path),
                "rgb_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape":list(rgb.shape),"views":["agentview","robot0_eye_in_hand"],
                "layout":"external_left_wrist_right"}

    def _act(self,action):
        kind=validate_action(action);before=np.asarray(self.obs["robot0_eef_pos"],float).copy();target=None
        target_source = None
        if kind=="move_to_point":
            ref=str(action.get("target_ref") or "")
            if ref:
                if ref not in self.references:
                    if ref in self._retired_references: raise LiberoDeploymentError("reference belongs to a previous environment generation")
                    raise LiberoDeploymentError("unknown target_ref")
                target=np.asarray(self.references[ref]["world_xyz"],float)
                target_source="reference"
            else:
                if action.get("frame") != "world":
                    raise LiberoDeploymentError("unsupported coordinate frame")
                target=np.asarray(action.get("position_m"),float)
                if target.shape!=(3,) or not np.isfinite(target).all():
                    raise LiberoDeploymentError("invalid numeric target")
                target_source="controller_numeric"
            target=target+np.asarray(action.get("offset") or [0,0,0],float)
            if target.shape!=(3,) or not np.isfinite(target).all():raise LiberoDeploymentError("invalid target")
            tolerance=float(np.clip(action.get("tolerance_m",.015),.002,.06));gain=float(np.clip(action.get("gain",20),1,30))
            maximum=int(np.clip(action.get("max_steps",50),1,100));gripper=float(action.get("gripper",-1));reached=False
            for _ in range(maximum):
                error=target-np.asarray(self.obs["robot0_eef_pos"],float)
                if np.linalg.norm(error)<=tolerance:reached=True;break
                command=np.zeros(7);command[:3]=np.clip(error*gain,-1,1);command[6]=gripper;self._sim_step(command)
            reached=reached or np.linalg.norm(target-np.asarray(self.obs["robot0_eef_pos"],float))<=tolerance
        elif kind=="move_to_pose":
            # A pose may use a live Tool reference or an explicit numeric
            # metric target computed from public Controller information.
            ref=str(action.get("pose_ref") or action.get("target_ref") or "")
            if ref:
                if ref not in self.references:
                    if ref in self._retired_references: raise LiberoDeploymentError("reference belongs to a previous environment generation")
                    raise LiberoDeploymentError("unknown pose_ref/target_ref")
                reference=self.references[ref]
                target=np.asarray(reference["world_xyz"],float)
                target_source="reference"
            else:
                if action.get("frame") != "world":
                    raise LiberoDeploymentError("unsupported coordinate frame")
                reference={}
                target=np.asarray(action.get("position_m"),float)
                target_source="controller_numeric"
            target=target+np.asarray(action.get("offset") or [0,0,0],float)
            if target.shape!=(3,) or not np.isfinite(target).all():raise LiberoDeploymentError("invalid target")
            if "quaternion_xyzw" in action:
                target_quaternion=_validated_quaternion(action["quaternion_xyzw"])
            elif "rotation_matrix" in action:
                from robosuite.utils.transform_utils import mat2quat
                target_quaternion=_validated_quaternion(mat2quat(_validated_rotation_matrix(action["rotation_matrix"])))
            elif "eef_rotation_world" in reference:
                from robosuite.utils.transform_utils import mat2quat
                target_quaternion=_validated_quaternion(mat2quat(
                    np.asarray(reference["eef_rotation_world"],dtype=float)))
            else:
                raise LiberoDeploymentError("move_to_pose requires quaternion_xyzw or rotation_matrix")
            position_tolerance=float(np.clip(action.get("position_tolerance_m",.012),.002,.06))
            orientation_tolerance=float(np.clip(action.get("orientation_tolerance_rad",.08),.02,.5))
            position_gain=float(np.clip(action.get("position_gain",20),1,30))
            orientation_gain=float(np.clip(action.get("orientation_gain",.35),.05,1.0))
            maximum=int(np.clip(action.get("max_steps",100),1,180));gripper=float(action.get("gripper",-1))
            reached=False
            from robosuite.utils.transform_utils import get_orientation_error
            for _ in range(maximum):
                position_error=target-np.asarray(self.obs["robot0_eef_pos"],float)
                current_quaternion=_validated_quaternion(self.obs["robot0_eef_quat"])
                angle_error=_quaternion_angle(target_quaternion,current_quaternion)
                if np.linalg.norm(position_error)<=position_tolerance and angle_error<=orientation_tolerance:
                    reached=True;break
                command=np.zeros(7)
                command[:3]=np.clip(position_error*position_gain,-1,1)
                command[3:6]=np.clip(get_orientation_error(target_quaternion,current_quaternion)*orientation_gain,-1,1)
                command[6]=gripper;self._sim_step(command)
            final_position_error=float(np.linalg.norm(target-np.asarray(self.obs["robot0_eef_pos"],float)))
            final_orientation_error=_quaternion_angle(target_quaternion,self.obs["robot0_eef_quat"])
            reached=reached or (final_position_error<=position_tolerance and
                                final_orientation_error<=orientation_tolerance)
        elif kind=="osc_delta":
            command=np.r_[np.asarray(action.get("translation") or [0,0,0],float),
                          np.asarray(action.get("rotation") or [0,0,0],float),float(action.get("gripper",-1))]
            if command.shape!=(7,):raise LiberoDeploymentError("invalid OSC command")
            for _ in range(int(np.clip(action.get("repeat",1),1,20))):self._sim_step(command)
            reached=True
        elif kind=="gripper":
            command=np.zeros(7);command[6]=-1 if action.get("command")=="open" else 1
            for _ in range(int(np.clip(action.get("repeat",12),1,40))):self._sim_step(command)
            reached=True
        elif kind=="settle":
            command=np.zeros(7);command[6]=float(action.get("gripper",-1))
            for _ in range(int(np.clip(action.get("steps",10),1,60))):self._sim_step(command)
            reached=True
        else:raise LiberoDeploymentError(f"unsupported action: {kind}")
        result={"type":kind,"step":self.step,"reached":bool(reached),"eef_before":before.tolist(),
                "eef_after":np.asarray(self.obs["robot0_eef_pos"]).tolist(),
                "gripper_width_m":float(np.abs(np.asarray(self.obs["robot0_gripper_qpos"], dtype=float)).sum())}
        if kind in ("move_to_point", "move_to_pose"):
            reference = self.references.get(str(action.get("pose_ref") or action.get("target_ref")), {})
            axis = reference.get("action_frame_axis") or reference.get("approach_world")
            if axis is not None:
                result["action_frame_axis"] = list(axis)
                result["action_frame_axis_frame"] = "world"
        if target is not None:
            result["target_xyz"]=target.tolist();result["target_frame"]="world"
            result["target_source"]=target_source
        if kind=="move_to_pose":
            result.update({"target_quaternion_xyzw":target_quaternion.tolist(),
                           "final_position_error_m":final_position_error,
                           "final_orientation_error_rad":final_orientation_error})
        self.trace.append({"event":"act","request":dict(action),"result":result});return result

    def _references(self,tool_id,value):
        if isinstance(value,Mapping):
            result={str(k):self._references(tool_id,v) for k,v in value.items()}
            xyz=result.get("world_xyz")
            if isinstance(xyz,list) and len(xyz)==3 and np.isfinite(np.asarray(xyz,float)).all():
                token="point-"+uuid.uuid4().hex[:16];reference={"world_xyz":xyz,"tool_id":tool_id,
                    "environment_generation":self.environment_generation}
                bounds=result.get("world_bounds_10_90")
                if bounds is not None:
                    metric_bounds=np.asarray(bounds,float)
                    if metric_bounds.shape==(2,3) and np.isfinite(metric_bounds).all():
                        reference["world_bounds_10_90"]=metric_bounds.tolist()
                rotation=result.get("eef_rotation_world")
                if rotation is not None:
                    reference["eef_rotation_world"]=_validated_rotation_matrix(rotation).tolist()
                    result["pose_ref"]=token
                axis=result.get("approach_world") or result.get("action_frame_axis")
                if axis is not None:
                    axis_array=np.asarray(axis,float)
                    if axis_array.shape != (3,) or not np.isfinite(axis_array).all() or np.linalg.norm(axis_array) < 1e-8:
                        raise LiberoDeploymentError("action frame axis must be a finite non-zero vector")
                    reference["action_frame_axis"]=(axis_array / np.linalg.norm(axis_array)).tolist()
                self.references[token]=reference
                result["point_ref"]=token
            return result
        if isinstance(value,list):return [self._references(tool_id,v) for v in value]
        return value

    def _use(self,tool_id,payload):
        if tool_id not in self.capabilities:raise LiberoDeploymentError(f"unregistered Tool: {tool_id}")
        try:
            Draft202012Validator(self.capability_contracts[tool_id]["input_schema"]).validate(payload)
            if tool_id in self._native_capability_ids:
                raw_result=self.capabilities[tool_id](self._tool_input(dict(payload)))
            else:
                # Shared ToolRuntime receives opaque handles and resolves only
                # the exact files named by this payload inside its sandbox.
                raw_result=self.capabilities[tool_id](dict(payload))
        except ValidationError as exc:
            result={"tool_error":{"type":"ToolContractError","message":str(exc.message)[:1000]},
                    "ok":False}
            self.trace.append({"event":"use","tool_id":tool_id,"step":self.step,
                               "tool_error":result["tool_error"]})
            return {"tool_id":tool_id,"step":self.step,"result":result}
        except Exception as exc:
            # A remote foundation-model outage or public capability failure is
            # task evidence, not a controller-program crash.  Preserve the
            # direct Tool-result contract with a fail-closed structured value
            # so generated code can retry, switch capability, or return a
            # sensor_failure.  SDK misuse (unknown Tool id) is still rejected
            # above and therefore remains a Harness/controller error.
            result={"tool_error":{"type":type(exc).__name__,
                                  "message":str(exc)[:1000]},"ok":False}
            self.trace.append({"event":"use","tool_id":tool_id,"step":self.step,
                               "tool_error":result["tool_error"]})
            return {"tool_id":tool_id,"step":self.step,"result":result}
        contract=self.capability_contracts[tool_id]
        try:
            raw_result=self._tool_output(raw_result)
            Draft202012Validator(contract["output_schema"]).validate(raw_result)
        except ValidationError as exc:
            result={"tool_error":{"type":"ToolContractError",
                    "message":f"output: {exc.message}"[:1000]},"ok":False}
            self.trace.append({"event":"use","tool_id":tool_id,"step":self.step,
                               "tool_error":result["tool_error"]})
            return {"tool_id":tool_id,"step":self.step,"result":result}
        result=self._references(tool_id,raw_result)
        receipt={"tool_id":tool_id,"step":self.step,"result":result}
        self.trace.append({"event":"use","tool_id":tool_id,"step":self.step});return receipt

    def project_public_entities(self, tool_id, result):
        """Translate a capability's native detections at the Adapter boundary."""
        if not isinstance(result, Mapping):
            return []
        grouped = result.get("detections")
        candidates = []
        if isinstance(grouped, Mapping):
            for label, values in grouped.items():
                if isinstance(values, list):
                    candidates.extend((label, item) for item in values)
        elif isinstance(grouped, list):
            candidates.extend((None, item) for item in grouped)
        entities = []
        for label, item in candidates:
            if not isinstance(item, Mapping) or not isinstance(item.get("world_xyz"), list):
                continue
            entity_id = item.get("point_ref") or item.get("entity_id")
            if not entity_id:
                continue
            perception = {}
            for key in ("mask_path", "rgb_path", "depth_path", "box_xyxy"):
                if key in item:
                    perception[{"mask_path": "mask_ref", "rgb_path": "rgb_ref",
                                 "depth_path": "depth_ref", "box_xyxy": "bbox"}[key]] = item[key]
            entities.append({"entity_id": str(entity_id),
                             "label": item.get("label") or label,
                             "confidence": item.get("score"),
                             "geometry": {"frame": "world", "center": list(item["world_xyz"]),
                                          "bounds": item.get("world_bounds_10_90")},
                             "perception": perception,
                             "uncertainty": {},
                             "provenance": {"tool_id": str(tool_id)}})
        return entities

    def _verify(self,name,payload):
        validate_verifier_request(name,payload)
        expanded=dict(payload)
        source_ref=str(payload.get("source_ref") or "")
        transport_ref=str(payload.get("transport_ref") or source_ref)
        for key in ("source_ref","target_ref"):
            if key in expanded:
                ref=str(expanded[key])
                if ref not in self.references:raise LiberoDeploymentError(f"unknown {key}")
                expanded[key.replace("_ref","_world_xyz")]=self.references[ref]["world_xyz"]
                bounds=self.references[ref].get("world_bounds_10_90")
                if bounds is not None:
                    expanded[key.replace("_ref","_world_bounds_10_90")]=bounds
        if name=="visual_support_relation":
            if transport_ref not in self.references:
                raise LiberoDeploymentError("unknown transport_ref")
            expanded["source_transport_verified"]=(
                transport_ref in self.verified_attachments)
        if name not in self.verifiers:raise LiberoDeploymentError(f"unknown verifier: {name}")
        try:
            expanded=self._resolve_verifier_payload(expanded)
            result=dict(self.verifiers[name](expanded))
        except LiberoDeploymentError as exc:
            result={"verified":False,"sensor_only":True,
                    "verifier_error":{"type":type(exc).__name__,
                                      "message":str(exc)[:1000]}}
            self.trace.append({"event":"verify","name":name,"result":result})
            return result
        except Exception as exc:
            result={"verified":False,"sensor_only":True,
                    "verifier_error":{"type":type(exc).__name__,
                                      "message":str(exc)[:1000]}}
            self.trace.append({"event":"verify","name":name,"result":result})
            return result
        if not isinstance(result.get("verified"),bool):raise LiberoDeploymentError("verifier contract")
        result=self._tool_output(result)
        if name=="visual_attachment" and result["verified"] and source_ref:
            self.verified_attachments.add(source_ref)
        self.last_verify=result["verified"];self.trace.append({"event":"verify","name":name,"result":result});return result

    def sensor_report(self,execution):
        if self._execution_sensor_report is not None:
            return dict(self._execution_sensor_report)
        independent=True
        if getattr(self, "execution_kind", "physical_trial") == "physical_trial" and self.outcome_verifier is not None:
            if self._outcome_report is None:
                self._outcome_after=self._capture_outcome_rgb("after")
                try:
                    self._outcome_report=dict(self.outcome_verifier({
                        "instruction":self._instruction,
                        "before":self._outcome_before,"after":self._outcome_after}))
                    if not isinstance(self._outcome_report.get("verified"),bool):
                        raise LiberoDeploymentError("outcome verifier contract")
                except Exception as exc:
                    self._outcome_report={"verified":False,
                        "error":f"{type(exc).__name__}: {exc}","sensor_only":True}
                self.trace.append({"event":"independent_task_outcome_verify",
                                   "result":self._outcome_report})
            independent=bool(self._outcome_report.get("verified"))
        self._finalize_execution_artifacts()
        report = {"sensor_verification_passed":bool(self.last_verify and independent),
                "controller_visual_verification_passed":bool(self.last_verify),
                "independent_task_outcome":self._outcome_report,
                # Canonical task-level sensor evidence. Exposing immutable
                # paths/hashes lets the coding Agent diagnose verifier outages
                # and disagreements from exact before/after views instead of
                # guessing rollout frame numbers.
                "outcome_observations":{"before":self._outcome_before,
                    "after":self._outcome_after},
                "final_step":self.step,
                "final_proprioception":self.canonical_embodied_state()["robot"],
                "trace_path":self._execution_artifacts.get("trace_path"),
                "rollout_path":self._execution_artifacts.get("rollout_path"),
                "trajectory_path":self._execution_artifacts.get("trajectory_path"),
                "benchmark_signal_exposed":False,
                # Consumed only by the Harness generalization gate.  Keys
                # prefixed with _harness_ are removed from model evidence.
                "_harness_case_id":self.episode.case_handle}
        self._execution_sensor_report = report
        return dict(report)

    def agent_evidence(self, execution, sensor_report):
        """Project diagnostic sensor artifacts without evaluator/Harness truth."""
        outcome = sensor_report.get("independent_task_outcome")
        verifier_diagnostic = None
        if isinstance(outcome, Mapping) and outcome.get("error"):
            verifier_diagnostic = {"error": _sanitize_public_text(outcome["error"])}
        return {
            **_public_execution_diagnostics(execution),
            "sanitized_trace": _bounded_public(self.trace),
            "outcome_observations": sensor_report.get("outcome_observations"),
            "final_step": sensor_report.get("final_step"),
            "final_proprioception": sensor_report.get("final_proprioception"),
            "trace_path": sensor_report.get("trace_path"),
            "rollout_path": sensor_report.get("rollout_path"),
            "trajectory_path": sensor_report.get("trajectory_path"),
            "verifier_diagnostic": verifier_diagnostic,
        }

    def verification_receipt(self, execution):
        report=self.sensor_report(execution)
        return {"verified":bool(execution.get("completed") is True
                    and not execution.get("error")
                    and execution.get("sensor_verification_observed") is True
                    and report.get("sensor_verification_passed") is True),
                "controller_sha256":execution.get("program_sha256"),
                "environment_identity":self.execution_identity(),
                "episode_id":self.execution_identity()["episode_id"],
                "environment_generation":self.environment_generation}

    def seal_controller_execution(self):
        """Permanently close robot I/O before the evaluator barrier opens."""
        if self.closed:raise LiberoDeploymentError("deployment closed")
        self._controller_execution_sealed=True

    def _sealed_check_once(self):
        """Read LIBERO success once, after controller I/O is irrevocably sealed."""
        if self.closed:raise LiberoDeploymentError("deployment closed")
        if not self._controller_execution_sealed:
            raise LiberoDeploymentError("controller execution is not sealed")
        if self._evaluator_calls:
            raise LiberoDeploymentError("evaluator already consumed")
        self._evaluator_calls=1
        return bool(self.env.check_success())

    def hidden_evaluator(self, execution=None):
        return self._sealed_check_once()

    def close(self):
        if self.closed:return
        self.closed=True;(self.artifact_dir/"adapter_trace.json").write_text(json.dumps(self.trace,indent=2,default=str)+"\n")
        if self.video:
            h,w=self.video[0].shape[:2];writer=cv2.VideoWriter(str(self.artifact_dir/"rollout.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),20,(w,h))
            for frame in self.video:writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            writer.release()
        self.env.close()

__all__=["LiberoDeployment","LiberoEpisode","LiberoDeploymentError"]
