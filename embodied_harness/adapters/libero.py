"""Evaluator-blind LIBERO deployment plugin.

This module owns environment lifecycle and generic robot I/O only.  It contains
no task controller, task/state branch, object identity lookup, or benchmark
success logic.  Controller-visible data is limited to language, RGB-D,
calibration, proprioception, registered capability results, and action history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import uuid

import cv2
import numpy as np


PROPRIO_KEYS = (
    "robot0_joint_pos", "robot0_joint_vel", "robot0_eef_pos",
    "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel",
)
CAMERAS = ("agentview", "robot0_eye_in_hand")


class LiberoAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiberoEpisode:
    suite: str
    task_index: int
    initial_state_index: int
    seed: int = 7
    image_size: int = 256
    horizon: int = 1200
    config_path: str | None = None


Capability = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class LiberoAdapter:
    """One persistent LIBERO episode behind the standard RobotAdapter RPC."""

    def __init__(
        self, *, episode: LiberoEpisode, artifact_dir: str | Path,
        capabilities: Mapping[str, Capability] | None = None,
        verifiers: Mapping[str, Capability] | None = None,
        verifier_tool_ids: Mapping[str, str] | None = None,
    ) -> None:
        if episode.config_path:
            os.environ["LIBERO_CONFIG_PATH"] = str(Path(episode.config_path).resolve())
        os.environ.setdefault("MUJOCO_GL", "egl")
        # Imports are intentionally local: the standalone core has no LIBERO
        # dependency and can be installed for a real robot deployment.
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.episode = episode
        self.artifact_dir = Path(artifact_dir).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        suites = benchmark.get_benchmark_dict()
        if episode.suite not in suites:
            raise LiberoAdapterError(f"unknown suite: {episode.suite}")
        suite = suites[episode.suite]()
        if not 0 <= episode.task_index < suite.n_tasks:
            raise LiberoAdapterError("task index is outside deployment suite")
        task = suite.get_task(episode.task_index)
        bddl_path = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file,
        )
        init_states = suite.get_task_init_states(episode.task_index)
        if not 0 <= episode.initial_state_index < len(init_states):
            raise LiberoAdapterError("initial state index is outside deployment task")
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_path, camera_names=list(CAMERAS),
            camera_heights=episode.image_size, camera_widths=episode.image_size,
            camera_depths=True, ignore_done=True, horizon=episode.horizon,
        )
        self.env.seed(episode.seed)
        self.obs = self.env.reset()
        self.obs = self.env.set_init_state(init_states[episode.initial_state_index])
        self.language = str(task.language)
        self.capabilities = dict(capabilities or {})
        self.verifiers = dict(verifiers or {})
        self.verifier_tool_ids = dict(verifier_tool_ids or {})
        self.step_count = 0
        self.frame_count = 0
        self.closed = False
        self.trace: list[dict[str, Any]] = []
        self.video_frames: list[np.ndarray] = []
        self.references: dict[str, dict[str, Any]] = {}
        self.last_sensor_verification = False
        self._sealed_evaluation_used = False
        self._write_deployment_manifest()

    def register_capability(self, tool_id: str, function: Capability) -> None:
        """Install a tested Harness Tool before this episode starts executing."""
        if not tool_id or not callable(function):
            raise LiberoAdapterError("invalid runtime capability registration")
        if tool_id in self.capabilities:
            raise LiberoAdapterError(f"capability Tool already registered: {tool_id}")
        self.capabilities[tool_id] = function

    @property
    def initial_context(self) -> Mapping[str, Any]:
        return {
            "task_instruction": self.language,
            "deployment_contract": {
                "sensors": ["rgbd", "rgb", "proprioception", "catalog"],
                "actions": ["osc_delta", "move_to_point", "gripper", "settle"],
                "capability_tools": sorted(self.capabilities),
                "verifiers": sorted(self.verifiers) + ["eef_near_point"],
                "forbidden": [
                    "reward", "done", "benchmark evaluator", "BDDL controller input",
                    "MuJoCo object poses or IDs", "task/state branches",
                ],
            },
        }

    def _write_deployment_manifest(self) -> None:
        # Task/state indices are deployment metadata. They are stored outside
        # model-visible execution evidence and never returned by dispatch.
        manifest = {
            "protocol": "standalone-libero-adapter-v1",
            "suite": self.episode.suite, "task_index": self.episode.task_index,
            "initial_state_index": self.episode.initial_state_index,
            "seed": self.episode.seed, "instruction": self.language,
            "controller_visible": [
                "language", "RGB-D", "camera calibration", "proprioception",
                "registered capability output", "action history",
            ],
            "controller_hidden": [
                "reward", "done", "check_success", "BDDL", "object state",
                "simulator segmentation and object identity",
            ],
            "created_unix": time.time(),
        }
        (self.artifact_dir / "deployment.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    def dispatch(self, method: str, arguments: Mapping[str, Any]) -> Any:
        if self.closed: raise LiberoAdapterError("episode is closed")
        if method == "instruction": return self.language
        if method == "sense":
            return self._sense(str(arguments.get("channel") or "rgbd"),
                               arguments.get("request") or {})
        if method == "act": return self._act(arguments.get("action") or {})
        if method == "use":
            return self._use(str(arguments.get("tool_id") or ""),
                             arguments.get("payload") or {})
        if method == "verify":
            return self._verify(str(arguments.get("verifier") or ""),
                                arguments.get("payload") or {})
        if method == "record":
            row = {"event": "controller_record", "payload": arguments.get("event")}
            self.trace.append(row); return {"recorded": True}
        raise LiberoAdapterError(f"unsupported RPC method: {method}")

    def _proprio(self) -> dict[str, Any]:
        missing = [key for key in PROPRIO_KEYS if key not in self.obs]
        if missing: raise LiberoAdapterError(f"missing proprioception: {missing}")
        return {key: np.asarray(self.obs[key]).tolist() for key in PROPRIO_KEYS}

    def _sense(self, channel: str, request: Mapping[str, Any]) -> dict[str, Any]:
        if channel == "catalog": return dict(self.initial_context["deployment_contract"])
        if channel == "proprioception":
            return {"step": self.step_count, "proprioception": self._proprio()}
        if channel not in {"rgb", "rgbd"}:
            raise LiberoAdapterError(f"unsupported sensor channel: {channel}")
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
            get_real_depth_map,
        )
        requested = request.get("cameras") or list(CAMERAS)
        if not isinstance(requested, list) or not requested:
            raise LiberoAdapterError("request.cameras must be a nonempty list")
        if set(requested) - set(CAMERAS):
            raise LiberoAdapterError("unknown or privileged camera requested")
        self.frame_count += 1
        frame_id = f"frame-{self.frame_count:06d}"
        frame_dir = self.artifact_dir / "sensors" / frame_id
        frame_dir.mkdir(parents=True)
        cameras: dict[str, Any] = {}
        for camera in requested:
            rgb = np.ascontiguousarray(self.obs[f"{camera}_image"][::-1])
            rgb_path = frame_dir / f"{camera}_rgb.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            camera_record: dict[str, Any] = {
                "rgb_path": str(rgb_path), "rgb_sha256": hashlib.sha256(
                    rgb_path.read_bytes()).hexdigest(), "shape": list(rgb.shape),
                "intrinsic": get_camera_intrinsic_matrix(
                    self.env.sim, camera, self.episode.image_size,
                    self.episode.image_size).tolist(),
                "camera_to_world": get_camera_extrinsic_matrix(
                    self.env.sim, camera).tolist(),
            }
            if channel == "rgbd":
                normalized = np.ascontiguousarray(self.obs[f"{camera}_depth"][::-1])
                depth = np.asarray(get_real_depth_map(self.env.sim, normalized), np.float32)
                depth_path = frame_dir / f"{camera}_depth_m.npy"
                np.save(depth_path, depth)
                camera_record.update({
                    "depth_path": str(depth_path),
                    "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                    "depth_range_m": [float(np.nanmin(depth)), float(np.nanmax(depth))],
                })
            cameras[camera] = camera_record
        report = {
            "frame_id": frame_id, "step": self.step_count,
            "cameras": cameras, "proprioception": self._proprio(),
        }
        (frame_dir / "sensor_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        self.trace.append({"event": "sense", "frame_id": frame_id,
                           "channel": channel, "step": self.step_count})
        return report

    def _step(self, action: np.ndarray) -> None:
        if self.step_count >= self.episode.horizon:
            raise LiberoAdapterError("deployment action horizon exhausted")
        if action.shape != (7,) or not np.isfinite(action).all():
            raise LiberoAdapterError("OSC action must be seven finite numbers")
        # LIBERO computes benchmark signals internally as part of env.step.
        # They are immediately discarded and never logged or returned.
        next_obs, _sealed_reward, _sealed_done, _sealed_info = self.env.step(
            np.clip(action, -1.0, 1.0).tolist()
        )
        self.obs = next_obs
        self.step_count += 1
        if self.step_count % 3 == 0:
            self.video_frames.append(np.ascontiguousarray(
                self.obs["agentview_image"][::-1]
            ))

    def _act(self, action: Mapping[str, Any]) -> dict[str, Any]:
        kind = str(action.get("type") or "")
        before = np.asarray(self.obs["robot0_eef_pos"], dtype=float).copy()
        if kind == "osc_delta":
            translation = np.asarray(action.get("translation") or [0, 0, 0], float)
            rotation = np.asarray(action.get("rotation") or [0, 0, 0], float)
            if translation.shape != (3,) or rotation.shape != (3,):
                raise LiberoAdapterError("osc_delta requires translation/rotation xyz")
            gripper = float(action.get("gripper", -1.0))
            repeats = int(np.clip(action.get("repeat", 1), 1, 20))
            command = np.r_[translation, rotation, gripper]
            for _ in range(repeats): self._step(command)
            reached = True
            target = None
        elif kind == "move_to_point":
            reference = str(action.get("target_ref") or "")
            if reference not in self.references:
                raise LiberoAdapterError("move_to_point requires a capability-issued target_ref")
            target = np.asarray(self.references[reference]["world_xyz"], float)
            offset = np.asarray(action.get("offset") or [0, 0, 0], float)
            if offset.shape != (3,) or np.linalg.norm(offset) > 0.35:
                raise LiberoAdapterError("invalid bounded target offset")
            target = target + offset
            tolerance = float(np.clip(action.get("tolerance_m", 0.015), 0.002, 0.06))
            gain = float(np.clip(action.get("gain", 20.0), 1.0, 30.0))
            maximum = int(np.clip(action.get("max_steps", 40), 1, 100))
            gripper = float(action.get("gripper", -1.0))
            reached = False
            for _ in range(maximum):
                error = target - np.asarray(self.obs["robot0_eef_pos"], float)
                if np.linalg.norm(error) <= tolerance:
                    reached = True; break
                command = np.zeros(7, float)
                command[:3] = np.clip(error * gain, -1.0, 1.0)
                command[6] = gripper
                self._step(command)
            if np.linalg.norm(target - np.asarray(self.obs["robot0_eef_pos"], float)) <= tolerance:
                reached = True
        elif kind == "gripper":
            command_name = str(action.get("command") or "")
            if command_name not in {"open", "close"}:
                raise LiberoAdapterError("gripper command must be open or close")
            repeats = int(np.clip(action.get("repeat", 12), 1, 40))
            command = np.zeros(7, float); command[6] = -1.0 if command_name == "open" else 1.0
            for _ in range(repeats): self._step(command)
            reached = True; target = None
        elif kind == "settle":
            repeats = int(np.clip(action.get("steps", 10), 1, 60))
            command = np.zeros(7, float); command[6] = float(action.get("gripper", -1.0))
            for _ in range(repeats): self._step(command)
            reached = True; target = None
        else:
            raise LiberoAdapterError(f"unsupported action type: {kind}")
        after = np.asarray(self.obs["robot0_eef_pos"], dtype=float)
        result = {
            "action_type": kind, "step": self.step_count, "reached": bool(reached),
            "eef_before": before.tolist(), "eef_after": after.tolist(),
            "gripper_qpos": np.asarray(self.obs["robot0_gripper_qpos"]).tolist(),
        }
        if target is not None: result["target_xyz"] = target.tolist()
        self.trace.append({"event": "act", "request": dict(action), "result": result})
        return result

    def _register_references(self, tool_id: str, result: Any) -> Any:
        if isinstance(result, Mapping):
            normalized = {str(key): self._register_references(tool_id, value)
                          for key, value in result.items()}
            xyz = normalized.get("world_xyz")
            if isinstance(xyz, list) and len(xyz) == 3:
                array = np.asarray(xyz, float)
                if np.isfinite(array).all():
                    token = "point-" + uuid.uuid4().hex[:16]
                    self.references[token] = {
                        "world_xyz": array.tolist(), "tool_id": tool_id,
                        "created_step": self.step_count,
                    }
                    normalized["point_ref"] = token
            return normalized
        if isinstance(result, list):
            return [self._register_references(tool_id, value) for value in result]
        return result

    def _use(self, tool_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if tool_id not in self.capabilities:
            raise LiberoAdapterError(f"unregistered capability Tool: {tool_id}")
        result = self.capabilities[tool_id](dict(payload))
        normalized = self._register_references(tool_id, result)
        receipt = {"tool_id": tool_id, "step": self.step_count,
                   "result": normalized}
        self.trace.append({"event": "use", **receipt})
        return receipt

    def _verify(self, verifier: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        expanded_payload = dict(payload)
        for field in ("source_ref", "target_ref"):
            if field in expanded_payload:
                reference = str(expanded_payload[field])
                if reference not in self.references:
                    raise LiberoAdapterError(f"{field} is not a capability-issued reference")
                expanded_payload[field.replace("_ref", "_world_xyz")] = list(
                    self.references[reference]["world_xyz"]
                )
        if verifier == "eef_near_point":
            reference = str(expanded_payload.get("target_ref") or "")
            if reference not in self.references:
                raise LiberoAdapterError("eef_near_point requires target_ref")
            target = np.asarray(self.references[reference]["world_xyz"], float)
            offset = np.asarray(expanded_payload.get("offset") or [0, 0, 0], float)
            distance = float(np.linalg.norm(
                np.asarray(self.obs["robot0_eef_pos"], float) - target - offset
            ))
            threshold = float(np.clip(expanded_payload.get("threshold_m", 0.03), 0.002, 0.1))
            result = {"verified": distance <= threshold, "distance_m": distance,
                      "threshold_m": threshold}
        elif verifier in self.verifiers:
            result = dict(self.verifiers[verifier](expanded_payload))
            if not isinstance(result.get("verified"), bool):
                raise LiberoAdapterError("verifier must return boolean verified")
            if verifier in self.verifier_tool_ids:
                result["capability_tool_id"] = self.verifier_tool_ids[verifier]
        else:
            raise LiberoAdapterError(f"unregistered sensor verifier: {verifier}")
        self.last_sensor_verification = result.get("verified") is True
        self.trace.append({"event": "verify", "verifier": verifier,
                           "payload": expanded_payload, "result": result})
        return result

    def sensor_report(self, execution: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "sensor_verification_passed": bool(
                execution.get("graph_outcome") == "success"
                and self.last_sensor_verification
            ),
            "final_step": self.step_count,
            "final_proprioception": self._proprio(),
            "trace_path": str(self.artifact_dir / "adapter_trace.json"),
            "rollout_path": str(self.artifact_dir / "rollout.mp4"),
            "benchmark_signal_exposed": False,
        }

    def _sealed_check_once(self) -> bool:
        """External evaluation runner only; unreachable through dispatch."""
        if self._sealed_evaluation_used:
            raise LiberoAdapterError("sealed benchmark check already consumed")
        self._sealed_evaluation_used = True
        return bool(self.env.check_success())

    def close(self) -> None:
        if self.closed: return
        (self.artifact_dir / "adapter_trace.json").write_text(
            json.dumps(self.trace, indent=2, default=str) + "\n"
        )
        if self.video_frames:
            height, width = self.video_frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(self.artifact_dir / "rollout.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height),
            )
            for frame in self.video_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
        self.env.close(); self.closed = True


__all__ = ["LiberoAdapter", "LiberoAdapterError", "LiberoEpisode"]
