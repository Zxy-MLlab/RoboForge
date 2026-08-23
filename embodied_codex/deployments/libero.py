"""Evaluator-blind LIBERO deployment for Embodied Codex.

The deployment contains robot I/O only. It never exposes reward, done, BDDL,
simulator identities, object state, or task-specific controller logic.
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

from ..sdk_contract import validate_action,validate_verifier_request


PROPRIO = ("robot0_joint_pos","robot0_joint_vel","robot0_eef_pos",
           "robot0_eef_quat","robot0_gripper_qpos","robot0_gripper_qvel")
CAMERAS = ("agentview","robot0_eye_in_hand")
Capability = Callable[[Mapping[str,Any]],Mapping[str,Any]]


class LiberoDeploymentError(RuntimeError): pass


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


class LiberoDeployment:
    def __init__(self, *, episode: LiberoEpisode, artifact_dir: str|Path,
                 capabilities: Mapping[str,Capability]|None=None,
                 verifiers: Mapping[str,Capability]|None=None,
                 outcome_verifier: Capability|None=None):
        if episode.config_path: os.environ["LIBERO_CONFIG_PATH"]=str(Path(episode.config_path).resolve())
        os.environ.setdefault("MUJOCO_GL","egl")
        from libero.libero import benchmark,get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.episode=episode;self.artifact_dir=Path(artifact_dir).resolve()
        self.artifact_dir.mkdir(parents=True,exist_ok=False)
        suite=benchmark.get_benchmark_dict()[episode.suite]();task=suite.get_task(episode.task_index)
        bddl=os.path.join(get_libero_path("bddl_files"),task.problem_folder,task.bddl_file)
        states=suite.get_task_init_states(episode.task_index)
        self.env=OffScreenRenderEnv(bddl_file_name=bddl,camera_names=list(CAMERAS),
            camera_heights=episode.image_size,camera_widths=episode.image_size,
            camera_depths=True,ignore_done=True,horizon=episode.horizon)
        self.env.seed(episode.seed);self.obs=self.env.reset()
        self.obs=self.env.set_init_state(states[episode.initial_state_index])
        self._instruction=str(task.language);self.capabilities=dict(capabilities or {})
        self.verifiers=dict(verifiers or {});self.references={};self.trace=[];self.video=[]
        self.step=0;self.frame=0;self.closed=False;self.last_verify=False
        self._controller_execution_sealed=False;self._evaluator_calls=0
        self.outcome_verifier=outcome_verifier;self._outcome_report=None
        # LIBERO init states can leave free objects several centimetres above
        # their support.  A generic no-motion settling period is part of the
        # deployment adapter, not learned task logic.  Reward/done/info remain
        # discarded exactly as during controller execution.
        warmup=int(np.clip(episode.warmup_steps,0,60))
        if warmup:
            for _ in range(warmup):self._sim_step(np.r_[np.zeros(6),-1.0])
            self.trace.append({"event":"adapter_warmup","steps":warmup,
                               "controller_visible":True})
        self._outcome_before=self._capture_outcome_rgb("before")
        (self.artifact_dir/"deployment.json").write_text(json.dumps({
            "protocol":"embodied-codex-libero-deployment-v1","suite":episode.suite,
            "task_index":episode.task_index,"state_index":episode.initial_state_index,
            "instruction":self._instruction,"controller_visible":["language","RGB-D",
            "calibration","proprioception","Tool output","action history"],
            "controller_hidden":["reward","done","evaluator","BDDL","object state","sim IDs"],
            "adapter_warmup_steps":warmup,
            "created_unix":time.time()},indent=2)+"\n")

    @property
    def instruction(self): return self._instruction

    def register_capability(self,tool_id,function):
        if tool_id in self.capabilities: raise LiberoDeploymentError("duplicate Tool")
        self.capabilities[str(tool_id)]=function

    def _proprio(self):
        return {key:np.asarray(self.obs[key]).tolist() for key in PROPRIO}

    def dispatch(self,method,arguments):
        if self.closed:raise LiberoDeploymentError("deployment closed")
        if self._controller_execution_sealed:
            raise LiberoDeploymentError("controller execution already sealed")
        if method=="observe":return self._observe(str(arguments.get("channel") or "rgbd"),arguments.get("request") or {})
        if method=="act":return self._act(arguments.get("action") or {})
        if method=="use":return self._use(str(arguments.get("tool_id") or ""),arguments.get("payload") or {})
        if method=="verify":return self._verify(str(arguments.get("verifier") or ""),arguments.get("payload") or {})
        if method=="record":
            self.trace.append({"event":"controller_record","payload":arguments.get("event")});return {"recorded":True}
        raise LiberoDeploymentError(f"unsupported method: {method}")

    def _observe(self,channel,request):
        if channel=="proprioception":return {"step":self.step,"proprioception":self._proprio()}
        if channel not in ("rgb","rgbd"):raise LiberoDeploymentError("unsupported sensor channel")
        from robosuite.utils.camera_utils import get_camera_extrinsic_matrix,get_camera_intrinsic_matrix,get_real_depth_map
        requested=request.get("cameras") or list(CAMERAS)
        if not isinstance(requested,list) or set(requested)-set(CAMERAS):raise LiberoDeploymentError("invalid cameras")
        self.frame+=1;frame_id=f"frame-{self.frame:06d}";folder=self.artifact_dir/"sensors"/frame_id
        folder.mkdir(parents=True,exist_ok=False);cameras={}
        for name in requested:
            rgb=np.ascontiguousarray(self.obs[f"{name}_image"][::-1]);rgb_path=folder/f"{name}_rgb.png"
            cv2.imwrite(str(rgb_path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            item={"rgb_path":str(rgb_path),"rgb_sha256":hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                  "shape":list(rgb.shape),"intrinsic":get_camera_intrinsic_matrix(self.env.sim,name,
                  self.episode.image_size,self.episode.image_size).tolist(),
                  "camera_to_world":get_camera_extrinsic_matrix(self.env.sim,name).tolist()}
            if channel=="rgbd":
                normalized=np.ascontiguousarray(self.obs[f"{name}_depth"][::-1])
                depth=np.asarray(get_real_depth_map(self.env.sim,normalized),np.float32)
                depth_path=folder/f"{name}_depth_m.npy";np.save(depth_path,depth)
                item.update({"depth_path":str(depth_path),"depth_sha256":hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                             "depth_range_m":[float(np.nanmin(depth)),float(np.nanmax(depth))]})
            cameras[name]=item
        report={"frame_id":frame_id,"step":self.step,"cameras":cameras,"proprioception":self._proprio()}
        (folder/"observation.json").write_text(json.dumps(report,indent=2)+"\n")
        self.trace.append({"event":"observe","frame_id":frame_id,"step":self.step});return report

    def _sim_step(self,action):
        if self.step>=self.episode.horizon:raise LiberoDeploymentError("action horizon exhausted")
        obs,_reward,_done,_info=self.env.step(np.clip(action,-1,1).tolist())
        self.obs=obs;self.step+=1
        if self.step%3==0:self.video.append(np.ascontiguousarray(self.obs["agentview_image"][::-1]))

    def _capture_outcome_rgb(self, name):
        folder=self.artifact_dir/"outcome";folder.mkdir(parents=True,exist_ok=True)
        rgb=np.ascontiguousarray(self.obs["agentview_image"][::-1])
        path=folder/f"{name}_agentview_rgb.png"
        cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        return {"rgb_path":str(path),
                "rgb_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape":list(rgb.shape)}

    def _act(self,action):
        kind=validate_action(action);before=np.asarray(self.obs["robot0_eef_pos"],float).copy();target=None
        if kind=="move_to_point":
            ref=str(action.get("target_ref") or "")
            if ref not in self.references:raise LiberoDeploymentError("unknown target_ref")
            target=np.asarray(self.references[ref]["world_xyz"],float)+np.asarray(action.get("offset") or [0,0,0],float)
            if target.shape!=(3,):raise LiberoDeploymentError("invalid target")
            tolerance=float(np.clip(action.get("tolerance_m",.015),.002,.06));gain=float(np.clip(action.get("gain",20),1,30))
            maximum=int(np.clip(action.get("max_steps",50),1,100));gripper=float(action.get("gripper",-1));reached=False
            for _ in range(maximum):
                error=target-np.asarray(self.obs["robot0_eef_pos"],float)
                if np.linalg.norm(error)<=tolerance:reached=True;break
                command=np.zeros(7);command[:3]=np.clip(error*gain,-1,1);command[6]=gripper;self._sim_step(command)
            reached=reached or np.linalg.norm(target-np.asarray(self.obs["robot0_eef_pos"],float))<=tolerance
        elif kind=="move_to_pose":
            # A pose may refer to a Tool-issued live pose, or combine a live
            # point reference with an explicitly supplied orientation.  Raw
            # world coordinates are deliberately not accepted here: every
            # translational target must retain sensor/Tool provenance.
            ref=str(action.get("pose_ref") or action.get("target_ref") or "")
            if ref not in self.references:raise LiberoDeploymentError("unknown pose_ref/target_ref")
            reference=self.references[ref]
            target=np.asarray(reference["world_xyz"],float)+np.asarray(action.get("offset") or [0,0,0],float)
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
                "gripper_qpos":np.asarray(self.obs["robot0_gripper_qpos"]).tolist()}
        if target is not None:result["target_xyz"]=target.tolist()
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
                token="point-"+uuid.uuid4().hex[:16];reference={"world_xyz":xyz,"tool_id":tool_id}
                bounds=result.get("world_bounds_10_90")
                if bounds is not None:
                    metric_bounds=np.asarray(bounds,float)
                    if metric_bounds.shape==(2,3) and np.isfinite(metric_bounds).all():
                        reference["world_bounds_10_90"]=metric_bounds.tolist()
                rotation=result.get("eef_rotation_world")
                if rotation is not None:
                    reference["eef_rotation_world"]=_validated_rotation_matrix(rotation).tolist()
                    result["pose_ref"]=token
                self.references[token]=reference
                result["point_ref"]=token
            return result
        if isinstance(value,list):return [self._references(tool_id,v) for v in value]
        return value

    def _use(self,tool_id,payload):
        if tool_id not in self.capabilities:raise LiberoDeploymentError(f"unregistered Tool: {tool_id}")
        try:raw_result=self.capabilities[tool_id](dict(payload))
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
        result=self._references(tool_id,raw_result)
        receipt={"tool_id":tool_id,"step":self.step,"result":result}
        self.trace.append({"event":"use","tool_id":tool_id,"step":self.step});return receipt

    def _verify(self,name,payload):
        validate_verifier_request(name,payload)
        expanded=dict(payload)
        for key in ("source_ref","target_ref"):
            if key in expanded:
                ref=str(expanded[key])
                if ref not in self.references:raise LiberoDeploymentError(f"unknown {key}")
                expanded[key.replace("_ref","_world_xyz")]=self.references[ref]["world_xyz"]
                bounds=self.references[ref].get("world_bounds_10_90")
                if bounds is not None:
                    expanded[key.replace("_ref","_world_bounds_10_90")]=bounds
        if name not in self.verifiers:raise LiberoDeploymentError(f"unknown verifier: {name}")
        try:result=dict(self.verifiers[name](expanded))
        except Exception as exc:
            result={"verified":False,"sensor_only":True,
                    "verifier_error":{"type":type(exc).__name__,
                                      "message":str(exc)[:1000]}}
            self.trace.append({"event":"verify","name":name,"result":result})
            return result
        if not isinstance(result.get("verified"),bool):raise LiberoDeploymentError("verifier contract")
        self.last_verify=result["verified"];self.trace.append({"event":"verify","name":name,"result":result});return result

    def sensor_report(self,execution):
        independent=True
        if self.outcome_verifier is not None:
            if self._outcome_report is None:
                after=self._capture_outcome_rgb("after")
                try:
                    self._outcome_report=dict(self.outcome_verifier({
                        "instruction":self._instruction,
                        "before":self._outcome_before,"after":after}))
                    if not isinstance(self._outcome_report.get("verified"),bool):
                        raise LiberoDeploymentError("outcome verifier contract")
                except Exception as exc:
                    self._outcome_report={"verified":False,
                        "error":f"{type(exc).__name__}: {exc}","sensor_only":True}
                self.trace.append({"event":"independent_task_outcome_verify",
                                   "result":self._outcome_report})
            independent=bool(self._outcome_report.get("verified"))
        return {"sensor_verification_passed":bool(self.last_verify and independent),
                "controller_visual_verification_passed":bool(self.last_verify),
                "independent_task_outcome":self._outcome_report,
                "final_step":self.step,
                "final_proprioception":self._proprio(),"trace_path":str(self.artifact_dir/"adapter_trace.json"),
                "rollout_path":str(self.artifact_dir/"rollout.mp4"),"benchmark_signal_exposed":False,
                # Consumed only by the Harness generalization gate.  Keys
                # prefixed with _harness_ are removed from model evidence.
                "_harness_case_id":f"state_{self.episode.initial_state_index:03d}"}

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
