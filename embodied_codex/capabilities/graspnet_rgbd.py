"""Deployment-owned GraspNet RGB-D capability for Embodied Codex.

Only Adapter-emitted RGB-D/calibration and an open-vocabulary detector/SAM
region are consumed.  The wrapper independently reconstructs the target center
from the mask and metric depth; controller-authored object coordinates are
never trusted.  Grasp poses are converted from GraspNet's gripper convention
to the LIBERO Panda EEF convention before the Adapter issues opaque pose refs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
import uuid

import cv2
import numpy as np

from embodied_codex.capabilities.open_vocab_rgbd import CapabilityInputError


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def _topdown_eef_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """Retain GraspNet closing-axis yaw with a calibrated downward wrist."""
    rotation=np.asarray(grasp_rotation,float)
    x_axis=rotation[:,1].copy();x_axis[2]=0.0
    norm=float(np.linalg.norm(x_axis))
    if norm<1e-6:
        x_axis=rotation[:,2].copy();x_axis[2]=0.0;norm=float(np.linalg.norm(x_axis))
    if norm<1e-6:x_axis=np.array([1.0,0.0,0.0]);norm=1.0
    x_axis/=norm;z_axis=np.array([0.0,0.0,-1.0]);y_axis=np.cross(z_axis,x_axis)
    return np.column_stack([x_axis,y_axis,z_axis])


def _eef_rotation(grasp_rotation: np.ndarray) -> np.ndarray:
    """Map GraspNet (approach, closing, vertical) axes to Panda EEF axes."""
    rotation=np.asarray(grasp_rotation,float)
    if rotation.shape!=(3,3) or not np.isfinite(rotation).all():
        raise CapabilityInputError("GraspNet rotation is invalid")
    result=rotation[:,[1,2,0]]
    # Model output has small floating-point drift; project onto SO(3) without
    # altering the represented pose materially.
    left,_,right=np.linalg.svd(result);result=left@right
    if np.linalg.det(result)<0:left[:,-1]*=-1;result=left@right
    return result


class GraspNetRGBD:
    def __init__(self,*,backend_script: str|Path,checkpoint: str|Path,
                 python: str|Path=sys.executable,timeout_seconds: int=300):
        self.backend_script=Path(backend_script).resolve()
        self.checkpoint=Path(checkpoint).resolve();self.python=str(Path(python).resolve())
        self.timeout_seconds=int(timeout_seconds)
        for path in (self.backend_script,self.checkpoint):
            if not path.is_file():raise FileNotFoundError(path)
        self._provenance=None

    @property
    def provenance(self)->dict[str,Any]:
        if self._provenance is None:
            self._provenance={
                "tool_id":"graspnet_rgbd_6dof:v001",
                "models":["GraspNet baseline checkpoint-rs epoch 18"],
                "source_urls":["https://github.com/graspnet/graspnet-baseline"],
                "checkpoint_sha256":{"graspnet":_sha256(self.checkpoint)},
                "backend_sha256":_sha256(self.backend_script),
                "trained_on_current_task":False,"privileged_state_used":False,
                "inputs":["RGB","metric depth","camera intrinsic","camera-to-world",
                          "GroundingDINO box","SAM mask"],
            }
        return dict(self._provenance)

    @staticmethod
    def _sensor_inputs(payload: Mapping[str,Any]):
        frame=payload.get("frame");detection=payload.get("detection")
        if not isinstance(frame,Mapping) or not isinstance(detection,Mapping):
            raise CapabilityInputError("GraspNet requires frame and detection")
        camera_name=str(payload.get("camera") or "agentview")
        camera=(frame.get("cameras") or {}).get(camera_name)
        if not isinstance(camera,Mapping):raise CapabilityInputError("camera is absent")
        rgb_path=Path(str(camera.get("rgb_path") or "")).resolve()
        depth_path=Path(str(camera.get("depth_path") or "")).resolve()
        mask_path=Path(str(detection.get("mask_path") or "")).resolve()
        if not (rgb_path.is_file() and depth_path.is_file() and mask_path.is_file()):
            raise CapabilityInputError("RGB-D or SAM artifact is missing")
        # A detection may only reference a mask stored beneath this exact
        # Adapter frame.  This prevents arbitrary host files becoming inputs.
        if rgb_path.parent!=depth_path.parent or rgb_path.parent not in mask_path.parents:
            raise CapabilityInputError("detection mask does not belong to the supplied frame")
        rgb_bgr=cv2.imread(str(rgb_path),cv2.IMREAD_COLOR)
        mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
        if rgb_bgr is None or mask is None:raise CapabilityInputError("cannot read sensor image")
        rgb=cv2.cvtColor(rgb_bgr,cv2.COLOR_BGR2RGB);depth=np.load(depth_path,allow_pickle=False).squeeze()
        if mask.shape!=depth.shape or rgb.shape[:2]!=depth.shape:
            raise CapabilityInputError("RGB-D/mask shapes differ")
        intrinsic=np.asarray(camera.get("intrinsic"),float);extrinsic=np.asarray(camera.get("camera_to_world"),float)
        if intrinsic.shape!=(3,3) or extrinsic.shape!=(4,4):raise CapabilityInputError("invalid calibration")
        valid=(mask>0)&np.isfinite(depth)&(depth>.1)&(depth<3.0)
        rows,cols=np.nonzero(valid)
        if len(rows)<100:raise CapabilityInputError("insufficient target mask depth")
        z=depth[rows,cols]
        points=np.c_[(cols-intrinsic[0,2])*z/intrinsic[0,0],
                     (rows-intrinsic[1,2])*z/intrinsic[1,1],z,np.ones_like(z)]
        world=(extrinsic@points.T).T[:,:3]
        source=np.median(world,axis=0);source[2]=np.quantile(world[:,2],.75)
        bbox=np.asarray(detection.get("box_xyxy"),float)
        if bbox.shape!=(4,) or not np.isfinite(bbox).all():raise CapabilityInputError("invalid detector box")
        return rgb,depth,intrinsic,extrinsic,bbox,valid.astype(np.uint8),source,rgb_path.parent

    def infer(self,payload: Mapping[str,Any])->Mapping[str,Any]:
        rgb,depth,intrinsic,extrinsic,bbox,mask,source,frame_dir=self._sensor_inputs(payload)
        output_dir=frame_dir/"graspnet_rgbd";output_dir.mkdir(exist_ok=True)
        stem="inference_"+uuid.uuid4().hex[:10];input_path=output_dir/f"{stem}_input.npz"
        output_path=output_dir/f"{stem}_output.json"
        np.savez_compressed(input_path,rgb=rgb,depth=depth,intrinsic=intrinsic,
            camera_to_world=extrinsic,bbox_xyxy=bbox,object_mask=mask,
            source_xyz=source,mask_center_world=source)
        command=[self.python,str(self.backend_script),"--input",str(input_path),
                 "--output",str(output_path),"--checkpoint",str(self.checkpoint),
                 "--downward-min",str(float(payload.get("downward_min",.55))),
                 "--preferred-downward-min",str(float(payload.get("preferred_downward_min",.75)))]
        completed=subprocess.run(command,capture_output=True,text=True,timeout=self.timeout_seconds)
        if completed.returncode!=0 or not output_path.is_file():
            detail=(completed.stderr or completed.stdout)[-2000:]
            raise RuntimeError(f"GraspNet inference failed ({completed.returncode}): {detail}")
        audit=json.loads(output_path.read_text());strict=[];topdown=[]
        for item in audit.get("grasps") or []:
            candidate=dict(item);rotation=np.asarray(candidate["rotation_world"],float)
            candidate.update({"world_xyz":candidate["translation_world"],
                "approach_world":rotation[:,0].tolist(),
                "eef_rotation_world":_eef_rotation(rotation).tolist(),"pose_kind":"full_6dof"})
            strict.append(candidate)
        for item in audit.get("orientation_override_grasps") or []:
            candidate=dict(item);rotation=np.asarray(candidate["rotation_world"],float)
            candidate.update({"world_xyz":candidate["translation_world"],
                "approach_world":[0.0,0.0,-1.0],
                "eef_rotation_world":_topdown_eef_rotation(rotation).tolist(),
                "pose_kind":"calibrated_topdown"})
            topdown.append(candidate)
        return {"frame_id":payload.get("frame",{}).get("frame_id"),
                "target_center_world":source.tolist(),"full_6dof_grasps":strict,
                "calibrated_topdown_grasps":topdown,
                "filter_thresholds":audit.get("filter_thresholds"),
                "filter_diagnostics":audit.get("filter_diagnostics"),
                "artifact_path":str(output_path),"provenance":self.provenance}


__all__=["GraspNetRGBD"]
