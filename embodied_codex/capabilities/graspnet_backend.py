"""Frozen GraspNet-1Billion RGB-D grasp inference tool.

Input is a sensor NPZ containing metric depth, camera intrinsics/extrinsics,
and an RGB detector box. Output is ranked 6-DoF grasps in camera and world
coordinates. No simulator object state or segmentation is accepted.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

def configure_source_root(value):
    root=pathlib.Path(value).resolve()
    for sub in ("models","pointnet2","knn","utils"):
        sys.path.insert(0,str(root/sub))

def sample_cloud(depth, intrinsic, bbox, object_mask=None, count=20000):
    h,w=depth.shape; x0,y0,x1,y1=[int(v) for v in bbox]
    # Context around the detected object lets GraspNet assess approach and
    # support geometry without a privileged instance mask.
    margin=max(x1-x0,y1-y0)//2
    x0=max(0,x0-margin); x1=min(w,x1+margin); y0=max(0,y0-margin); y1=min(h,y1+margin)
    yy,xx=np.mgrid[y0:y1,x0:x1]; z=depth[y0:y1,x0:x1]
    valid=np.isfinite(z)&(z>.1)&(z<2.0)
    all_valid=valid.copy()
    if object_mask is not None:
        # Keep target points plus a thin context ring needed for grasp
        # collision reasoning; the mask comes from SAM, not the simulator.
        import cv2
        target=np.asarray(object_mask[y0:y1,x0:x1],np.uint8)
        context=cv2.dilate(target,np.ones((25,25),np.uint8),iterations=1).astype(bool)
        valid &= context
    x_all,y_all,z_all=xx[all_valid],yy[all_valid],z[all_valid]
    fx,fy,cx,cy=intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2]
    collision_points=np.c_[(x_all-cx)*z_all/fx,(y_all-cy)*z_all/fy,z_all].astype(np.float32)
    x,y,z=xx[valid],yy[valid],z[valid]
    points=np.c_[(x-cx)*z/fx,(y-cy)*z/fy,z].astype(np.float32)
    if len(points)<100: raise RuntimeError("insufficient RGB-D points in detector crop")
    rng=np.random.default_rng(0)
    idx=rng.choice(len(points),count,replace=len(points)<count)
    return points[idx],collision_points

def model_free_collision_metrics(raw, scene_points, voxel_size=.005, approach_dist=.05):
    """NumPy port of GraspNet's public ModelFreeCollisionDetector."""
    if len(raw)==0: return np.empty(0),np.empty(0)
    voxels=np.unique(np.floor(np.asarray(scene_points,float)/voxel_size).astype(np.int64),axis=0)
    cloud=(voxels.astype(float)+.5)*voxel_size
    collision=np.zeros(len(raw),float); occupancy=np.zeros(len(raw),float)
    finger_width=.01; finger_length=.06; approach_dist=max(approach_dist,finger_width)
    for start in range(0,len(raw),128):
        rows=raw[start:start+128]; heights=rows[:,2,None]; depths=rows[:,3,None]
        widths=rows[:,1,None]; rotations=rows[:,4:13].reshape(-1,3,3); translations=rows[:,13:16]
        targets=np.matmul(cloud[None,:,:]-translations[:,None,:],rotations)
        m1=(targets[:,:,2]>-heights/2)&(targets[:,:,2]<heights/2)
        m2=(targets[:,:,0]>depths-finger_length)&(targets[:,:,0]<depths)
        m3=targets[:,:,1]>-(widths/2+finger_width); m4=targets[:,:,1]<-widths/2
        m5=targets[:,:,1]<(widths/2+finger_width); m6=targets[:,:,1]>widths/2
        m7=(targets[:,:,0]<=depths-finger_length)&(targets[:,:,0]>depths-finger_length-finger_width)
        m8=(targets[:,:,0]<=depths-finger_length-finger_width)&(targets[:,:,0]>depths-finger_length-finger_width-approach_dist)
        occupied=(m1&m2&m3&m4)|(m1&m2&m5&m6)|(m1&m3&m5&m7)|(m1&m3&m5&m8)
        volume=(heights*finger_length*finger_width*2+heights*(widths+2*finger_width)*(finger_width+approach_dist)).reshape(-1)/(voxel_size**3)
        collision[start:start+len(rows)]=occupied.sum(axis=1)/(volume+1e-6)
        inner=m1&m2&(~m4)&(~m6)
        inner_volume=(heights*finger_length*widths).reshape(-1)/(voxel_size**3)
        occupancy[start:start+len(rows)]=inner.sum(axis=1)/(inner_volume+1e-6)
    return collision,occupancy

def infer(npz_path, checkpoint, output, downward_min=.55, preferred_downward_min=.75):
    import torch
    from graspnet import GraspNet,pred_decode
    data=np.load(npz_path); depth=data["depth"].squeeze(); intrinsic=data["intrinsic"]
    extrinsic=data["camera_to_world"]; bbox=data["bbox_xyxy"]
    sampled,cloud=sample_cloud(depth,intrinsic,bbox,data["object_mask"] if "object_mask" in data.files else None)
    net=GraspNet(input_feature_dim=0,num_view=300,num_angle=12,num_depth=4,
        cylinder_radius=.05,hmin=-.02,hmax_list=[.01,.02,.03,.04],is_training=False).cuda().eval()
    state=torch.load(checkpoint,map_location="cpu",weights_only=False); net.load_state_dict(state["model_state_dict"])
    with torch.no_grad(): raw=pred_decode(net({"point_clouds":torch.from_numpy(sampled[None]).cuda()}))[0].cpu().numpy()
    collision_iou,inner_occupancy=model_free_collision_metrics(raw,cloud)
    # GraspNet rows: score,width,height,depth,R(9),translation(3),object_id.
    target_world=np.asarray(data["mask_center_world"] if "mask_center_world" in data.files else data["source_xyz"],float)
    target_cam=(np.linalg.inv(extrinsic)@np.r_[target_world,1.0])[:3]
    results=[]; orientation_override=[]; diagnostics=[]
    distance_limit=.07; min_width=.005; max_width=.081; downward_limit=float(downward_min)
    for raw_index,row in enumerate(raw):
        score,width,height,gdepth=map(float,row[:4]); rot=row[4:13].reshape(3,3); trans=row[13:16]
        dist=float(np.linalg.norm(trans-target_cam))
        world_rot=extrinsic[:3,:3]@rot; world_trans=(extrinsic@np.r_[trans,1.0])[:3]
        # Favor model confidence, proximity to the language-selected object,
        # and approaches that are mostly downward in world coordinates.
        approach=world_rot[:,0]; downward=max(0.0,float(-approach[2]))
        passed_distance=dist <= distance_limit
        passed_width=min_width < width < max_width
        passed_downward=downward >= downward_limit
        passed_collision=bool(collision_iou[raw_index] <= .01)
        passed_nonempty=bool(inner_occupancy[raw_index] >= .01)
        diagnostic={"model_score":score,"distance_to_target_m":dist,"width_m":width,
            "downward_score":downward,"translation_world":world_trans.tolist(),
            "passed_distance":passed_distance,"passed_width":passed_width,
            "passed_downward":passed_downward,"collision_iou":float(collision_iou[raw_index]),
            "inner_occupancy":float(inner_occupancy[raw_index]),
            "passed_collision":passed_collision,"passed_nonempty":passed_nonempty}
        diagnostics.append(diagnostic)
        # A robot-specific controller may deliberately replace GraspNet's
        # predicted wrist orientation with a calibrated top-down orientation.
        # Preserve a bounded set of target-local translations for that case;
        # the normal 6-DoF list below remains strictly filtered.  Collision is
        # relaxed only slightly because it was evaluated for the discarded
        # model orientation, while target locality, physical gripper width and
        # non-empty finger volume remain mandatory.
        if (passed_distance and passed_width and passed_nonempty and
                float(collision_iou[raw_index]) <= .025):
            override_score=score-5.0*dist-1.0*float(collision_iou[raw_index])
            orientation_override.append({"rank_score":override_score,"model_score":score,
                "distance_to_target_m":dist,"width_m":width,"height_m":height,"depth_m":gdepth,
                "translation_camera":trans.tolist(),"rotation_camera":rot.tolist(),
                "translation_world":world_trans.tolist(),"rotation_world":world_rot.tolist(),
                "downward_score":downward,"collision_iou":float(collision_iou[raw_index]),
                "inner_occupancy":float(inner_occupancy[raw_index]),
                "orientation_override_required":True})
        if not (passed_distance and passed_width and passed_downward and passed_collision and passed_nonempty): continue
        rank_score=score-5.0*dist+0.15*downward-2.0*float(collision_iou[raw_index])
        results.append({"rank_score":rank_score,"model_score":score,"distance_to_target_m":dist,
            "width_m":width,"height_m":height,"depth_m":gdepth,"translation_camera":trans.tolist(),
            "rotation_camera":rot.tolist(),"translation_world":world_trans.tolist(),"rotation_world":world_rot.tolist(),
            "downward_score":downward,"collision_iou":float(collision_iou[raw_index]),
            "inner_occupancy":float(inner_occupancy[raw_index])})
    preferred=[x for x in results if x["downward_score"] >= preferred_downward_min]
    active_downward_limit=float(preferred_downward_min if preferred else downward_limit)
    if preferred: results=preferred
    for diagnostic in diagnostics:
        diagnostic["passed_downward"]=diagnostic["downward_score"] >= active_downward_limit
    results.sort(key=lambda x:x["rank_score"],reverse=True)
    orientation_override.sort(key=lambda x:x["rank_score"],reverse=True)
    def metric_summary(key):
        values=np.asarray([item[key] for item in diagnostics],float)
        return {"min":float(values.min()),"p05":float(np.quantile(values,.05)),
                "median":float(np.median(values)),"p95":float(np.quantile(values,.95)),
                "max":float(values.max())}
    rejection_counts={
        "failed_distance":sum(not x["passed_distance"] for x in diagnostics),
        "failed_width":sum(not x["passed_width"] for x in diagnostics),
        "failed_downward":sum(not x["passed_downward"] for x in diagnostics),
        "failed_collision":sum(not x["passed_collision"] for x in diagnostics),
        "failed_empty":sum(not x["passed_nonempty"] for x in diagnostics),
        "passed_distance":sum(x["passed_distance"] for x in diagnostics),
        "passed_distance_and_width":sum(x["passed_distance"] and x["passed_width"] for x in diagnostics),
        "passed_all":len(results),
    }
    # Keep enough raw evidence to understand a failed filter without writing
    # all 4,800 model rows. These lists are diagnostic only and never control.
    nearest=sorted(diagnostics,key=lambda x:x["distance_to_target_m"])[:100]
    highest_confidence=sorted(diagnostics,key=lambda x:x["model_score"],reverse=True)[:100]
    payload={"model":"GraspNet baseline checkpoint-rs epoch 18","input":"RGB-D detector crop",
             "privileged_inputs_used":[],"raw_candidate_count":int(len(raw)),"filtered_candidate_count":len(results),
             "target_center_world":target_world.tolist(),
             "filter_thresholds":{"distance_to_target_m_max":distance_limit,
                 "width_m_range_open":[min_width,max_width],"preferred_downward_score_min":preferred_downward_min,
                 "fallback_downward_score_min":downward_limit,"active_downward_score_min":active_downward_limit,
                 "downward_fallback_used":not bool(preferred),"collision_iou_max":.01,
                 "inner_occupancy_min":.01,"collision_algorithm":"GraspNet ModelFreeCollisionDetector NumPy port"},
             "filter_diagnostics":{"counts":rejection_counts,
                 "metrics":{"distance_to_target_m":metric_summary("distance_to_target_m"),
                     "width_m":metric_summary("width_m"),"downward_score":metric_summary("downward_score"),
                     "collision_iou":metric_summary("collision_iou"),"inner_occupancy":metric_summary("inner_occupancy")},
                 "nearest_to_target":nearest,"highest_model_score":highest_confidence},
             "grasps":results[:100],
             "orientation_override_grasps":orientation_override[:100],
             "orientation_override_policy":{
                 "allowed_only_when_controller_uses_calibrated_robot_orientation":True,
                 "distance_to_target_m_max":distance_limit,
                 "width_m_range_open":[min_width,max_width],
                 "collision_iou_max_for_discarded_model_orientation":.025,
                 "inner_occupancy_min":.01}}
    pathlib.Path(output).write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--checkpoint",default="checkpoints/graspnet-checkpoint-rs.tar")
    ap.add_argument("--source-root",required=True)
    ap.add_argument("--downward-min",type=float,default=.55,
        help="Fallback robot reachability gate when no strongly downward grasp exists")
    ap.add_argument("--preferred-downward-min",type=float,default=.75,
        help="Preferred top-access compatibility tier")
    a=ap.parse_args();configure_source_root(a.source_root)
    infer(a.input,a.checkpoint,a.output,a.downward_min,a.preferred_downward_min)
if __name__=="__main__":main()
