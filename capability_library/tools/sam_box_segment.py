"""Generate an instance mask from RGB and a detector box using frozen SAM."""
import argparse, json, pathlib, sys
import cv2, numpy as np, torch
sys.path.insert(0,"/data/zxy/embodied_frontier/third_party/segment-anything")
from segment_anything import sam_model_registry, SamPredictor

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True)
    ap.add_argument("--visual",required=True);a=ap.parse_args()
    d=np.load(a.input); rgb=np.asarray(d["rgb"],np.uint8); box=np.asarray(d["bbox_xyxy"],np.float32)
    sam=sam_model_registry["vit_b"](checkpoint="/data/zxy/embodied_frontier/checkpoints/sam_vit_b_01ec64.pth").cuda().eval()
    predictor=SamPredictor(sam); predictor.set_image(rgb)
    masks,scores,_=predictor.predict(box=box[None,:],multimask_output=True)
    # Prefer the highest-IoU mask that stays mostly within the detector box.
    x0,y0,x1,y1=box.astype(int); ranked=[]
    for i,(m,s) in enumerate(zip(masks,scores)):
        inside=m[max(0,y0):y1,max(0,x0):x1].sum(); containment=float(inside/max(m.sum(),1))
        ranked.append((float(s)+.2*containment,i,containment))
    _,idx,containment=max(ranked); mask=masks[idx]
    depth=np.asarray(d["depth"]).squeeze(); K=np.asarray(d["intrinsic"]); T=np.asarray(d["camera_to_world"])
    yy,xx=np.nonzero(mask & np.isfinite(depth) & (depth>.1) & (depth<2.0)); z=depth[yy,xx]
    cam=np.c_[(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z,np.ones_like(z)]
    world=(T@cam.T).T[:,:3]; lo,hi=np.quantile(world,[.1,.9],axis=0)
    mask_center=np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,np.quantile(world[:,2],.75)])
    payload={k:d[k] for k in d.files}; payload["object_mask"]=mask.astype(np.uint8);payload["mask_center_world"]=mask_center
    np.savez_compressed(a.output,**payload)
    vis=rgb.copy(); overlay=np.zeros_like(vis);overlay[...,1]=255;vis[mask]=(0.55*vis[mask]+0.45*overlay[mask]).astype(np.uint8)
    cv2.rectangle(vis,(x0,y0),(x1,y1),(255,80,80),2);cv2.imwrite(a.visual,cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
    print(json.dumps({"sam_score":float(scores[idx]),"containment":containment,"mask_pixels":int(mask.sum()),"mask_center_world":mask_center.tolist(),"privileged_inputs_used":[]}))
if __name__=="__main__":main()
