"""Experimental high-resolution crops around automatic tactical ball guidance.

Crowding is an image-space proxy (nearby detected player feet), not an assertion
that the ball is occluded. Existing ball tracks are never replaced by this probe.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
from stage1_cv_baseline import ROOT,private_path,digest
from local_cv_possession import atomic

CONFIG=dict(radii_px=[48,96],imgsz=1280,confidence=.15,
            nearby_feet_radius_px=90.,minimum_players=2)


def player_context_crop(guide,boxes,width,height):
    """Cover the guide plus nearby full bodies, including room around hands.

    The guide is a ground-plane projection, so a tiny square around it is not
    sufficient for a held ball. This does not infer player identity/possession.
    """
    if guide is None:return None
    guide=np.asarray(guide,float)
    if not np.isfinite(guide).all() or not(0<=guide[0]<width and 0<=guide[1]<height):return None
    boxes=np.asarray(boxes,float).reshape(-1,4)
    feet=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3]]
    distances=np.linalg.norm(feet-guide,axis=1)
    nearby=[j for j in np.argsort(distances)[:2] if distances[j]<=CONFIG['nearby_feet_radius_px']]
    lo=guide-96;hi=guide+96
    for j in nearby:
        x,y,xx,yy=boxes[j];margin=max(12.,.2*(yy-y))
        lo=np.minimum(lo,[x-margin,y-margin]);hi=np.maximum(hi,[xx+margin,yy+margin])
    lo=np.maximum(0,np.floor(lo)).astype(int);hi=np.minimum([width,height],np.ceil(hi)).astype(int)
    return [int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]


def crowded_crops(guide,boxes,width,height,config=None):
    c=dict(CONFIG,**(config or {}))
    if guide is None:return [],0
    guide=np.asarray(guide,float)
    if not np.isfinite(guide).all() or not(0<=guide[0]<width and 0<=guide[1]<height):return [],0
    boxes=np.asarray(boxes,float).reshape(-1,4)
    feet=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3]]
    nearby=int((np.linalg.norm(feet-guide,axis=1)<=c['nearby_feet_radius_px']).sum())
    if nearby<c['minimum_players']:return [],nearby
    crops=[]
    for radius in c['radii_px']:
        lo=np.maximum(0,np.floor(guide-radius)).astype(int)
        hi=np.minimum([width,height],np.ceil(guide+radius)).astype(int)
        if min(hi-lo)>=32:crops.append([int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])])
    return crops,nearby


def detect(model,image,crops,device,config=None):
    from local_cv_tiling import merge_points
    c=dict(CONFIG,**(config or {}))
    if not crops:return np.empty((0,3))
    images=[image[y:yy,x:xx] for x,y,xx,yy in crops]
    points=[]
    # Sequential high-resolution passes avoid retaining a 1280px batch in GPU
    # memory on shared-memory laptops.
    for image,box in zip(images,crops):
        result=model.predict([image],classes=[32],conf=c['confidence'],iou=.7,
                             imgsz=c['imgsz'],device=device,half=device!='cpu',verbose=False)[0]
        xy=result.boxes.xyxy.cpu().numpy();score=result.boxes.conf.cpu().numpy()
        points.append(np.c_[(xy[:,:2]+xy[:,2:])/2+np.array(box[:2]),score])
    return merge_points(points)


def run(cache,out,device,features=None):
    import cv2,torch
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings');os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    cache,out=private_path(cache),private_path(out)
    rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    config=json.loads((cache/'config.json').read_text())
    feature_path=private_path(features or ROOT/'outputs/two-stage-cv-view/tactical-fused-v1.npz')
    if digest(feature_path)!=config['features_sha256']:raise ValueError('Feature source differs from cache')
    with np.load(feature_path,allow_pickle=False) as z:
        frames=z['frame_ids'];guides=z['atd_ball_xy'];boxes=z['player_boxes'];offset=z['player_offsets'];meta=json.loads(str(z['metadata']))
    if [r['frame'] for r in rows]!=frames.tolist():raise ValueError('Frame alignment differs')
    weights=ROOT/'outputs/two-stage-cv-models/yolo11x.pt'
    config.update(crowded_zoom=dict(CONFIG,code_sha256=digest(__file__),model_sha256=digest(weights),
        parent_sha256=digest(cache/'detections.jsonl'),parent_cache=str(cache),input_pixels='original decoded native 1920x1080 BGR',precision='FP16 on accelerator; FP32 CPU',
        track_replacement=False),new_detector_inference=True)
    out.mkdir(parents=True,exist_ok=True)
    old_config=out/'config.json'
    if old_config.exists():
        previous=json.loads(old_config.read_text())
        # Inference batching can change for memory use; numerical settings,
        # source and model must still match for a resumable run.
        for key in ['radii_px','imgsz','confidence','model_sha256','parent_sha256']:
            if previous['crowded_zoom'][key]!=config['crowded_zoom'][key]:raise ValueError('Resume configuration differs')
    atomic(old_config,config)
    torch.set_num_threads(2);cv2.setNumThreads(2);model=YOLO(weights)
    cached=[json.loads(s) for s in (out/'detections.jsonl').read_text().splitlines()] if (out/'detections.jsonl').exists() else []
    if [r['frame'] for r in cached]!=frames[:len(cached)].tolist():raise ValueError('Resume frame sequence differs')
    cap=cv2.VideoCapture(meta['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0])+len(cached))
    result=cached+copy.deepcopy(rows[len(cached):]);start=time.monotonic()
    tested=sum(bool(r['crowded_zoom_crops']) for r in cached);found=sum(bool(r['crowded_zoom_candidates']) for r in cached)
    try:
        with (out/'detections.jsonl').open('a') as handle:
            for i in range(len(cached),len(result)):
                row=result[i]
                ok,image=cap.read()
                if not ok:raise ValueError('Frame decode failed')
                crops,count=crowded_crops(guides[i],boxes[offset[i]:offset[i+1]],meta['width'],meta['height'])
                points=detect(model,image,crops,device)
                if crops:tested+=1
                if len(points):found+=1
                row.update(crowded_zoom_crops=crops,crowded_players=count,crowded_zoom_candidates=points.tolist(),
                    crowded_zoom_selected=points[np.argmin(np.linalg.norm(points[:,:2]-guides[i],axis=1))].tolist() if len(points) else None)
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
                if (i+1)%250==0:
                    atomic(out/'status.json',dict(state='running',completed=i+1,total=len(rows),crowded_frames=tested,candidate_frames=found))
                    print(f'{i+1}/{len(rows)} frames; {tested} crowded, {found} with candidates; {time.monotonic()-start:.1f}s',flush=True)
    finally:cap.release()
    assert all(a['context_position']==b['context_position'] for a,b in zip(rows,result))
    summary=dict(frames=len(rows),crowded_frames=tested,candidate_frames=found,elapsed_seconds=time.monotonic()-start,
        existing_positions_unchanged=True,accuracy_measured=False)
    atomic(out/'summary.json',summary);atomic(out/'status.json',dict(state='complete',completed=len(rows),total=len(rows)))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='mps')
    p.add_argument('--features',type=Path)
    a=p.parse_args();run(a.cache,a.out,a.device,a.features)
