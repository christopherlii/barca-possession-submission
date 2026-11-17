"""Recover short bracketed gaps with native RGB crops and bounded interpolation.

Endpoint measurements are immutable. Search starts with 128px crops, then 64px
crops if necessary. Only an unambiguous candidate close to BOTH endpoint-based
motion constraints is an observation; otherwise output remains an estimate.
No human ball annotations, public targets or possession labels are used.
"""
from cached_video_reader import CachedVideoCapture
import argparse
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
from stage1_cv_baseline import ROOT,private_path,digest
from aerial_gap_recovery import gap_poses,apply

CONFIG=dict(max_gap_frames=5,max_speed_px_frame=60.,radii=[64,32],
            min_confidence=.12,base_tolerance_px=12.,acceleration_allowance=3.,
            ambiguity_distance_px=6.,ambiguity_score_margin=.15)


def plans(rows,config=None):
    c=dict(CONFIG,**(config or {}));result=[]
    observed=[i for i,r in enumerate(rows) if r.get('selected') is not None]
    for left,right in zip(observed,observed[1:]):
        gap=right-left-1
        if not 1<=gap<=c['max_gap_frames']:continue
        if any(rows[k].get('camera_affine') is None or rows[k]['frame']!=rows[k-1]['frame']+1 for k in range(left+1,right+1)):continue
        poses=gap_poses(rows,left,right);start=np.asarray(rows[left]['selected'][:2]);end=apply(np.linalg.inv(poses[-1]),rows[right]['selected'])
        if np.linalg.norm(end-start)/(right-left)>c['max_speed_px_frame']:continue
        for i in range(left+1,right):
            dt=i-left;duration=right-left;u=dt/duration
            local=(1-u)*start+u*end
            tolerance=c['base_tolerance_px']+c['acceleration_allowance']*dt*(duration-dt)/2
            scale=np.linalg.norm(poses[dt][:2,:2],ord=2)
            result.append(dict(index=i,center=apply(poses[dt],local),tolerance=tolerance*scale,
                left=left,right=right,pose=poses[dt],start=start,end=end,duration=duration,dt=dt))
    return result


def choose(points,plan,config=None):
    c=dict(CONFIG,**(config or {}));scored=[]
    for p in points:
        p=np.asarray(p,float)
        if p.shape!=(3,) or not np.isfinite(p).all() or p[2]<c['min_confidence']:continue
        distance=float(np.linalg.norm(p[:2]-plan['center']))
        if distance>plan['tolerance']:continue
        local=apply(np.linalg.inv(plan['pose']),p)
        if np.linalg.norm(local-plan['start'])/plan['dt']>c['max_speed_px_frame']:continue
        if np.linalg.norm(plan['end']-local)/(plan['duration']-plan['dt'])>c['max_speed_px_frame']:continue
        scored.append((p[2]-.5*distance/plan['tolerance'],p))
    scored.sort(key=lambda v:-v[0])
    if not scored:return None
    if len(scored)>1 and scored[0][0]-scored[1][0]<c['ambiguity_score_margin'] and np.linalg.norm(scored[0][1][:2]-scored[1][1][:2])>c['ambiguity_distance_px']:return None
    return scored[0][1].tolist()


def crop_box(center,radius,width,height):
    size=radius*2;x=int(np.clip(round(center[0]-radius),0,max(0,width-size)));y=int(np.clip(round(center[1]-radius),0,max(0,height-size)))
    return [x,y,min(width,x+size),min(height,y+size)]


def recover(rows,load_image,detect,config=None):
    c=dict(CONFIG,**(config or {}));out=copy.deepcopy(rows);search=plans(rows,c);logs=[]
    for n,plan in enumerate(search):
        i=plan['index'];image=load_image(rows[i]['frame']);h,w=image.shape[:2];accepted=None;attempts=[]
        for radius in c['radii']:
            box=crop_box(plan['center'],radius,w,h);points=detect(image,box)
            accepted=choose(points,plan,c);attempts.append(dict(crop=box,candidates=points))
            if accepted is not None:break
        row=out[i];was_missing=row.get('context_position') is None
        row['short_gap_anchors']=[rows[plan['left']]['frame'],rows[plan['right']]['frame']]
        if accepted is not None:
            row.update(context_position=accepted[:2],context_source='short_gap_detection',short_gap_detection=accepted)
        elif was_missing:
            row.update(context_position=plan['center'].tolist(),context_source='short_gap_estimate')
        row['short_gap_uncertainty_px']=float(plan['tolerance'])
        logs.append(dict(frame=row['frame'],accepted=accepted,attempts=attempts))
        if (n+1)%50==0:print(f'{n+1}/{len(search)} short-gap frames searched',flush=True)
    return out,logs


def run(args):
    import cv2,torch
    from frame_preprocessing_experiment import detect as yolo_detect
    from ultralytics import YOLO
    cache,out=private_path(args.cache),private_path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings');os.environ['YOLO_AUTOINSTALL']='False'
    source_config=json.loads((cache/'config.json').read_text())
    probe=None
    if args.probe:
        probe=json.loads((private_path(args.probe)/'result.json').read_text());rows=[]
        source={r['frame']:r for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())}
        for r in probe['rows']:
            s=r['window_smoothing'][args.mode];row=copy.deepcopy(source[r['frame']])
            row.update(selected=s['selected'],context_position=s['position'],context_source=s['source']);rows.append(row)
    else:rows=list(map(json.loads,(cache/'detections.jsonl').read_text().splitlines()))
    cap=CachedVideoCapture(source_config['source']);model=YOLO(ROOT/'outputs/two-stage-cv-models/yolo11x.pt');torch.set_num_threads(2);cv2.setNumThreads(2)
    def load(frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        if not ok:raise ValueError('Video decode failed')
        return image
    def detect(image,box):return yolo_detect(model,image,args.device,1280,box).tolist()
    out.mkdir(parents=True);started=time.monotonic()
    try:result,logs=recover(rows,load,detect)
    finally:cap.release()
    summary=dict(frames=len(rows),searched_frames=len(logs),new_direct_detections=sum(r.get('short_gap_detection') is not None for r in result),
        crops=sum(len(r['attempts']) for r in logs),new_position_frames=sum(a.get('context_position') is None and b.get('context_position') is not None for a,b in zip(rows,result)),
        elapsed_seconds=time.monotonic()-started,accuracy_measured=False)
    config=copy.deepcopy(source_config);config['short_gap_recovery']=dict(CONFIG,code_sha256=digest(__file__))
    (out/'config.json').write_text(json.dumps(config,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'detections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result));(out/'searches.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in logs))
    if probe:
        import shutil
        for original,r in zip(probe['rows'],result):
            s=original['window_smoothing'][args.mode];s.update(position=r['context_position'],source='smoothed_gap' if r['context_source']=='short_gap_estimate' else r['context_source'])
        probe['short_gap_summary']=summary
        (out/'result.json').write_text(json.dumps(probe,indent=2));(out/'index.html').write_bytes((ROOT/'scripts/neighbor_frames_view.html').read_bytes())
        for image in private_path(args.probe).glob('*.jpg'):os.link(image,out/image.name)
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--probe',type=Path);p.add_argument('--mode',default='fixed_raw_1280');p.add_argument('--device',default='mps');run(p.parse_args())
