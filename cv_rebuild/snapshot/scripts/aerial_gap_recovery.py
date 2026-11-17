"""Conservative offline aerial-gap recovery; existing visible output is immutable.

Search original pixels above ground-projected guides in overlapping native-size
tiles. Only bounded gaps with observations on both sides are eligible. A path
must join both endpoints and contain multiple motion-consistent observations.
No target labels, ball-height claims, or arbitrary track reacquisition.
"""
from cached_video_reader import CachedVideoCapture
import argparse
import copy
import json
import os
import time
from pathlib import Path
import numpy as np
from stage1_cv_baseline import ROOT,private_path,digest
from local_cv_possession import atomic
from small_ball_preprocessing import native_tiles

CONFIG=dict(max_gap_frames=50,sample_stride=2,tile_size=512,tile_overlap=128,
            horizontal_margin_px=180.,curve_margin_px=240.,max_speed_px_frame=55.,
            max_acceleration_px_frame2=8.,endpoint_slack_px=35.,
            min_observations=4,min_travel_px=25.,max_step_frames=6,
            confidence=.15,missing_penalty=.65,min_gain=1.5)


def apply(pose,p):
    return (pose@np.r_[p[:2],1])[:2]


def anchored_gaps(rows,config=None):
    c=dict(CONFIG,**(config or {}));gaps=[];i=1
    while i<len(rows)-1:
        if rows[i].get('context_position') is not None:i+=1;continue
        j=i
        while j<len(rows) and rows[j].get('context_position') is None:j+=1
        # Direct measurements, not forecasts, must bound both sides.
        left=i-1;right=j
        # A gap often touches the online track's five predicted frames. Walk
        # through that short estimate tail to find actual measured anchors.
        while left>=max(0,i-6) and rows[left].get('selected') is None:left-=1
        while right<min(len(rows),j+6) and rows[right].get('selected') is None:right+=1
        if (0<=left<right<len(rows) and right-left-1<=c['max_gap_frames']
            and rows[left].get('selected') is not None and rows[right].get('selected') is not None
            and all(rows[k].get('camera_affine') is not None and rows[k]['frame']==rows[k-1]['frame']+1 for k in range(left+1,right+1))):
            if (left,right) not in gaps:gaps.append((left,right))
        i=j+1
    return gaps


def gap_poses(rows,left,right):
    poses=[np.eye(3)]
    for row in rows[left+1:right+1]:
        affine=np.eye(3);affine[:2]=row['camera_affine']
        poses.append(affine@poses[-1])
    return poses


def search_tiles(width,height,center,guide=None,config=None):
    """Include the skyward corridor, without resizing the whole scene."""
    c=dict(CONFIG,**(config or {}));x,y=np.asarray(center,float)
    low=y-c['curve_margin_px'];high=y+c['curve_margin_px']*.4
    if guide is not None and np.isfinite(guide).all():
        # Ground projection can be below an airborne ball; never use it as a
        # hard center. Include both the motion prediction and the space above it.
        low=min(low,float(guide[1])-c['curve_margin_px']*1.5)
        high=max(high,float(guide[1])+40.)
    x1=max(0,int(x-c['horizontal_margin_px']));x2=min(width,int(x+c['horizontal_margin_px']))
    y1=max(0,int(low));y2=min(height,int(high))
    if x2-x1<32 or y2-y1<32:return []
    return [(x+x1,y+y1,xx+x1,yy+y1) for x,y,xx,yy in native_tiles(x2-x1,y2-y1,c['tile_size'],c['tile_overlap'])]


def recover_path(left,right,samples,config=None):
    """Select a supported path between pixel anchors in a camera-fixed plane.

    Nodes are (relative frame, x/y/confidence). Dynamic-programming edges limit
    speed and favor the anchor-to-anchor velocity; skips cost missing evidence.
    Final validation checks acceleration, spatial progress and observation count.
    Only observed nodes are returned; this function invents no pixel locations.
    """
    c=dict(CONFIG,**(config or {}));end_frame=int(right[0]);start=np.asarray(left[1:3],float);end=np.asarray(right[1:3],float)
    average=(end-start)/end_frame
    if np.linalg.norm(end-start)<c['min_travel_px']:return []
    nodes=[(0,start,1.)]
    for frame,points in sorted(samples.items()):
        if not 0<frame<end_frame:continue
        for p in points:
            p=np.asarray(p,float)
            expected=start+(end-start)*frame/end_frame
            if (p.shape==(3,) and np.isfinite(p).all() and p[2]>=c['confidence']
                and abs(p[0]-expected[0])<=c['horizontal_margin_px']
                and abs(p[1]-expected[1])<=c['curve_margin_px']):nodes.append((int(frame),p[:2],float(p[2])))
    nodes.append((end_frame,end,1.));scores=np.full(len(nodes),-np.inf);scores[0]=0;parents=np.full(len(nodes),-1,int)
    for j in range(1,len(nodes)):
        t,p,confidence=nodes[j]
        for k in range(j-1,-1,-1):
            s,q,_=nodes[k];dt=t-s
            if dt<=0 or dt>c['max_step_frames']:continue
            velocity=(p-q)/dt
            if np.linalg.norm(velocity)>c['max_speed_px_frame']:continue
            previous=parents[k]
            if previous>=0:
                previous_t,previous_p,_=nodes[previous]
                old_velocity=(q-previous_p)/(s-previous_t)
                if np.linalg.norm(velocity-old_velocity)/((dt+s-previous_t)/2)>c['max_acceleration_px_frame2']:continue
            # Reward repeated observations, not high confidence at a lone spot.
            value=scores[k]+1.+confidence-c['missing_penalty']*(dt-1)-.025*np.linalg.norm(velocity-average)
            if value>scores[j]:scores[j]=value;parents[j]=k
    if not np.isfinite(scores[-1]) or scores[-1]<c['min_gain']:return []
    path=[];j=len(nodes)-1
    while j>=0:
        path.append(nodes[j]);j=parents[j]
    path.reverse();observed=path[1:-1]
    if len(observed)<c['min_observations']:return []
    velocities=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(path,path[1:])]
    for i in range(1,len(velocities)):
        dt=(path[i+1][0]-path[i-1][0])/2
        if np.linalg.norm(velocities[i]-velocities[i-1])/dt>c['max_acceleration_px_frame2']:return []
    # Exclude paths that mostly dwell on a fixed marking then jump to an anchor.
    if len(observed)>2:
        moving=sum(np.linalg.norm(b[1]-a[1])>1. for a,b in zip(observed,observed[1:]))
        if moving/(len(observed)-1)<.6:return []
    return [(t,np.r_[p,score].tolist()) for t,p,score in observed]


def run(cache,out,device):
    import cv2,torch
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings');os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    from frame_preprocessing_experiment import detect
    from local_cv_tiling import merge_points
    cache,out=private_path(cache),private_path(out)
    if out.exists():raise ValueError('New output required')
    rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    config=json.loads((cache/'config.json').read_text());gaps=anchored_gaps(rows)
    config.update(aerial_gap_recovery=dict(CONFIG,code_sha256=digest(__file__),parent_sha256=digest(cache/'detections.jsonl'),
        raw_rgb=True,uses_future_anchors=True),new_detector_inference=True)
    out.mkdir(parents=True);atomic(out/'config.json',config)
    result=copy.deepcopy(rows);model=YOLO(ROOT/'outputs/two-stage-cv-models/yolo11x.pt');torch.set_num_threads(2);cv2.setNumThreads(2)
    cap=CachedVideoCapture(config['source']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    added=0;inferred=0;recovered=0;started=time.monotonic()
    try:
        for n,(left,right) in enumerate(gaps):
            poses=gap_poses(rows,left,right);inv=[np.linalg.inv(p) for p in poses]
            p0=np.asarray(rows[left]['selected'][:2]);p1=apply(inv[-1],rows[right]['selected']);samples={};tile_log={}
            for k in range(left+1,right,CONFIG['sample_stride']):
                dt=k-left;center=apply(poses[dt],p0+(p1-p0)*dt/(right-left))
                tiles=search_tiles(w,h,center,rows[k].get('guide'));tile_log[k]=tiles
                cap.set(cv2.CAP_PROP_POS_FRAMES,rows[k]['frame']);ok,image=cap.read()
                if not ok:raise ValueError('Frame decode failed')
                points=merge_points([detect(model,image,device,640,box) for box in tiles])
                inferred+=len(tiles)
                # All marking proximity alone remains permitted: fixed-marking
                # dwell is rejected by temporal path validation instead.
                samples[dt]=[np.r_[apply(inv[dt],p),p[2]].tolist() for p in points]
            path=recover_path([0,*p0],[right-left,*p1],samples)
            if path:
                recovered+=1
                for dt,p in path:
                    k=left+dt;xy=apply(poses[dt],p).tolist()
                    if rows[k].get('context_position') is not None:continue
                    result[k].update(context_position=xy,context_source='aerial_observed_path',
                        aerial_detection=[*xy,p[2]],aerial_search_tiles=tile_log[k],
                        aerial_anchor_frames=[rows[left]['frame'],rows[right]['frame']])
                    added+=1
                # Fill at most one unsampled frame, bracketed by verified path
                # nodes, and only inside the previously missing region.
                for (ta,a),(tb,b) in zip(path,path[1:]):
                    if tb-ta!=2:continue
                    dt=ta+1;k=left+dt
                    if rows[k].get('context_position') is not None:continue
                    point=(np.asarray(a[:2])+np.asarray(b[:2]))/2
                    result[k].update(context_position=apply(poses[dt],point).tolist(),context_source='aerial_interpolated_path',
                        aerial_anchor_frames=[rows[left]['frame'],rows[right]['frame']])
            atomic(out/'status.json',dict(state='running',gaps_done=n+1,gaps_total=len(gaps),new_observations=added))
            print(f'{n+1}/{len(gaps)} bounded gaps; {added} new supported observations',flush=True)
    finally:cap.release()
    assert all(a['context_position']==b['context_position'] for a,b in zip(rows,result) if a['context_position'] is not None)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    summary=dict(frames=len(rows),eligible_gaps=len(gaps),supported_gaps=recovered,new_observations=added,
        additional_visible_frames=sum(a['context_position'] is None and b['context_position'] is not None for a,b in zip(rows,result)),
        detector_crops=inferred,elapsed_seconds=time.monotonic()-started,existing_visible_positions_unchanged=True,accuracy_measured=False)
    atomic(out/'summary.json',summary);atomic(out/'status.json',dict(state='complete',completed=len(rows),total=len(rows)))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='mps')
    a=p.parse_args();run(a.cache,a.out,a.device)
