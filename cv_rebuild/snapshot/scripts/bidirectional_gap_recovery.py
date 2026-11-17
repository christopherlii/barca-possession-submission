"""Offline, endpoint-conditioned ball search through unobserved video intervals.

Accepted positions are immutable. Both endpoint velocities guide image crops;
a beam search joins candidate observations between the endpoints and refuses
unsupported long jumps. New interpolations are explicitly distinguished from
observations. No possession labels or supplied-match ground truth are read.
"""
from cached_video_reader import CachedVideoCapture
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from aerial_gap_recovery import apply, gap_poses
from stage1_cv_baseline import ROOT, private_path, digest
from local_cv_possession import atomic

CONFIG = dict(max_gap_frames=150, anchor_window=8, minimum_anchor_observations=3,
              sample_stride=3, crop_size=512, max_crops_per_sample=3,
              crop_margin_px=140., vertical_margin_px=220.,
              confidence=.12, max_candidates=10, beam_width=8,
              max_step_frames=9, max_speed_px_frame=70.,
              max_acceleration_px_frame2=2., endpoint_velocity_slack=12.,
              min_observations=4, min_observation_fraction=.75, min_travel_px=20., interpolation_gap_frames=12,
              missing_cost=.22, acceleration_cost=.2, endpoint_cost=.025,
              corridor_px=240.)


def finite_point(value):
    return value is not None and len(value) >= 2 and np.isfinite(value[:2]).all()


def endpoint_velocity(rows, index, direction, config=None):
    """Robust local velocity in the endpoint's camera coordinate system."""
    c = dict(CONFIG, **(config or {})); samples = []
    pose = np.eye(3)
    for distance in range(c['anchor_window'] + 1):
        j = index + direction * distance
        if not 0 <= j < len(rows): break
        if distance:
            edge = j if direction > 0 else j + 1
            if rows[edge]['frame'] != rows[edge-1]['frame']+1: break
            affine = rows[edge].get('camera_affine')
            if affine is None: break
            h = np.eye(3); h[:2] = affine
            pose = (h if direction > 0 else np.linalg.inv(h)) @ pose
        p = rows[j].get('selected')
        if finite_point(p): samples.append((direction*distance, apply(np.linalg.inv(pose), p)))
    if len(samples) < c['minimum_anchor_observations']: return None
    samples.sort(key=lambda s: s[0])
    velocities = [(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(samples,samples[1:])]
    velocity = np.median(velocities, axis=0)
    if np.linalg.norm(velocity) > c['max_speed_px_frame']: return None
    return velocity


def discover_gaps(rows, config=None):
    """Connect stable direct anchors, including across online track expiry."""
    c = dict(CONFIG, **(config or {})); result = []; i = 1
    while i < len(rows)-1:
        if finite_point(rows[i].get('context_position')): i += 1; continue
        end = i
        while end < len(rows) and not finite_point(rows[end].get('context_position')): end += 1
        left, right = i-1, end
        while left >= max(0,i-c['anchor_window']) and not finite_point(rows[left].get('selected')): left -= 1
        while right < min(len(rows),end+c['anchor_window']) and not finite_point(rows[right].get('selected')): right += 1
        if (0 <= left < right < len(rows) and right-left-1 <= c['max_gap_frames']
            and finite_point(rows[left].get('selected')) and finite_point(rows[right].get('selected'))
            and all(rows[k].get('camera_affine') is not None and rows[k]['frame']==rows[k-1]['frame']+1 for k in range(left+1,right+1))):
            v0 = endpoint_velocity(rows,left,-1,c); v1 = endpoint_velocity(rows,right,1,c)
            if v0 is not None and v1 is not None: result.append((left,right,v0,v1))
        i = end+1
    return result


def search_centers(start, end, v0, v1, t, duration, config=None):
    """Bridge both ends, plus bounded one-sided motion proposals."""
    c = dict(CONFIG, **(config or {})); u=t/duration
    chord = start+(end-start)*u
    forward = start+v0*t; backward = end-v1*(duration-t)
    # Divergent velocity forecasts must not grow a search to the entire image.
    def bounded(p):
        delta=p-chord; norm=np.linalg.norm(delta)
        return chord+delta*min(1.,c['corridor_px']/max(norm,1e-9))
    forward, backward = bounded(forward), bounded(backward)
    return [(1-u)*forward+u*backward, forward, backward, chord]


def search_crops(width, height, centers, config=None):
    c = dict(CONFIG, **(config or {})); size = c['crop_size']; crops=[]
    for x,y in centers:
        x=int(np.clip(round(x-size/2),0,max(0,width-size)))
        y=int(np.clip(round(y-size/2),0,max(0,height-size)))
        box=[x,y,min(width,x+size),min(height,y+size)]
        if not any(abs(x-old[0])<size*.35 and abs(y-old[1])<size*.35 for old in crops): crops.append(box)
        if len(crops)>=c['max_crops_per_sample']: break
    return crops


def solve_bridge(start, end, v0, v1, duration, samples, config=None):
    """Second-order beam search with constraints at BOTH endpoint velocities.

    Each state preserves its incoming velocity so an early locally attractive
    distractor does not force the only continuation. Future endpoint evidence
    scores complete paths, including those crossing online track resets.
    """
    c = dict(CONFIG, **(config or {})); start=np.asarray(start);end=np.asarray(end)
    if np.linalg.norm(end-start)<c['min_travel_px']: return []
    nodes=[(0,start,1.)]
    for t,points in sorted(samples.items()):
        if not 0<t<duration: continue
        chord=start+(end-start)*t/duration
        for p in sorted(points,key=lambda p:p[2],reverse=True)[:c['max_candidates']]:
            if len(p)==3 and np.isfinite(p).all() and p[2]>=c['confidence'] and np.linalg.norm(np.asarray(p[:2])-chord)<=c['corridor_px']:
                nodes.append((t,np.asarray(p[:2]),float(p[2])))
    nodes.append((duration,end,1.)); states=[[] for _ in nodes]
    states[0]=[(0.,np.asarray(v0),0,[0])]
    for j,(t,p,conf) in enumerate(nodes[1:],1):
        proposals=[]
        for k in range(j-1,-1,-1):
            s,q,_=nodes[k];dt=t-s
            if dt>c['max_step_frames']: break
            if dt<=0: continue
            velocity=(p-q)/dt
            if np.linalg.norm(velocity)>c['max_speed_px_frame']: continue
            if j==len(nodes)-1 and np.linalg.norm(velocity-v1)>c['endpoint_velocity_slack']: continue
            for score,old_velocity,old_dt,path in states[k]:
                if k==0 and np.linalg.norm(velocity-v0)>c['endpoint_velocity_slack']: continue
                acc=np.linalg.norm(velocity-old_velocity)/max(1.,(dt+old_dt)/2)
                if acc>c['max_acceleration_px_frame2']: continue
                reward=1.+conf-c['missing_cost']*(dt-1)-c['acceleration_cost']*acc
                if np.linalg.norm(p-q)<=1.:reward-=2.
                if j==len(nodes)-1:reward-=c['endpoint_cost']*np.linalg.norm(velocity-v1)
                proposals.append((score+reward,velocity,dt,path+[j]))
        proposals.sort(key=lambda p:p[0],reverse=True)
        states[j]=proposals[:c['beam_width']]
    if not states[-1]: return []
    for best in states[-1]:
        observed=[nodes[j] for j in best[3][1:-1]]
        if len(observed)<max(c['min_observations'],int(np.ceil((duration-1)/c['sample_stride']*c['min_observation_fraction']))): continue
        # Repeated stationary spots cannot masquerade as a travelling ball.
        moving=sum(np.linalg.norm(b[1]-a[1])>1. for a,b in zip(observed,observed[1:]))
        if len(observed)>1 and moving/(len(observed)-1)<.6: continue
        return [(t,[*p,confidence]) for t,p,confidence in observed]
    return []


def fill_bridge(rows, left, right, poses, path, config=None):
    """Add supported observations / bounded interpolation, preserving incumbents."""
    c=dict(CONFIG,**(config or {})); result=copy.deepcopy(rows);observed=interpolated=0
    points=[(0,rows[left]['selected'])]+path+[(right-left,[*apply(np.linalg.inv(poses[-1]),rows[right]['selected']),1.])]
    for t,p in path:
        i=left+t
        if finite_point(rows[i].get('context_position')):continue
        result[i].update(context_position=apply(poses[t],p).tolist(),context_source='bidirectional_observation',
                         bidirectional_detection=[*apply(poses[t],p),p[2]],bidirectional_anchors=[rows[left]['frame'],rows[right]['frame']])
        observed+=1
    for (ta,a),(tb,b) in zip(points,points[1:]):
        if tb-ta-1>c['interpolation_gap_frames']:continue
        for t in range(ta+1,tb):
            i=left+t
            if finite_point(result[i].get('context_position')):continue
            u=(t-ta)/(tb-ta);p=(1-u)*np.asarray(a[:2])+u*np.asarray(b[:2])
            result[i].update(context_position=apply(poses[t],p).tolist(),context_source='bidirectional_interpolated',
                             bidirectional_anchors=[rows[left]['frame'],rows[right]['frame']],
                             bidirectional_uncertainty_px=float(5.+min(t-ta,tb-t)*3.))
            interpolated+=1
    return result,observed,interpolated


def run(args):
    import cv2,torch
    from ultralytics import YOLO
    from local_cv_tiling import merge_points
    from frame_preprocessing_experiment import detect
    cache,out=private_path(args.cache),private_path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    config=json.loads((cache/'config.json').read_text());gaps=discover_gaps(rows)
    out.mkdir(parents=True)
    config.update(bidirectional_recovery=dict(CONFIG,code_sha256=digest(__file__),
        parent_sha256=digest(cache/'detections.jsonl'),uses_future=True,preserves_existing=True),new_detector_inference=True)
    atomic(out/'config.json',config)
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings');os.environ['YOLO_AUTOINSTALL']='False'
    model=YOLO(ROOT/'outputs/two-stage-cv-models/yolo11x.pt');torch.set_num_threads(2);cv2.setNumThreads(2)
    cap=CachedVideoCapture(config['source']);w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result=copy.deepcopy(rows);logs=[];crop_count=observed=interpolated=recovered=0;started=time.monotonic()
    try:
        for n,(left,right,v0,v1) in enumerate(gaps):
            poses=gap_poses(rows,left,right);inv=[np.linalg.inv(p) for p in poses]
            start=np.asarray(rows[left]['selected'][:2]);end=apply(inv[-1],rows[right]['selected']);v1=inv[-1][:2,:2]@v1
            samples={};crop_log={}
            for k in range(left+1,right,CONFIG['sample_stride']):
                t=k-left;centers=[apply(poses[t],p) for p in search_centers(start,end,v0,v1,t,right-left)]
                crops=search_crops(w,h,centers);crop_log[t]=crops
                cap.set(cv2.CAP_PROP_POS_FRAMES,rows[k]['frame']);ok,image=cap.read()
                if not ok:raise ValueError('Video decode failed')
                proposals=[np.asarray(rows[k].get(key) or []).reshape(-1,3) for key in ['candidates','crowded_zoom_candidates','player_context_candidates']]
                proposals += [detect(model,image,args.device,640,box) for box in crops]
                crop_count+=len(crops);points=merge_points(proposals)
                samples[t]=[[*apply(inv[t],p),p[2]] for p in points]
            path=solve_bridge(start,end,v0,v1,right-left,samples)
            if path:
                result,a,b=fill_bridge(result,left,right,poses,path);observed+=a;interpolated+=b;recovered+=1
            logs.append(dict(left_frame=rows[left]['frame'],right_frame=rows[right]['frame'],samples=len(samples),accepted_nodes=len(path),
                crops=crop_log,candidates=samples))
            atomic(out/'status.json',dict(state='running',completed=n+1,total=len(gaps),observations=observed,interpolated=interpolated))
            print(f'{n+1}/{len(gaps)} gaps; {observed} observations + {interpolated} estimates; {crop_count} crops; {time.monotonic()-started:.1f}s',flush=True)
    finally:cap.release()
    if any(a.get('context_position')!=b.get('context_position') for a,b in zip(rows,result) if finite_point(a.get('context_position'))):raise AssertionError('Existing position changed')
    (out/'searches.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in logs))
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    summary=dict(frames=len(rows),eligible_gaps=len(gaps),recovered_gaps=recovered,new_observations=observed,new_interpolations=interpolated,
        before_position_frames=int(sum(finite_point(r.get('context_position')) for r in rows)),after_position_frames=int(sum(finite_point(r.get('context_position')) for r in result)),
        detector_crops=crop_count,elapsed_seconds=time.monotonic()-started,existing_positions_unchanged=True,accuracy_measured=False)
    atomic(out/'summary.json',summary);atomic(out/'status.json',dict(state='complete',completed=len(gaps),total=len(gaps)))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='mps')
    run(p.parse_args())
