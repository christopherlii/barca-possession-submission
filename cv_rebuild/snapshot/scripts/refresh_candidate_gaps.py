"""Refresh detector proposals in short image-evidence gaps across a cache.

Bracketing detections define search crops only. This stage never accepts a ball
position or converts interpolation into evidence. New proposals must pass the
subsequent sequence/SAM verification. Source video is decoded at native resolution.
"""
from cached_video_reader import CachedVideoCapture
import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
from stage1_cv_baseline import ROOT, private_path, digest
from local_cv_possession import atomic
from local_cv_tiling import merge_points
from recover_blank_tracks import candidates, trusted, search_plans
from extend_sam_ball_tracks import is_target
from sam2_gap_recovery import transport
from sam_confirmation_retry import crop_box
from sam_confirmation_recovery import read_rows

CONFIG = dict(max_gap_frames=5, max_candidates=4, max_crops=3, crop_size=256,
              detail_size=128, detector_size=640, max_speed_px_frame=45.)


def gap_plans(rows):
    missing = [is_target(r) and trusted(r) is None and not candidates(r) for r in rows]
    jobs, i = [], 0
    while i < len(rows):
        if not missing[i]:
            i += 1
            continue
        j = i+1
        while j < len(rows) and missing[j]:
            j += 1
        if i > 0 and j < len(rows) and j-i <= CONFIG['max_gap_frames']:
            same_shot = all(rows[k].get('camera_affine') is not None
                            and rows[k]['frame'] == rows[k-1]['frame']+1 for k in range(i,j+1))
            if same_shot and evidence(rows[i-1]) and evidence(rows[j]):
                jobs.append(dict(left=i-1, right=j, indices=list(range(i,j))))
        i = j
    return jobs


def evidence(row):
    measured = trusted(row)
    return [[*measured,.99]] if measured is not None else sorted(candidates(row),key=lambda p:-p[2])[:CONFIG['max_candidates']]


def crops_for_gap(rows, job, index, width, height):
    left, right = job['left'], job['right']
    pairs = []
    for a in evidence(rows[left]):
        for b in evidence(rows[right]):
            end = transport(b[:2],rows,right,left)
            if np.linalg.norm(end-a[:2]) > CONFIG['max_speed_px_frame']*(right-left):
                continue
            u = (index-left)/(right-left)
            center = transport((1-u)*np.asarray(a[:2])+u*end,rows,left,index)
            if 0 <= center[0] < width and 0 <= center[1] < height:
                pairs.append((a[2]+b[2],center))
    plans = []
    for _, center in sorted(pairs,key=lambda p:-p[0]):
        if any(np.linalg.norm(center-p['center'])<64 for p in plans):
            continue
        plans.append(dict(source='candidate_endpoint_search',center=center.tolist(),
                          box=crop_box(center,CONFIG['crop_size'],width,height),mode='raw'))
        if len(plans)==CONFIG['max_crops']:
            break
    if plans and len(plans)<CONFIG['max_crops']:
        center=plans[0]['center']
        plans.append(dict(source='candidate_endpoint_detail',center=center,
                          box=crop_box(center,CONFIG['detail_size'],width,height),mode='clahe'))
    return plans or search_plans(rows,index,width,height)


def run(args):
    import cv2
    import torch
    from frame_preprocessing_experiment import preprocess, detect
    out=private_path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    rows=read_rows(args.cache/'detections.jsonl')
    config=json.loads((args.cache/'config.json').read_text())
    if config.get('continuous_trajectory'):raise ValueError('Use the unfilled evidence cache')
    jobs=gap_plans(rows)
    out.mkdir(parents=True)
    signature=dict(config=CONFIG,source_sha256=digest(args.cache/'detections.jsonl'),code_sha256=digest(__file__),
                   detector_sha256=digest(ROOT/'outputs/two-stage-cv-models/yolo11x.pt'),device=args.device)
    atomic(out/'plan.json',dict(signature,jobs=jobs))
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings')
    os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    cv2.setNumThreads(2);torch.set_num_threads(2)
    model=YOLO(ROOT/'outputs/two-stage-cv-models/yolo11x.pt')
    cap=CachedVideoCapture(config['source'])
    if not cap.isOpened():raise ValueError('Cannot open source video')
    result=copy.deepcopy(rows)
    completed=found_frames=0
    total=sum(len(job['indices']) for job in jobs)
    start=time.monotonic()
    try:
        with (out/'searches.jsonl').open('w') as log:
            for job in jobs:
                cap.set(cv2.CAP_PROP_POS_FRAMES,rows[job['indices'][0]]['frame'])
                for i in job['indices']:
                    ok,image=cap.read()
                    if not ok:raise ValueError('Video decode failed')
                    plans=crops_for_gap(rows,job,i,image.shape[1],image.shape[0])
                    groups=[]
                    for plan in plans:
                        x,y,xx,yy=plan['box']
                        size=1280 if plan['source']=='full_frame' else CONFIG['detector_size']
                        points=detect(model,preprocess(image[y:yy,x:xx],plan['mode']),args.device,size)
                        groups.append([[p[0]+x,p[1]+y,p[2]] for p in points])
                    fresh=merge_points(groups).tolist()
                    result[i]['blank_recovery_candidates']=merge_points([rows[i].get('blank_recovery_candidates') or [],fresh]).tolist()
                    result[i]['candidate_gap_search']=dict(plans=plans,candidates=fresh)
                    log.write(json.dumps(dict(frame=rows[i]['frame'],index=i,plans=plans,candidates=fresh))+'\n');log.flush()
                    completed+=1
                    found_frames+=int(bool(candidates(result[i])))
                    if completed%10==0 or completed==total:
                        status=dict(state='running',completed=completed,total=total,candidate_frames=found_frames,
                                    elapsed_seconds=time.monotonic()-start)
                        atomic(out/'status.json',status);print(json.dumps(status),flush=True)
    finally:cap.release()
    assert all(a.get('context_position')==b.get('context_position') for a,b in zip(rows,result))
    config['candidate_gap_refresh']=signature
    summary=dict(frames=len(rows),gap_runs=len(jobs),searched_frames=completed,new_candidate_frames=found_frames,
                 changed_positions=0,elapsed_seconds=time.monotonic()-start,accuracy_measured=False)
    atomic(out/'config.json',config)
    with (out/'detections.jsonl').open('w') as log:
        for row in result:log.write(json.dumps(row,allow_nan=False)+'\n')
    atomic(out/'summary.json',summary);atomic(out/'status.json',dict(state='complete',**summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='mps')
    run(parser.parse_args())
