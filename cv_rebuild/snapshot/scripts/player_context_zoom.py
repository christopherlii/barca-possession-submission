"""Add full-player context crops to an existing crowded zoom comparison.

Retains all old predictions. New candidates are explicit diagnostic proposals,
never possession labels or automatic replacements for the current ball track.
"""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from stage1_cv_baseline import ROOT,private_path,digest
from local_cv_possession import atomic
from crowded_ball_zoom import player_context_crop,detect


def run(cache,out,device,features=None):
    import cv2,torch
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings');os.environ['YOLO_AUTOINSTALL']='False'
    from ultralytics import YOLO
    cache,out=private_path(cache),private_path(out)
    if out.exists():raise ValueError('Use a new output directory')
    rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    config=json.loads((cache/'config.json').read_text())
    feature_path=private_path(features or ROOT/'outputs/two-stage-cv-view/tactical-fused-v1.npz')
    if digest(feature_path)!=config['features_sha256']:raise ValueError('Feature source differs from cache')
    with np.load(feature_path,allow_pickle=False) as z:
        frames=z['frame_ids'];guides=z['atd_ball_xy'];boxes=z['player_boxes'];offset=z['player_offsets'];meta=json.loads(str(z['metadata']))
    if [r['frame'] for r in rows]!=frames.tolist():raise ValueError('Frame alignment differs')
    config['player_context_zoom']=dict(code_sha256=digest(__file__),crop_code_sha256=digest(ROOT/'scripts/crowded_ball_zoom.py'),
        parent_sha256=digest(cache/'detections.jsonl'),parent_cache=str(cache),imgsz=1280,full_body_margin_height_fraction=.2,
        purpose='Cover held ball above ground guide; no player identity or possession inference')
    out.mkdir(parents=True);atomic(out/'config.json',config)
    torch.set_num_threads(2);cv2.setNumThreads(2);model=YOLO(ROOT/'outputs/two-stage-cv-models/yolo11x.pt')
    cap=cv2.VideoCapture(meta['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
    tested=found=0;start=time.monotonic()
    try:
        with (out/'detections.jsonl').open('w') as handle:
            for i,row in enumerate(rows):
                ok,image=cap.read()
                if not ok:raise ValueError('Frame decode failed')
                crop=player_context_crop(guides[i],boxes[offset[i]:offset[i+1]],meta['width'],meta['height'])
                points=detect(model,image,[crop],device) if crop and row.get('crowded_zoom_crops') else np.empty((0,3))
                tested+=bool(crop and row.get('crowded_zoom_crops'));found+=bool(len(points))
                item=dict(row,player_context_crop=crop,player_context_candidates=points.tolist(),
                          player_context_inferred=bool(crop and row.get('crowded_zoom_crops')))
                handle.write(json.dumps(item,allow_nan=False)+'\n');handle.flush()
                if (i+1)%500==0:
                    atomic(out/'status.json',dict(state='running',completed=i+1,total=len(rows)))
                    print(f'{i+1}/{len(rows)} context frames, {tested} detector crops, {found} candidate frames',flush=True)
    finally:cap.release()
    summary=dict(frames=len(rows),detector_crops=tested,candidate_frames=found,elapsed_seconds=time.monotonic()-start,
                 existing_positions_unchanged=True,accuracy_measured=False)
    atomic(out/'summary.json',summary);atomic(out/'status.json',dict(state='complete',completed=len(rows),total=len(rows)))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='mps')
    p.add_argument('--features',type=Path)
    a=p.parse_args();run(a.cache,a.out,a.device,a.features)
