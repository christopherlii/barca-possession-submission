"""Run ball CV on CLAHE frames cropped around same-frame automatic ATD.

No targets or hand corrections. Missing/offscreen ATD falls back to a full-frame
pass. Candidates are measured by YOLO, never fabricated from the guiding point.
Original fused/whole-frame outputs remain intact and guide/crop provenance is
recorded separately. Player detections are reused from the original CV cache.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from stage1_cv_baseline import ROOT, private_path, digest
from frame_preprocessing_experiment import preprocess, guided_box, detect
from local_cv_possession import atomic
from ball_motion_context import BallMotion, smooth_path
from guide_search_region import plan_search_region


def select_point(points, guide):
    points = np.asarray(points, float).reshape(-1, 3)
    if not len(points): return None
    if guide is not None and np.isfinite(guide).all():
        # Current-frame ATD is a location prior, not a visual detection.
        index = np.argmin(np.linalg.norm(points[:, :2]-guide, axis=1))
    else:
        index = np.argmax(points[:, 2])
    return points[index].tolist()


def run(args):
    import cv2, torch
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT/'outputs/two-stage-cv-settings')
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    from ultralytics import YOLO
    out = private_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    meta = json.loads(str(data['metadata']))
    if meta.get('feature_mode') != 'tactical_cv_fusion':
        raise ValueError('Expected frozen same-camera tactical fusion cache')
    frames = data['frame_ids']
    if not np.all(np.diff(frames)==1): raise ValueError('Nonconsecutive frames')
    weights = ROOT/'outputs/two-stage-cv-models/yolo11x.pt'
    config = dict(features_sha256=digest(args.features), code_sha256=digest(__file__),
        preprocess_code_sha256=digest(ROOT/'scripts/frame_preprocessing_experiment.py'),
        model_sha256=digest(weights), source=meta['source'], frames=len(frames),
        start_frame=int(frames[0]), crop_radius=args.radius, input_size=640,
        confidence=.1, preprocess='CLAHE full native frame before cropping', clahe_clip=args.clahe_clip,
        temporal=args.temporal, motion_code_sha256=digest(ROOT/'scripts/ball_motion_context.py'),
        guide='same-frame ATD projected via valid homography, reprojection error <=2',
        missing_guide='full-frame CLAHE inference', candidate_selection='nearest ATD, otherwise highest score',
        answers_read=False, fitting=False, accuracy_measured=False,
        guide_inclusive_search=args.guide_inclusive,search_region_code_sha256=digest(ROOT/'scripts/guide_search_region.py'))
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=config:
        raise ValueError('Locked run differs; use new directory')
    atomic(out/'config.json', config)
    raw = out/'detections.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    if [r['frame'] for r in rows]!=frames[:len(rows)].tolist():
        raise ValueError('Noncontiguous cached detections')
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    motion=BallMotion()
    if args.temporal or args.guide_inclusive:
        for i,row in enumerate(rows):
            motion.update(row['candidates'],row.get('guide'),i,
                          i==0 or data['segments'][i]!=data['segments'][i-1])
    model = YOLO(weights)
    cap = cv2.VideoCapture(meta['source'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0])+len(rows))
    started = time.monotonic()
    try:
        with raw.open('a') as handle:
            for i in range(len(rows),len(frames)):
                ok,image = cap.read()
                if not ok: raise ValueError('Source frame decode failed')
                guide = data['atd_ball_xy'][i]
                cut=i==0 or data['segments'][i]!=data['segments'][i-1]
                if cut:motion.history=[]
                forecast=motion.forecast(i) if (args.temporal or args.guide_inclusive) else None
                center=guide if np.isfinite(guide).all() else forecast
                region=plan_search_region(guide,forecast,meta['width'],meta['height'],args.radius) if args.guide_inclusive else None
                box = region['crop'] if region is not None else guided_box(center,meta['width'],meta['height'],args.radius)
                # Filter prior to crop so it matches the full-frame CLAHE display.
                lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
                lab[:,:,0]=cv2.createCLAHE(clipLimit=args.clahe_clip,tileGridSize=(8,8)).apply(lab[:,:,0])
                filtered=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
                points = detect(model,filtered,args.device,640,box)
                selected,predicted=motion.update(points,guide if np.isfinite(guide).all() else None,i,cut) if args.temporal else (select_point(points,guide if box else None),None)
                if args.guide_inclusive and not args.temporal:
                    motion.update(points,guide if np.isfinite(guide).all() else None,i,cut)
                row = dict(frame=int(frames[i]),crop=list(box) if box else None,search_region=region,
                    guide=guide.tolist() if np.isfinite(guide).all() else None,candidates=points.tolist(),
                    selected=selected,motion_prediction=predicted,
                    crop_source='atd' if np.isfinite(guide).all() and box else 'motion' if box else 'full_frame',
                    method='guided_crop' if box else 'full_frame_fallback')
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();rows.append(row)
                if len(rows)%100==0:
                    atomic(out/'status.json',dict(state='running',completed=len(rows),total=len(frames)))
                    print(f'{len(rows)}/{len(frames)} guided/CLAHE frames',flush=True)
    finally:
        cap.release()
    if args.temporal: rows=smooth_path(rows,data['segments'])
    atomic(out/'context.json',rows)
    selected=np.array([r['selected'][:2] if r['selected'] else [np.nan,np.nan] for r in rows])
    crops=np.array([r['crop'] if r['crop'] else [np.nan]*4 for r in rows])
    meta.update(guided_cv=config, detector_preprocessing='clahe',
        visual_ball_method='YOLO on CLAHE tracking-guided crops; current-frame guide only',
        inference_layers='new guided ball detections; cached original player boxes and ATD baseline',
        transformer_for_guided_run=False)
    data.update(metadata=np.array(json.dumps(meta)), previous_cv_ball_xy=data['cv_ball_xy'],
        cv_ball_xy=selected,guided_crops=crops,
        context_positions=np.array([r.get('context_position') or [np.nan,np.nan] for r in rows]),
        guided_visible=np.isfinite(selected).all(1).astype(np.uint8),
        guided_ball_score=np.array([r['selected'][2] if r['selected'] else np.nan for r in rows]))
    np.savez_compressed(out/'viewer.npz',**data)
    summary=dict(frames=len(rows),guided_frames=sum(r['crop'] is not None for r in rows),
        fallback_frames=sum(r['crop'] is None for r in rows),
        cv_detected_frames=int(np.isfinite(selected).all(1).sum()),
        candidate_count=sum(len(r['candidates']) for r in rows),
        elapsed_this_run=time.monotonic()-started,accuracy_measured=False,
        viewer_sha256=digest(out/'viewer.npz'))
    atomic(out/'summary.json',summary)
    atomic(out/'status.json',dict(state='complete',completed=len(rows),total=len(frames)))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--radius',type=int,default=160)
    p.add_argument('--device',default='mps')
    p.add_argument('--clahe-clip',type=float,default=2.)
    p.add_argument('--temporal',action='store_true')
    p.add_argument('--guide-inclusive',action=argparse.BooleanOptionalAction,default=True,help='Always include the tactical guide in the detector search region, with bounded motion lead')
    args=p.parse_args()
    if args.radius<32:p.error('Crop radius must be at least 32 pixels')
    if not 0<args.clahe_clip<=10:p.error('CLAHE clip must be in (0,10]')
    run(args)
