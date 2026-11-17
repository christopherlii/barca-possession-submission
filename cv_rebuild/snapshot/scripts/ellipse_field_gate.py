"""Pixel-space uncertainty ellipse + persistent pitch-marking distractor gate."""
import argparse
from collections import Counter
import csv
import json
import copy
import time
import numpy as np
from stage1_cv_baseline import ROOT,private_path,digest
from local_cv_possession import atomic
from image_ball_gate import PixelGate,CONFIG,background_motion
from field_marking_evidence import PersistentMarkingEvidence,project_geometry,PERSISTENT_CONFIG as FIELD_CONFIG
from ball_continuity import ContinuityGuard,CONFIG as CONTINUITY_CONFIG


def run(cache,out,recover=False,device='mps',features=None,ignore_field_boundary=False):
    import cv2
    cv2.setNumThreads(2);cv2.setRNGSeed(109)
    cache,out=private_path(cache),private_path(out)
    if out.exists():raise ValueError('New output directory required')
    config=json.loads((cache/'config.json').read_text())
    rows=[json.loads(r) for r in (cache/'detections.jsonl').read_text().splitlines()]
    source=private_path(features or ROOT/'outputs/two-stage-cv-view/tactical-fused-v1.npz')
    if digest(source)!=config['features_sha256']:raise ValueError('Source identity changed')
    with np.load(source,allow_pickle=False) as z:
        frames=z['frame_ids'];segments=z['segments'];offsets=z['player_offsets'];boxes=z['player_boxes'];meta=json.loads(str(z['metadata']))
    if [r['frame'] for r in rows]!=frames.tolist():raise ValueError('Incomplete frame sequence')
    homography=ROOT/'data/tactical/BAR-ATH_tactical_homography.csv';matrices={}
    with homography.open() as f:
        for row in csv.DictReader(f):
            frame=int(row['frame'])
            if frame<frames[0]:continue
            if frame>frames[-1]:break
            if int(row['valid'])==1 and float(row['reprojection_error'])<=2:
                matrices[frame]=np.array([[float(row[f'h{i}{j}']) for j in range(3)] for i in range(3)])
    settings=dict(CONFIG,ellipse=True,adaptive_process=True)
    field_settings=dict(FIELD_CONFIG,ignore_field_boundary=ignore_field_boundary)
    model=None
    if recover:
        import os,torch
        os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/two-stage-cv-settings')
        os.environ['YOLO_AUTOINSTALL']='False'
        from ultralytics import YOLO
        from ball_recovery import proposals,detect_recovery,CONFIG as RECOVERY_CONFIG
        from frame_preprocessing_experiment import preprocess
        from local_cv_tiling import merge_points
        torch.set_num_threads(2)
        model=YOLO(ROOT/'outputs/two-stage-cv-models/yolo11x.pt')
        config['recovery']=dict(RECOVERY_CONFIG,device=device,code_sha256=digest(ROOT/'scripts/ball_recovery.py'))
        config['new_detector_inference']=True
    config.update(image_gate=settings,field_gate=field_settings,continuity_guard=CONTINUITY_CONFIG,context_precomputed=True,temporal=True,
        coordinate_space='image pixels',boundary_shape='uncertainty ellipse',
        camera_compensation='background LK flow + RANSAC partial affine',
        homography_used_for_ball=False,homography_role='project static field geometry only',
        homography_sha256=digest(homography),new_detector_inference=False,
        parent_cache_sha256=digest(cache/'detections.jsonl'),
        code_sha256={n:digest(ROOT/'scripts'/n) for n in ['ellipse_field_gate.py','image_ball_gate.py','field_marking_evidence.py','ball_continuity.py']})
    out.mkdir(parents=True);atomic(out/'config.json',config)
    if recover:
        config['new_detector_inference']=True;atomic(out/'config.json',config)
    cap=cv2.VideoCapture(meta['source']);cap.set(cv2.CAP_PROP_POS_FRAMES,int(frames[0]))
    tracker=PixelGate(settings);static=PersistentMarkingEvidence(field_settings);result=[];previous=None;scale=480/meta['width']
    continuity=ContinuityGuard()
    started=time.monotonic()
    try:
        for i,row in enumerate(rows):
            ok,bgr=cap.read()
            if not ok:raise ValueError('Source decode failed')
            gray=cv2.cvtColor(cv2.resize(bgr,(480,round(meta['height']*scale))),cv2.COLOR_BGR2GRAY)
            cut=i==0 or segments[i]!=segments[i-1]
            affine,error=(None,None) if cut else background_motion(previous,gray,boxes[offsets[i-1]:offsets[i]],scale)
            previous=gray
            geometry=project_geometry(matrices.get(row['frame']))
            raw_candidates=row['candidates'];recovery_plans=[];new_points=[]
            frame_boxes=boxes[offsets[i]:offsets[i+1]]
            reset=cut or affine is None
            if recover:
                # Probe copies: one state transition per frame regardless of retry.
                trial_candidates,_=copy.deepcopy(static).update(raw_candidates,geometry,affine,frame_boxes,reset)
                trial_candidates,_=copy.deepcopy(continuity).filter(trial_candidates,affine,reset)
                trial=copy.deepcopy(tracker).update(trial_candidates,affine,error or 0.,reset)
                if trial['accepted'] is None:
                    center=velocity=None
                    if tracker.x is not None and not reset:
                        velocity=affine[:,:2]@tracker.x[2:]
                        center=affine[:,:2]@tracker.x[:2]+affine[:,2]+velocity
                    elif continuity.anchor is not None and not reset:
                        velocity=affine[:,:2]@continuity.velocity
                        center=affine[:,:2]@continuity.anchor+affine[:,2]+velocity*(continuity.elapsed+1)
                    recovery_plans=proposals(meta['width'],meta['height'],center,velocity,frame_boxes,row.get('guide'))
                    if recovery_plans:
                        new_points=detect_recovery(model,preprocess(bgr,'clahe'),recovery_plans,device).tolist()
                        raw_candidates=merge_points([np.asarray(raw_candidates).reshape(-1,3),np.asarray(new_points).reshape(-1,3)]).tolist()
            candidates,evidence=static.update(raw_candidates,geometry,affine,frame_boxes,reset)
            candidates,continuity_rejections=continuity.filter(candidates,affine,cut or affine is None)
            # A rejected marking must not remain alive as a Kalman prediction.
            # Compare in this frame after camera transport and velocity prediction.
            revoked=False
            if tracker.x is not None and affine is not None:
                predicted=affine[:,:2]@(tracker.x[:2]+tracker.x[2:])+affine[:,2]
                revoked=any(e['rejected'] and np.linalg.norm(np.asarray(p[:2])-predicted)<=FIELD_CONFIG['marking_distance_px']
                            for p,e in zip(raw_candidates,evidence))
                if revoked:tracker.reset()
            state=tracker.update(candidates,affine,error or 0.,cut or affine is None)
            continuity.observe(state['accepted'])
            polygon=None;axes=None
            if state['ellipse'] is not None:
                axes,vectors=state['ellipse'];theta=np.linspace(0,2*np.pi,49)
                polygon=(state['center']+(np.c_[np.cos(theta),np.sin(theta)]*axes)@vectors.T).tolist()
                axes=axes.tolist()
            def serial(x):return x.tolist() if x is not None else None
            result.append(dict(row,raw_selected=row['selected'],selected=serial(state['accepted']),
                candidates=raw_candidates,original_candidates=row['candidates'],recovery_crops=recovery_plans,
                recovery_candidates=new_points,
                recovery_accepted=bool(state['accepted'] is not None and len(new_points) and
                    min(np.linalg.norm(np.asarray(p[:2])-state['accepted'][:2]) for p in new_points)<1.),
                context_position=serial(state['position']),context_source=state['source'],
                forecast_3_frames=serial(state['forecast']),reachable_polygon=polygon,
                gate_track=state['track'],gate_rejected=state['rejected']+sum(r['rejected'] for r in evidence)+continuity_rejections,
                continuity_rejections=continuity_rejections,
                ellipse_axes_px=axes,marking_evidence=evidence,
                camera_affine=serial(affine),
                marking_lock_revoked=revoked,
                field_geometry=[dict(name=r['name'],points=r['points'].tolist()) for r in geometry],
                camera_motion_valid=affine is not None))
            if (i+1)%(100 if recover else 500)==0:
                atomic(out/'status.json',dict(state='running',completed=i+1,total=len(rows),elapsed_seconds=time.monotonic()-started))
                print(f'{i+1}/{len(rows)} ellipse/field frames ({time.monotonic()-started:.1f}s)',flush=True)
    finally:cap.release()
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    atomic(out/'status.json',dict(state='complete',completed=len(result),total=len(result)))
    summary=dict(frames=len(result),sources=dict(Counter(r['context_source'] for r in result)),
        field_rejections=sum(e['rejected'] for r in result for e in r['marking_evidence']),
        geometry_frames=sum(bool(r['field_geometry']) for r in result),
        gate_rejections=sum(r['gate_rejected'] for r in result),accuracy_measured=False)
    summary['marking_locks_revoked']=sum(r['marking_lock_revoked'] for r in result)
    summary['continuity_rejections']=sum(r['continuity_rejections'] for r in result)
    summary.update(recovery_frames=sum(bool(r['recovery_crops']) for r in result),
        recovery_crops=sum(len(r['recovery_crops']) for r in result),
        recovery_candidates=sum(len(r['recovery_candidates']) for r in result),
        recovery_accepted_frames=sum(r['recovery_accepted'] for r in result),elapsed_seconds=time.monotonic()-started)
    atomic(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=__import__('pathlib').Path,required=True);p.add_argument('--out',type=__import__('pathlib').Path,required=True)
    p.add_argument('--recover',action='store_true');p.add_argument('--device',default='mps')
    p.add_argument('--features',type=__import__('pathlib').Path)
    p.add_argument('--ignore-field-boundary',action='store_true',help='Keep penalty/interior marking checks while ignoring touchline and corner distractors')
    a=p.parse_args();run(a.cache,a.out,a.recover,a.device,a.features,a.ignore_field_boundary)
