"""Experimental SAM retry cascade: moving tight crops, boxes, multiple seeds.

No manual prompts or supplied-match labels. Seed and propagation failures have
explicit diagnostics. Recovered positions require independent direction or CV
support. Previous results are preserved when --baseline is provided.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from stage1_cv_baseline import ROOT,private_path,digest
from local_cv_possession import atomic
from sam2_gap_recovery import observation,gap_jobs,transport,runtime,mask_record,agreeing,CONFIG as BASE

CONFIG=dict(BASE,anchor_lookaround=8,max_anchors=3,crop_sizes=[192,96],
            prompt_modes=['box','point'],box_radius=7.,max_direction_frames=32,
            max_step_px=60.,max_seed_velocity=45.)


def mask_diagnostics(logits,origin,seed=None,config=None):
    """Explain rejection without silently relaxing the accepted geometry."""
    c=dict(CONFIG,**(config or {}));mask=np.squeeze(np.asarray(logits))>0
    if mask.ndim!=2:return None,dict(reason='wrong_shape')
    y,x=np.nonzero(mask);area=len(x)
    stats=dict(area=area)
    if area<c['min_area_px']:return None,dict(stats,reason='empty_or_tiny')
    w=int(np.ptp(x)+1);h=int(np.ptp(y)+1)
    stats.update(width=w,height=h,fill=float(area/(w*h)))
    if area>c['max_area_px']:return None,dict(stats,reason='oversized')
    if max(w/h,h/w)>c['max_aspect']:return None,dict(stats,reason='elongated')
    if area/(w*h)<c['min_fill']:return None,dict(stats,reason='fragmented')
    record=mask_record(logits,origin,c)
    if record is None:return None,dict(stats,reason='touches_crop_edge')
    if seed is not None:
        distance=float(np.linalg.norm(np.asarray(record['point'])-seed));stats['seed_distance']=distance
        if distance>c['seed_radius_px']:return None,dict(stats,reason='off_prompt')
    return record,dict(stats,reason='valid')


def anchors_for(rows,job,reverse,config=None):
    c=dict(CONFIG,**(config or {}));boundary=job['right'] if reverse else job['left']
    # Also recover when the immediate endpoint was weak but a nearby one is good.
    if boundary is None:boundary=job['hi']+1 if reverse else job['lo']-1
    indices=[]
    for delta in range(c['anchor_lookaround']+1):
        i=boundary+delta if reverse else boundary-delta
        if not 0<=i<len(rows):break
        edge=i if reverse else i+1
        if delta and (rows[edge].get('camera_affine') is None or rows[edge]['frame']!=rows[edge-1]['frame']+1):break
        p=observation(rows[i])
        if p is not None and p[2]>=c['minimum_seed_confidence']:indices.append(i)
    if not indices:return []
    nearest=indices[0]
    others=sorted(indices[1:],key=lambda i:(-observation(rows[i])[2],abs(i-nearest)))
    return [nearest,*others[:c['max_anchors']-1]]


def seed_velocity(rows,anchor,reverse,config=None):
    c=dict(CONFIG,**(config or {}));values=[];seed=observation(rows[anchor])[:2]
    for dt in range(1,5):
        j=anchor+dt if reverse else anchor-dt
        if not 0<=j<len(rows):break
        lo,hi=sorted([j,anchor])
        if any(rows[k].get('camera_affine') is None for k in range(lo+1,hi+1)):break
        p=observation(rows[j])
        if p is not None:
            q=transport(p[:2],rows,j,anchor);values.append((q-seed)/(j-anchor))
    v=np.median(values,axis=0) if values else np.zeros(2)
    speed=np.linalg.norm(v)
    return v*min(1.,c['max_seed_velocity']/max(speed,1e-9))


def crop_origins(rows,anchor,lo,hi,width,height,size,reverse):
    v=seed_velocity(rows,anchor,reverse);p=observation(rows[anchor])[:2];origins=[]
    for i in range(lo,hi+1):
        center=transport(p+v*(i-anchor),rows,anchor,i)
        x=int(np.clip(round(center[0]-size/2),0,max(0,width-size)));y=int(np.clip(round(center[1]-size/2),0,max(0,height-size)))
        origins.append([x,y])
    return origins


def window_for(rows,job,anchor,reverse,config=None):
    c=dict(CONFIG,**(config or {}));edge=anchor;limit=job['lo'] if reverse else job['hi']
    for _ in range(c['max_direction_frames']):
        j=edge-1 if reverse else edge+1
        if not 0<=j<len(rows) or (reverse and j<limit) or (not reverse and j>limit):break
        k=max(j,edge)
        if rows[k].get('camera_affine') is None or rows[k]['frame']!=rows[k-1]['frame']+1:break
        edge=j
    if edge==anchor:return None
    return min(anchor,edge),max(anchor,edge)


def attempt_direction(predictor,rows,job,reverse,cap,temp_root,config=None):
    import cv2,torch
    c=dict(CONFIG,**(config or {}));best={};attempts=[]
    anchors=anchors_for(rows,job,reverse,c)
    for anchor in anchors:
        window=window_for(rows,job,anchor,reverse,c)
        if window is None:continue
        lo,hi=window;images=[];cap.set(cv2.CAP_PROP_POS_FRAMES,rows[lo]['frame'])
        for _ in range(lo,hi+1):
            ok,im=cap.read()
            if not ok:raise ValueError('Video decode failed')
            images.append(im)
        h,w=images[0].shape[:2]
        for size in c['crop_sizes']:
            origins=crop_origins(rows,anchor,lo,hi,w,h,size,reverse)
            with tempfile.TemporaryDirectory(dir=temp_root,prefix='sam-retry-') as td:
                for i,(im,(x,y)) in enumerate(zip(images,origins)):
                    if not cv2.imwrite(str(Path(td)/f'{i:05d}.jpg'),im[y:y+size,x:x+size],[cv2.IMWRITE_JPEG_QUALITY,98]):raise ValueError('Crop write failed')
                with torch.inference_mode():
                    state=predictor.init_state(td,offload_video_to_cpu=True,offload_state_to_cpu=True)
                    for mode in c['prompt_modes']:
                        predictor.reset_state(state)
                        seed_index=anchor-lo;origin=origins[seed_index];seed=observation(rows[anchor])[:2];local=seed-origin
                        options=dict(frame_idx=seed_index,obj_id=1,points=np.asarray([local],np.float32),labels=np.array([1],np.int32))
                        if mode=='box':options['box']=np.r_[np.maximum(0,local-c['box_radius']),np.minimum([size-1,size-1],local+c['box_radius'])].astype(np.float32)
                        _,_,logits=predictor.add_new_points_or_box(state,**options)
                        record,diag=mask_diagnostics(logits[0].cpu().numpy(),origin,seed,c)
                        note=dict(anchor_frame=rows[anchor]['frame'],reverse=reverse,crop_size=size,prompt=mode,seed=diag)
                        attempts.append(note)
                        if record is None:continue
                        result={};last_index=anchor;last_point=seed
                        for frame,_,masks in predictor.propagate_in_video(state,start_frame_idx=seed_index,max_frame_num_to_track=hi-lo,reverse=reverse):
                            i=lo+frame;r,why=mask_diagnostics(masks[0].cpu().numpy(),origins[frame],config=c)
                            if r is None:note['stop']=why['reason'];break
                            ratio=r['area']/record['area']
                            if not c['area_ratio_min']<=ratio<=c['area_ratio_max']:note['stop']='area_change';break
                            expected=transport(last_point,rows,last_index,i)
                            if np.linalg.norm(np.asarray(r['point'])-expected)>c['max_step_px']*max(1,abs(i-last_index)):note['stop']='motion_jump';break
                            last_point=np.asarray(r['point']);last_index=i
                            if job['lo']<=i<=job['hi']:result[i]=r
                        note['propagated_gap_frames']=len(result)
                        if len(result)>len(best):best=result
                        # Stop if this direction reached its furthest planned gap frame.
                        required=sum(job['lo']<=i<=job['hi'] for i in range(lo,hi+1))
                        if required and len(best)==required:return best,attempts
                    del state
            gc.collect()
    return best,attempts


def run(args):
    import cv2,torch
    cache,out=private_path(args.cache),private_path(args.out)
    rows=list(map(json.loads,(cache/'detections.jsonl').read_text().splitlines()));config=json.loads((cache/'config.json').read_text())
    baseline=rows if args.baseline is None else list(map(json.loads,(private_path(args.baseline)/'detections.jsonl').read_text().splitlines()))
    if [r['frame'] for r in rows]!=[r['frame'] for r in baseline]:raise ValueError('Baseline frame alignment mismatch')
    jobs=gap_jobs(rows);jobs=jobs if args.max_gaps is None else jobs[:args.max_gaps]
    signature=dict(source_sha256=digest(cache/'detections.jsonl'),code_sha256=digest(__file__),config=CONFIG,max_gaps=args.max_gaps)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=signature:raise ValueError('Resume plan mismatch')
    atomic(out/'plan.json',signature)
    done={}
    if (out/'jobs.jsonl').exists():done={r['job_index']:r for r in map(json.loads,(out/'jobs.jsonl').read_text().splitlines())}
    predictor=runtime(args.device);cap=cv2.VideoCapture(config['source']);cv2.setNumThreads(2);started=time.monotonic()
    try:
        with (out/'jobs.jsonl').open('a') as f:
            for n,job in enumerate(jobs):
                if n in done:continue
                forward,fa=attempt_direction(predictor,rows,job,False,cap,out)
                backward,ba=attempt_direction(predictor,rows,job,True,cap,out)
                additions={};proposals={}
                for i in forward.keys()|backward.keys():
                    records=[d[i] for d in [forward,backward] if i in d];proposals[str(i)]=records
                    if len(records)==2 and agreeing(*records):
                        additions[str(i)]=dict(position=((np.asarray(records[0]['point'])+records[1]['point'])/2).tolist(),source='sam2_retry_bidirectional',mask=records[0])
                    elif len(records)==1:
                        r=records[0];candidates=[p for key in ['candidates','player_context_candidates','crowded_zoom_candidates'] for p in (rows[i].get(key) or [])]
                        if any(p[2]>=CONFIG['minimum_seed_confidence'] and np.linalg.norm(np.asarray(p[:2])-r['point'])<=CONFIG['candidate_agreement_px'] for p in candidates):
                            additions[str(i)]=dict(position=r['point'],source='sam2_retry_cv_supported',mask=r)
                record=dict(job_index=n,attempts=fa+ba,additions=additions,proposals=proposals)
                f.write(json.dumps(record)+'\n');f.flush();done[n]=record
                atomic(out/'status.json',dict(state='running',completed=len(done),total=len(jobs),elapsed_seconds=time.monotonic()-started))
                print(f'SAM retries {len(done)}/{len(jobs)} gaps; {sum(len(x["additions"]) for x in done.values())} supported; {time.monotonic()-started:.1f}s',flush=True)
                gc.collect()
                if args.device=='mps':torch.mps.empty_cache()
    finally:cap.release()
    result=copy.deepcopy(baseline)
    for job in done.values():
        for key,p in job['proposals'].items():result[int(key)]['sam2_retry_proposals']=p
        for key,a in job['additions'].items():
            i=int(key)
            if observation(rows[i]) is not None or baseline[i].get('sam2_mask') is not None:continue
            result[i].update(context_position=a['position'],context_source=a['source'],sam2_mask=a['mask'])
    from collections import Counter
    reasons=Counter(a['seed']['reason'] for j in done.values() for a in j['attempts'])
    summary=dict(frames=len(rows),gaps=len(jobs),attempts=sum(len(j['attempts']) for j in done.values()),seed_outcomes=dict(reasons),
        supported_frames=sum(r.get('sam2_mask') is not None for r in result),new_over_baseline=sum(a.get('sam2_mask') is None and b.get('sam2_mask') is not None for a,b in zip(baseline,result)),
        newly_filled=sum(a.get('context_position') is None and b.get('context_position') is not None for a,b in zip(baseline,result)),elapsed_seconds=time.monotonic()-started,accuracy_measured=False)
    config['sam2_gap_recovery']=dict(signature,variant='multi-seed tight moving crops and box prompts',baseline_preserved=True)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result));atomic(out/'summary.json',summary);atomic(out/'config.json',config);atomic(out/'status.json',dict(state='complete',**summary));print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='mps');p.add_argument('--max-gaps',type=int);run(p.parse_args())
