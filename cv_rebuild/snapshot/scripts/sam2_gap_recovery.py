"""Stage 12: bounded, automatic SAM 2.1 gap proposals and validated recovery.

Prompts come only from prior direct detector measurements. Two independent
video states propagate from opposite endpoints; agreement is required before
adding a location. Long gaps receive bounded one-sided searches, retained only
as proposals unless corroborated by a direct detector candidate. Existing
observed locations are immutable. Estimates and SAM tracking are distinct.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from stage1_cv_baseline import ROOT, private_path, digest
from local_cv_possession import atomic

CONFIG=dict(max_direction_frames=24, minimum_seed_confidence=.15,
    crop_padding_px=96, max_crop_side=640, seed_radius_px=8.,
    min_area_px=2,max_area_px=400,max_aspect=4.,min_fill=.15,
    area_ratio_min=.2,area_ratio_max=5.,max_speed_px_frame=60.,
    agreement_px=12.,agreement_iou=.2,candidate_agreement_px=8.)
SOURCE_REVISION='2b90b9f5ceec907a1c18123530e92e794ad901a4'


def observation(row):
    p=row.get('short_gap_detection') or row.get('selected')
    if p is None and row.get('aerial_detection') is not None:p=row['aerial_detection']
    if p is None:return None
    p=np.asarray(p,float)
    return p if p.shape==(3,) and np.isfinite(p).all() else None


def gap_jobs(rows,config=None):
    """Cover each missing-detection run from every trustworthy available end."""
    c=dict(CONFIG,**(config or {}));jobs=[];i=0
    while i<len(rows):
        if observation(rows[i]) is not None:i+=1;continue
        end=i
        while end<len(rows) and observation(rows[end]) is None:end+=1
        left=i-1 if i else None;right=end if end<len(rows) else None
        trusted=lambda j:j is not None and observation(rows[j])[2]>=c['minimum_seed_confidence']
        left=left if trusted(left) else None;right=right if trusted(right) else None
        # Reset at cuts, failed camera transport, and nonconsecutive video IDs.
        windows=[]
        for anchor,reverse in [(left,False),(right,True)]:
            if anchor is None:continue
            edge=anchor;limit=i if reverse else end-1
            for _ in range(c['max_direction_frames']):
                nxt=edge-1 if reverse else edge+1
                if (reverse and nxt<limit) or (not reverse and nxt>limit):break
                e=max(edge,nxt)
                if rows[e]['frame']!=rows[e-1]['frame']+1 or rows[e].get('camera_affine') is None:break
                edge=nxt
            if edge!=anchor:windows.append(dict(anchor=anchor,lo=min(anchor,edge),hi=max(anchor,edge),reverse=reverse))
        jobs.append(dict(lo=i,hi=end-1,left=left,right=right,windows=windows))
        i=end
    return jobs


def mask_record(logits,origin,config=None):
    c=dict(CONFIG,**(config or {}));mask=np.asarray(logits)>0
    mask=np.squeeze(mask)
    if mask.ndim!=2:return None
    y,x=np.nonzero(mask);area=len(x)
    if not c['min_area_px']<=area<=c['max_area_px']:return None
    w=int(x.max()-x.min()+1);h=int(y.max()-y.min()+1)
    if max(w/h,h/w)>c['max_aspect'] or area/(w*h)<c['min_fill']:return None
    if x.min()==0 or y.min()==0 or x.max()==mask.shape[1]-1 or y.max()==mask.shape[0]-1:return None
    point=np.array([x.mean()+origin[0],y.mean()+origin[1]])
    # Store sparse positive pixels rather than float logits / whole-frame masks.
    return dict(point=point.tolist(),area=area,box=[int(x.min()+origin[0]),int(y.min()+origin[1]),int(x.max()+origin[0]+1),int(y.max()+origin[1]+1)],
        pixels=np.c_[x+origin[0],y+origin[1]].astype(int).tolist())


def agreeing(a,b,config=None):
    c=dict(CONFIG,**(config or {}))
    if not a or not b:return False
    if np.linalg.norm(np.asarray(a['point'])-b['point'])>c['agreement_px']:return False
    x={tuple(p) for p in a['pixels']};y={tuple(p) for p in b['pixels']}
    return len(x&y)/max(1,len(x|y))>=c['agreement_iou']


def transport(point,rows,anchor,target):
    p=np.asarray(point,float)
    if target>anchor:
        for i in range(anchor+1,target+1):
            a=np.asarray(rows[i]['camera_affine']);p=a[:,:2]@p+a[:,2]
    else:
        for i in range(anchor,target,-1):
            a=np.eye(3);a[:2]=rows[i]['camera_affine'];p=(np.linalg.inv(a)@np.r_[p,1])[:2]
    return p


def supported(record,seed_record,rows,anchor,index,config=None):
    c=dict(CONFIG,**(config or {}))
    if record is None or seed_record is None:return False
    ratio=record['area']/seed_record['area']
    if not c['area_ratio_min']<=ratio<=c['area_ratio_max']:return False
    seed=observation(rows[anchor])
    expected=transport(seed[:2],rows,anchor,index)
    return np.linalg.norm(np.asarray(record['point'])-expected)<=12+c['max_speed_px_frame']*abs(index-anchor)


def runtime(device):
    sys.path.extend([str(ROOT/'outputs/sam2-runtime'),str(ROOT/'outputs/sam2-source')])
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    source=ROOT/'outputs/sam2-source'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if revision!=SOURCE_REVISION:raise ValueError('SAM source revision differs from pinned version')
    torch.set_num_threads(2)
    model=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(ROOT/'outputs/sam2-models/sam2.1_hiera_tiny.pt'),device=device,apply_postprocessing=False)
    return model


def propagate(predictor,rows,window,cap,temp_root,config=None):
    import cv2,torch
    c=dict(CONFIG,**(config or {}));lo,hi,anchor=window['lo'],window['hi'],window['anchor']
    seeds=[transport(observation(rows[anchor])[:2],rows,anchor,j) for j in range(lo,hi+1)]
    for j in range(lo,hi+1):
        guide=rows[j].get('guide')
        if guide is not None and np.isfinite(guide).all():seeds.append(np.asarray(guide))
    p=np.asarray(seeds);lower=p.min(0)-c['crop_padding_px'];upper=p.max(0)+c['crop_padding_px']
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center=(lower+upper)/2;size=np.minimum(upper-lower,c['max_crop_side']);lower=np.maximum(0,np.minimum(center-size/2,[width,height]-size));upper=np.minimum([width,height],lower+size)
    x,y=np.floor(lower).astype(int);xx,yy=np.ceil(upper).astype(int)
    seed=observation(rows[anchor])[:2]-[x,y]
    if not(0<seed[0]<xx-x and 0<seed[1]<yy-y):return {},dict(reason='seed_outside_crop')
    with tempfile.TemporaryDirectory(dir=temp_root,prefix='sam-window-') as td:
        cap.set(cv2.CAP_PROP_POS_FRAMES,rows[lo]['frame'])
        for i in range(lo,hi+1):
            ok,image=cap.read()
            if not ok:raise ValueError('Video decode failed')
            if not cv2.imwrite(str(Path(td)/f'{i-lo:05d}.jpg'),image[y:yy,x:xx],[cv2.IMWRITE_JPEG_QUALITY,98]):raise ValueError('Crop save failed')
        with torch.inference_mode():
            state=predictor.init_state(td,offload_video_to_cpu=True,offload_state_to_cpu=True)
            seed_index=anchor-lo
            _,_,logits=predictor.add_new_points_or_box(state,frame_idx=seed_index,obj_id=1,points=np.asarray([seed],np.float32),labels=np.array([1],np.int32))
            record=mask_record(logits[0].cpu().numpy(),[x,y],c)
            if record is None:return {},dict(reason='seed_mask_invalid',crop=[int(x),int(y),int(xx),int(yy)])
            if np.linalg.norm(np.asarray(record['point'])-observation(rows[anchor])[:2])>c['seed_radius_px']:return {},dict(reason='seed_mask_off_center')
            results={}
            for frame,_,masks in predictor.propagate_in_video(state,start_frame_idx=seed_index,max_frame_num_to_track=hi-lo,reverse=window['reverse']):
                i=lo+frame
                candidate=mask_record(masks[0].cpu().numpy(),[x,y],c)
                if supported(candidate,record,rows,anchor,i,c):results[i]=candidate
                elif i!=anchor:break  # do not reconnect after drifting/loss
            del state
        return results,dict(reason='propagated',crop=[int(x),int(y),int(xx),int(yy)],seed_area=record['area'])


def run(args):
    import cv2
    import torch
    cache,out=private_path(args.cache),private_path(args.out)
    rows=list(map(json.loads,(cache/'detections.jsonl').read_text().splitlines()));config=json.loads((cache/'config.json').read_text())
    all_jobs=gap_jobs(rows);jobs=all_jobs if args.max_gaps is None else all_jobs[:args.max_gaps]
    out.mkdir(parents=True,exist_ok=True)
    weights=ROOT/'outputs/sam2-models/sam2.1_hiera_tiny.pt'
    signature=dict(source_sha256=digest(cache/'detections.jsonl'),code_sha256=digest(__file__),model_sha256=digest(weights),config=CONFIG,max_gaps=args.max_gaps)
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=signature:raise ValueError('Resume plan mismatch')
    atomic(out/'plan.json',signature)
    completed={}
    if (out/'jobs.jsonl').exists():completed={j['job_index']:j for j in map(json.loads,(out/'jobs.jsonl').read_text().splitlines())}
    predictor=runtime(args.device);cap=cv2.VideoCapture(config['source']);cv2.setNumThreads(2);start=time.monotonic()
    try:
        with (out/'jobs.jsonl').open('a') as handle:
            for number,job in enumerate(jobs):
                if number in completed:continue
                directions=[];notes=[]
                for window in job['windows']:
                    results,note=propagate(predictor,rows,window,cap,out)
                    directions.append((window,results));notes.append(note)
                additions={};proposals={}
                for window,results in directions:
                    for i,r in results.items():
                        if job['lo']<=i<=job['hi']:proposals.setdefault(i,[]).append(r)
                for i,records in proposals.items():
                    if len(records)==2 and agreeing(*records):
                        additions[i]=dict(position=((np.asarray(records[0]['point'])+records[1]['point'])/2).tolist(),source='sam2_bidirectional',mask=records[0])
                    elif len(records)==1:
                        r=records[0];candidates=rows[i].get('candidates') or []
                        match=any(np.linalg.norm(np.asarray(p[:2])-r['point'])<=CONFIG['candidate_agreement_px'] and p[2]>=CONFIG['minimum_seed_confidence'] for p in candidates)
                        if match:additions[i]=dict(position=r['point'],source='sam2_cv_supported',mask=r)
                record=dict(job_index=number,gap=[rows[job['lo']]['frame'],rows[job['hi']]['frame']],notes=notes,
                    proposals={str(i):r for i,r in proposals.items()},additions={str(i):r for i,r in additions.items()})
                handle.write(json.dumps(record)+'\n');handle.flush();completed[number]=record
                atomic(out/'status.json',dict(state='running',completed=len(completed),total=len(jobs),all_gaps=len(all_jobs),elapsed_seconds=time.monotonic()-start))
                print(f'SAM {len(completed)}/{len(jobs)} gaps; {sum(len(r["additions"]) for r in completed.values())} supported positions; {time.monotonic()-start:.1f}s',flush=True)
                gc.collect()
                if args.device=='mps':torch.mps.empty_cache()
    finally:cap.release()
    result=copy.deepcopy(rows)
    for job in completed.values():
        for key,records in job['proposals'].items():result[int(key)]['sam2_proposals']=records
        for key,a in job['additions'].items():
            i=int(key)
            if observation(rows[i]) is not None:continue
            result[i].update(context_position=a['position'],context_source=a['source'],sam2_mask=a['mask'],sam2_gap=job['gap'])
    config['sam2_gap_recovery']=dict(signature,source_revision=SOURCE_REVISION,device=args.device,all_gap_count=len(all_jobs),processed_gaps=len(jobs),bounded_direction_frames=CONFIG['max_direction_frames'])
    summary=dict(frames=len(rows),all_gaps=len(all_jobs),processed_gaps=len(completed),proposal_frames=sum(bool(r.get('sam2_proposals')) for r in result),
        sam_supported_frames=sum(r.get('sam2_mask') is not None for r in result),new_position_frames=sum(a.get('context_position') is None and b.get('context_position') is not None for a,b in zip(rows,result)),
        elapsed_seconds=time.monotonic()-start,accuracy_measured=False)
    atomic(out/'config.json',config);atomic(out/'summary.json',summary)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result));atomic(out/'status.json',dict(state='complete',**summary));print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='mps');p.add_argument('--max-gaps',type=int,help='Explicit diagnostic limit; default attempts every gap');run(p.parse_args())
