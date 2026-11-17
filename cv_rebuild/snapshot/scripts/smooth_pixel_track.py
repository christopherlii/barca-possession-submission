"""Offline RTS smoothing of accepted pixel observations; no labels or new detections.

Camera transforms transport observations into a common image coordinate system.
Only short, bracketed gaps within the same track are filled. Raw observations,
online forecasts and uncertainty boundaries are preserved separately.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from stage1_cv_baseline import private_path, digest
from local_cv_possession import atomic

CONFIG=dict(max_gap_frames=12,measurement_sigma_px=5.,process_sigma_px=2.,
            max_adjustment_px=15.,max_speed_px_frame=80.)


def smooth_rows(rows, config=None):
    c=dict(CONFIG,**(config or {}));out=[dict(r) for r in rows]
    f=np.eye(4);f[0,2]=f[1,3]=1
    groups=[];group=[];pose=np.eye(3);previous=None
    for i,row in enumerate(rows):
        a=row.get('camera_affine')
        broken=(previous is None or a is None or row['frame']!=previous['frame']+1
                or row.get('gate_track')!=previous.get('gate_track'))
        if broken:
            if group:groups.append(group)
            group=[];pose=np.eye(3)
        else:
            matrix=np.eye(3);matrix[:2]=a;pose=matrix@pose
        group.append((i,pose.copy()));previous=row
        out[i]['online_context_position']=row.get('context_position')
        out[i]['online_context_source']=row.get('context_source')
    if group:groups.append(group)
    for group in groups:
        observations=[j for j,(i,_) in enumerate(group) if rows[i].get('selected') is not None]
        chains=[];chain=[]
        for j in observations:
            if chain and j-chain[-1]-1>c['max_gap_frames']:
                chains.append(chain);chain=[]
            chain.append(j)
        if chain:chains.append(chain)
        for chain in chains:
            if len(chain)<3:continue
            samples=group[chain[0]:chain[-1]+1];zs=[];rs=[]
            for i,pose in samples:
                inv=np.linalg.inv(pose);selected=rows[i].get('selected')
                zs.append((inv@np.r_[selected[:2],1])[:2] if selected is not None else None)
                rs.append(c['measurement_sigma_px']**2*inv[:2,:2]@inv[:2,:2].T)
            # Endpoint initialization with measured average velocity.
            second=next(j for j in range(1,len(zs)) if zs[j] is not None)
            x=np.r_[zs[0],(zs[second]-zs[0])/second];p=np.diag([25.,25.,100.,100.])
            filtered=[];covs=[];predicted=[];predcov=[]
            for j,z in enumerate(zs):
                inv=np.linalg.inv(samples[j][1]);scale=float(np.linalg.norm(inv[:2,:2],ord=2))
                q=(c['process_sigma_px']*scale)**2*np.diag([.25,.25,1.,1.])
                xp=f@x if j else x.copy();pp=f@p@f.T+q if j else p.copy()
                predicted.append(xp.copy());predcov.append(pp.copy())
                if z is not None:
                    k=np.linalg.solve(pp[:2,:2]+rs[j],pp[:2,:]).T
                    x=xp+k@(z-xp[:2]);a=np.eye(4);a[:,:2]-=k
                    p=a@pp@a.T+k@rs[j]@k.T
                else:x,p=xp,pp
                filtered.append(x.copy());covs.append(p.copy())
            smooth=[x.copy() for x in filtered]
            for j in range(len(smooth)-2,-1,-1):
                gain=np.linalg.solve(predcov[j+1],f@covs[j]).T
                smooth[j]=filtered[j]+gain@(smooth[j+1]-predicted[j+1])
            for j,(i,pose) in enumerate(samples):
                point=(pose@np.r_[smooth[j][:2],1])[:2]
                raw=rows[i].get('selected')
                if raw is not None:
                    delta=point-np.asarray(raw[:2]);distance=np.linalg.norm(delta)
                    if distance>c['max_adjustment_px']:point=np.asarray(raw[:2])+delta*c['max_adjustment_px']/distance
                    source='smoothed_detection'
                else:
                    # Never fill an interval whose endpoint displacement implies
                    # implausible motion in the camera-compensated coordinates.
                    left=max(k for k in range(j) if zs[k] is not None)
                    right=next(k for k in range(j+1,len(zs)) if zs[k] is not None)
                    speed=np.linalg.norm(pose[:2,:2]@(zs[right]-zs[left]))/(right-left)
                    if speed>c['max_speed_px_frame']:continue
                    source='smoothed_gap'
                out[i].update(context_position=point.tolist(),context_source=source)
    return out


def run(cache,out):
    cache,out=private_path(cache),private_path(out)
    if out.exists():raise ValueError('Use a new output directory')
    rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    result=smooth_rows(rows);config=json.loads((cache/'config.json').read_text())
    config.update(offline_smoothing=CONFIG,smoothing_uses_future=True,
                  smoothing_code_sha256=digest(__file__),smoothing_parent_sha256=digest(cache/'detections.jsonl'))
    out.mkdir(parents=True);atomic(out/'config.json',config)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    summary=dict(frames=len(result),sources=dict(Counter(r['context_source'] for r in result)),
                 added_visible_frames=sum(a.get('context_position') is None and b.get('context_position') is not None for a,b in zip(rows,result)),accuracy_measured=False)
    atomic(out/'summary.json',summary);atomic(out/'status.json',dict(state='complete',completed=len(result),total=len(result)))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.cache,a.out)
