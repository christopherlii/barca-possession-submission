"""Associate intermittent crop proposals and smooth one diagnostic video window.

All candidates are algorithmic proposals. No manually selected ball or label is
used. Camera transforms compensate pan/zoom, and missing positions stay missing
outside the observed endpoints. This never overwrites the main tracking cache.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
from stage1_cv_baseline import ROOT, private_path, digest
from smooth_pixel_track import smooth_rows

CONFIG=dict(max_gap_frames=5,max_speed_px_frame=45.,max_acceleration_px_frame2=6.,
            beam_width=8,min_observations=4,gap_cost=.3,acceleration_cost=.15,
            ambiguity_margin=1.5)


def associate(samples, config=None):
    """Second-order paths through candidate sets; empty frames do not reset them."""
    c=dict(CONFIG,**(config or {}));nodes=[]
    for t,points in enumerate(samples):
        for p in points:
            if len(p)==3 and np.isfinite(p).all():nodes.append((t,np.asarray(p[:2],float),float(p[2])))
    states=[];complete=[]
    for j,(t,p,confidence) in enumerate(nodes):
        choices=[(1.+confidence,None,0,[j])]
        for k in range(j-1,-1,-1):
            s,q,_=nodes[k];dt=t-s
            if dt>c['max_gap_frames']+1:break
            if dt<=0:continue
            velocity=(p-q)/dt
            if np.linalg.norm(velocity)>c['max_speed_px_frame']:continue
            for score,old,old_dt,path in states[k]:
                acceleration=0 if old is None else np.linalg.norm(velocity-old)/((dt+old_dt)/2)
                if acceleration>c['max_acceleration_px_frame2']:continue
                value=score+1.+confidence-c['gap_cost']*(dt-1)-c['acceleration_cost']*acceleration
                choices.append((value,velocity,dt,path+[j]))
        choices.sort(key=lambda s:s[0],reverse=True);states.append(choices[:c['beam_width']])
        complete.extend(s for s in states[-1] if len(s[3])>=c['min_observations'])
    complete.sort(key=lambda s:s[0],reverse=True)
    if not complete:return [],dict(status='insufficient_evidence',observations=0)
    best=complete[0];chosen={nodes[j][0]:nodes[j][1] for j in best[3]}
    # Competing spatial paths are more meaningful than variants of the same path.
    competitor=None
    for other in complete[1:]:
        comparison={nodes[j][0]:nodes[j][1] for j in other[3]}
        overlap=chosen.keys()&comparison.keys()
        if len(overlap)>=c['min_observations'] and np.median([np.linalg.norm(chosen[t]-comparison[t]) for t in overlap])>20:
            competitor=other[0];break
    ambiguous=competitor is not None and best[0]-competitor<c['ambiguity_margin']
    result=[(nodes[j][0],[*nodes[j][1],nodes[j][2]]) for j in best[3]]
    return result,dict(status='ambiguous' if ambiguous else 'associated',observations=len(result),
                       path_score=best[0],competing_score=competitor)


def process(data,camera_rows,mode):
    rows=data['rows'];poses=[];pose=np.eye(3);group=0;groups=[]
    for i,row in enumerate(rows):
        a=camera_rows[row['frame']].get('camera_affine')
        if i and (a is None or row['frame']!=rows[i-1]['frame']+1):group+=1;pose=np.eye(3)
        elif i:
            h=np.eye(3);h[:2]=a;pose=h@pose
        poses.append(pose.copy());groups.append(group)
    samples=[]
    for row,pose in zip(rows,poses):
        inv=np.linalg.inv(pose)
        samples.append([[*(inv@np.r_[p[:2],1])[:2],p[2]] for p in row['probes'][mode]['candidates']])
    selected={};reports=[]
    for g in sorted(set(groups)):
        indices=[i for i,v in enumerate(groups) if v==g]
        path,report=associate([samples[i] for i in indices]);reports.append(report)
        for j,p in path:
            i=indices[j];selected[i]=[*(poses[i]@np.r_[p[:2],1])[:2],p[2]]
    work=[]
    for i,row in enumerate(rows):
        p=selected.get(i)
        work.append(dict(frame=row['frame'],selected=p,context_position=p[:2] if p else None,
                         context_source='gated_detection' if p else 'missing',gate_track=groups[i],
                         camera_affine=camera_rows[row['frame']].get('camera_affine')))
    smoothed=smooth_rows(work,dict(max_gap_frames=CONFIG['max_gap_frames'],max_adjustment_px=15.))
    for row,result in zip(rows,smoothed):
        row.setdefault('window_smoothing',{})[mode]=dict(position=result['context_position'],
            source=result['context_source'],selected=result['selected'],
            camera_group=result['gate_track'])
    for i,row in enumerate(rows):row['window_smoothing'][mode]['camera_pose']=poses[i].tolist()
    return dict(mode=mode,frames=len(rows),candidate_frames=sum(bool(s) for s in samples),
        associated_frames=len(selected),smoothed_position_frames=sum(r['context_position'] is not None for r in smoothed),
        interpolated_frames=sum(r['context_source']=='smoothed_gap' for r in smoothed),groups=reports)


def run(args):
    source=private_path(args.probe);cache=private_path(args.cache);out=private_path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    data=json.loads((source/'result.json').read_text());camera_rows={r['frame']:r for r in map(json.loads,(cache/'detections.jsonl').read_text().splitlines())}
    reports=[process(data,camera_rows,mode) for mode in ['raw_full_1280','clahe_guided_640','fixed_raw_1280','fixed_clahe_1280']]
    data['smoothing_config']=CONFIG;data['smoothing_reports']=reports
    data['smoothing_note']='Offline association + camera-compensated RTS; all raw candidates retained; no manual ball labels or global tracking changes.'
    out.mkdir(parents=True)
    for image in source.glob('*.jpg'):os.link(image,out/image.name)
    (out/'result.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    (out/'index.html').write_bytes((ROOT/'scripts/neighbor_frames_view.html').read_bytes())
    (out/'summary.json').write_text(json.dumps(dict(reports=reports,config=CONFIG,accuracy_measured=False,
        source_sha256=digest(source/'result.json'),code_sha256=digest(__file__)),indent=2))
    print(json.dumps(reports))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
