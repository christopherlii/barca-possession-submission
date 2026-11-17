"""Opt-in confidence/ground-uncertainty adapter; never runs CV or reads answers.

Emits a compatible possession manifest and a separate quality NPZ. Detector
scores and verification priors are explicitly uncalibrated. Unknown airborne
state remains NaN. The selected policy either requires short baseline-anchor
support or admits confidence-qualified uncertain projections without that veto.
"""
import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
from cv_to_possession import project_recovery
from games import asset_paths
from possession_observation_quality import select_observations,QualityConfig
from stage1_cv_baseline import ROOT,digest,private_path


def quality_from_rows(rows,observed,homography,pitch):
    matrices={}
    ids={r['frame'] for r,seen in zip(rows,observed) if seen}
    with Path(homography).open() as stream:
        for r in csv.DictReader(stream):
            if int(r['frame']) in ids:
                matrices[int(r['frame'])]=np.linalg.inv(np.array([[float(r[f'h{i}{j}']) for j in range(3)] for i in range(3)]))
    n=len(rows);confidence=np.full(n,np.nan);sigma=np.full(n,np.nan);airborne=np.full(n,np.nan)
    kind=np.full(n,'unknown',dtype='<U40')
    for i,row in enumerate(rows):
        if not observed[i]:continue
        point=np.asarray(row['evidence_position'],float)
        candidates=[]
        for key in ('selected','raw_selected','blank_recovery_detection','short_gap_detection','bidirectional_detection'):
            p=row.get(key)
            if p is not None and len(p)==3 and np.isfinite(p).all() and np.linalg.norm(np.asarray(p[:2])-point)<=8.:
                candidates.append(float(p[2]))
        if candidates:
            confidence[i]=np.clip(max(candidates),0,1);kind[i]='raw_detector_score_uncalibrated'
        elif row.get('sam2_mask') and row.get('evidence_source') in ('sam2_extended_verified','sam2_reacquired'):
            confidence[i]=.5;kind[i]='mask_verification_prior'
        # Propagate a declared 2-pixel localization scale through the inverse
        # homography. This does NOT include unknown ball height uncertainty.
        h=matrices[row['frame']]
        def project(p):
            v=h@np.r_[p,1.]
            if abs(v[2])<1e-12:return np.full(2,np.nan)
            return v[:2]/v[2]*pitch
        jacobian=np.column_stack([(project(point+axis)-project(point-axis))/2 for axis in np.eye(2)])
        sigma[i]=2.*np.linalg.svd(jacobian,compute_uv=False)[0] if np.isfinite(jacobian).all() else np.inf
        if row.get('airborne_risk') is not None:
            airborne[i]=float(row['airborne_risk'])
        elif str(row.get('evidence_source','')).startswith('aerial_'):
            airborne[i]=1.  # Conservative declared aerial-source flag, not inferred height.
    return confidence,sigma,airborne,kind


def run(manifest_path,features_path,cache,out,unknown_airborne_policy='require_anchors'):
    manifest_path=Path(manifest_path);features_path=Path(features_path);cache=Path(cache)
    out=private_path(out)
    if out.exists():raise ValueError('Use a new private output directory')
    m=json.loads(manifest_path.read_text());paths=asset_paths(data_dir=str(ROOT/'data'),source='tactical')
    if m.get('training_eligible') is not False or m.get('camera')!='tactical':raise ValueError('Expected tactical inference-only manifest')
    expected={'features_sha256':features_path,'recovery_sha256':cache/'detections.jsonl',
        'recovery_config_sha256':cache/'config.json','homography_sha256':Path(paths['homography'])}
    if any(digest(p)!=m['provenance'][key] for key,p in expected.items()):raise ValueError('CV provenance differs')
    pins={str(p):digest(p) for p in [manifest_path,manifest_path.parent/m['files']['observations'],
        manifest_path.parent/m['files']['tagging'],*expected.values()]}
    with np.load(manifest_path.parent/m['files']['observations'],allow_pickle=False) as data:
        original={k:data[k].copy() for k in data.files}
    with np.load(features_path,allow_pickle=False) as data:
        baseline=data['atd_ball_m'].copy()
        if not np.array_equal(data['frame_ids'],original['frame_ids']):raise ValueError('Baseline timeline differs')
    rows=[json.loads(s) for s in (cache/'detections.jsonl').read_text().splitlines()]
    reconstructed,sources=project_recovery(rows,original['frame_ids'],paths['homography'],np.asarray(m['pitch']))
    if not np.array_equal(reconstructed,original['ball'],equal_nan=True) or not np.array_equal(sources,original['ball_source']):
        raise ValueError('Accepted CV observations do not reproduce')
    confidence,sigma,airborne,kind=quality_from_rows(rows,original['ball_observed'],paths['homography'],np.asarray(m['pitch']))
    policy=QualityConfig(unknown_airborne_policy=unknown_airborne_policy)
    selected,quality=select_observations(baseline,original['ball'],confidence,sigma,airborne,m['blocks'],policy)
    out.mkdir(parents=True)
    np.savez_compressed(out/'quality.npz',frame_ids=original['frame_ids'],cv_ball=original['ball'],baseline_ball=baseline,
        cv_source=original['ball_source'],confidence=confidence,confidence_kind=kind,
        localization_sigma_m=sigma,airborne_risk=airborne,**quality)
    original.update(ball=selected,ball_observed=quality['ground_observed'],ball_source=quality['decision'])
    np.savez_compressed(out/'observations.npz',**original)
    tags=manifest_path.parent/m['files']['tagging'];(out/'restarts.json').write_bytes(tags.read_bytes())
    m.update(files=dict(observations='observations.npz',tagging='restarts.json'),
        ball_source='baseline observations plus confidence/ground-qualified CV fills',
        quality_sidecar='quality.npz',quality_policy=asdict(policy),
        quality_adapter_sha256=digest(__file__),quality_policy_sha256=digest(ROOT/'scripts/possession_observation_quality.py'),
        parent_manifest_sha256=digest(manifest_path))
    m['quality_outputs_sha256']={name:digest(out/name) for name in ('observations.npz','quality.npz','restarts.json')}
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    summary=dict(decisions=dict(Counter(quality['decision'].tolist())),
        uncalibrated_scores=True,unknown_airborne_frames=int(np.isnan(airborne).sum()),
        height_estimated=False,training_performed=False,input_sha256=pins,default_changed=False)
    summary.update(quality_policy=asdict(policy),
        uncertain_projection_fills=int((quality['cv_used']&quality['ground_projection_uncertain']).sum()))
    if any(digest(p)!=h for p,h in pins.items()):raise ValueError('Input changed during conversion')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','features','cache','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--unknown-airborne-policy',choices=['require_anchors','allow_uncertain'],default='require_anchors')
    a=p.parse_args();print(json.dumps(run(a.manifest,a.features,a.cache,a.out,a.unknown_airborne_policy),indent=2))
