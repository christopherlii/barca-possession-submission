"""Reject weak ball anchors contradicted by nearby strong image observations.

A low-score observation or estimated position is discarded only when unreachable
from BOTH bracketing strong image measurements under the existing zoom-aware
speed envelope. The bad point is removed before interpolation/motion smoothing,
so smoothing cannot turn an implausible jump into a plausible-looking excursion.
No ground labels, tactical ball priors or manually specified frames are used.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from stage1_cv_baseline import private_path, digest
from local_cv_possession import atomic
from sam_confirmation_recovery import read_rows
from sam2_gap_recovery import transport
from recover_blank_tracks import finite
from constrained_ball_trajectory import camera_zoom_scales, same_camera, CONFIG as MOTION

CONFIG = dict(strong_score=.4, maximum_bracket_frames=12, measurement_padding_px=16.)
DETECTIONS = ('selected','short_gap_detection','aerial_detection','blank_recovery_detection','bidirectional_detection')


def strong_point(row):
    if row.get('sam2_mask') and finite(row.get('context_position')):
        return np.asarray(row['context_position'][:2],float)
    for key in DETECTIONS:
        p=row.get(key)
        if finite(p) and len(p)==3 and p[2]>=CONFIG['strong_score']:
            return np.asarray(p[:2],float)
    return None


def audit(rows):
    result=copy.deepcopy(rows)
    scales=camera_zoom_scales(rows)
    strong={i:p for i,r in enumerate(rows) if (p:=strong_point(r)) is not None}
    keys=sorted(strong)
    for left,right in zip(keys,keys[1:]):
        if right-left>CONFIG['maximum_bracket_frames'] or not same_camera(rows,left,right):
            continue
        for i in range(left+1,right):
            row=rows[i]
            def incompatible(p):
                if not finite(p):return False
                a=transport(strong[left],rows,left,i)
                b=transport(strong[right],rows,right,i)
                padding=CONFIG['measurement_padding_px']*scales[i]
                limit=MOTION['max_speed_px_frame']*scales[i]
                return (np.linalg.norm(np.asarray(p[:2])-a)>limit*(i-left)+padding
                        and np.linalg.norm(np.asarray(p[:2])-b)>limit*(right-i)+padding)
            removed={}
            for key in DETECTIONS:
                if incompatible(row.get(key)):
                    removed[key]=row[key]
                    result[i][key]=None
            if incompatible(row.get('context_position')):
                removed.update(context_position=row['context_position'],context_source=row.get('context_source'))
                result[i].update(context_position=None,context_source='anchor_consistency_rejected')
                for key in ('forecast_3_frames','motion_prediction'):
                    if row.get(key) is not None:
                        removed[key]=row[key];result[i][key]=None
            for key in ('candidates','crowded_zoom_candidates','player_context_candidates','recovery_candidates','blank_recovery_candidates'):
                if row.get(key):
                    kept=[p for p in row[key] if not incompatible(p)]
                    if kept!=row[key]:removed[key]=row[key];result[i][key]=kept
            if removed:
                result[i]['anchor_consistency']=dict(rejected=True,
                    reason='unreachable_from_both_strong_image_anchors',
                    anchor_frames=[rows[left]['frame'],rows[right]['frame']],removed_evidence=removed)
    return result


def run(args):
    out=private_path(args.out)
    if out.exists():raise ValueError('Use a new output directory')
    rows=read_rows(args.cache/'detections.jsonl')
    config=json.loads((args.cache/'config.json').read_text())
    if config.get('continuous_trajectory'):raise ValueError('Audit evidence before continuous filling')
    result=audit(rows)
    summary=dict(frames=len(rows),rejected_position_frames=sum(finite(a.get('context_position')) and not finite(b.get('context_position')) for a,b in zip(rows,result)),
                 flagged_frames=sum(bool(r.get('anchor_consistency')) for r in result),accuracy_measured=False)
    config['anchor_consistency']=dict(CONFIG,parent_sha256=digest(args.cache/'detections.jsonl'),code_sha256=digest(__file__),speed_limit=MOTION['max_speed_px_frame'])
    out.mkdir(parents=True)
    atomic(out/'config.json',config);atomic(out/'summary.json',summary)
    with (out/'detections.jsonl').open('w') as f:
        for r in result:f.write(json.dumps(r,allow_nan=False)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
