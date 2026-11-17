"""Create a gap-free estimated display track with explicit provenance.

The pre-fill recovery result is retained on every row. New points are labeled
inferred and have a heuristic uncertainty radius, not a calibrated probability.
Camera-compensated interpolation uses both endpoints where available. Leading
or trailing gaps use bounded extrapolation. Shots without observations use a
tactical prior or an explicitly unlocalized image-center placeholder.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
from stage1_cv_baseline import private_path, digest
from local_cv_possession import atomic
from sam_confirmation_recovery import read_rows
from recover_blank_tracks import finite, coverage


def fill(rows, width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Positive image dimensions required')
    result = copy.deepcopy(rows)
    diagonal = float(np.hypot(width, height))
    poses, groups = [], []
    start = 0
    pose = np.eye(3)
    for i, row in enumerate(rows):
        affine = row.get('camera_affine')
        valid = affine is not None and np.shape(affine) == (2, 3) and np.isfinite(affine).all()
        valid = valid and abs(np.linalg.det(np.asarray(affine)[:, :2])) > 1e-8
        cut = i == 0 or not valid or row['frame'] != rows[i-1]['frame'] + 1
        if cut:
            if i:
                groups.append((start, i))
            start, pose = i, np.eye(3)
        else:
            transform = np.eye(3)
            transform[:2] = affine
            pose = transform @ pose
        poses.append(pose.copy())
    if rows:
        groups.append((start, len(rows)))
    for row in result:
        row['evidence_position'] = copy.deepcopy(row.get('context_position'))
        row['evidence_source'] = row.get('context_source', 'missing')
        row['inferred_position'] = False
    for start, stop in groups:
        anchors = [i for i in range(start, stop) if finite(rows[i].get('context_position'))]
        local = {i: np.linalg.solve(poses[i], np.r_[rows[i]['context_position'], 1])[:2] for i in anchors}
        for i in range(start, stop):
            if finite(rows[i].get('context_position')):
                continue
            split = int(np.searchsorted(anchors, i))
            left = anchors[split-1] if split else None
            right = anchors[split] if split < len(anchors) else None
            anchor_frames = []
            if left is not None and right is not None:
                u = (i-left)/(right-left)
                point = (poses[i] @ np.r_[(1-u)*local[left] + u*local[right], 1])[:2]
                distance = min(i-left, right-i)
                basis = 'camera_compensated_interpolation'
                anchor_frames = [rows[left]['frame'], rows[right]['frame']]
            elif anchors:
                anchor = left if left is not None else right
                neighbor = anchors[-2] if left is not None and len(anchors)>1 else anchors[1] if right is not None and len(anchors)>1 else anchor
                velocity = (local[neighbor]-local[anchor])/(neighbor-anchor) if neighbor != anchor else np.zeros(2)
                speed = np.linalg.norm(velocity)
                velocity *= min(1., 45./max(speed, 1e-9))
                dt = i-anchor
                horizon = float(np.clip(dt, -25, 25))
                point = (poses[i] @ np.r_[local[anchor]+velocity*horizon, 1])[:2]
                distance = abs(dt)
                basis = 'bounded_extrapolation'
                anchor_frames = [rows[anchor]['frame']]
            else:
                point = np.asarray(rows[i]['guide'][:2]) if finite(rows[i].get('guide')) else np.array([width/2, height/2])
                distance = np.inf
                basis = 'tactical_ground_prior' if finite(rows[i].get('guide')) else 'unlocalized_placeholder'
            radius = min(diagonal, 8. + 3.*distance + .25*distance**2)
            offscreen = not (0 <= point[0] < width and 0 <= point[1] < height)
            result[i].update(context_position=point.tolist(), context_source='continuous_inferred',
                             inferred_position=True, inference=dict(basis=basis, anchors=anchor_frames,
                                 uncertainty_radius_px=float(radius), high_uncertainty=radius>=80,
                                 offscreen=offscreen, calibrated=False))
    return result


def run(args):
    cache, out = private_path(args.cache), private_path(args.out)
    if out.exists():
        raise ValueError('Use a new output directory')
    rows = read_rows(cache/'detections.jsonl')
    with np.load(args.features, allow_pickle=False) as features:
        meta = json.loads(str(features['metadata']))
        if features['frame_ids'].tolist() != [r['frame'] for r in rows]:
            raise ValueError('Feature alignment differs')
    result = fill(rows, meta['width'], meta['height'])
    inferred = [r for r in result if r['inferred_position']]
    summary = dict(frames=len(rows), before=coverage(rows), after=coverage(result),
                   inferred_frames=len(inferred), high_uncertainty_frames=sum(r['inference']['high_uncertainty'] for r in inferred),
                   bases=dict(Counter(r['inference']['basis'] for r in inferred)),
                   uncertainty_calibrated=False, accuracy_measured=False)
    if any(a.get('context_position') != b['context_position'] for a,b in zip(rows,result) if finite(a.get('context_position'))):
        raise AssertionError('Existing position changed')
    config = json.loads((cache/'config.json').read_text())
    config['continuous_trajectory'] = dict(parent_sha256=digest(cache/'detections.jsonl'),
                                          code_sha256=digest(__file__), user_authorized_estimates=True,
                                          uncertainty='heuristic pixel radius, not a calibrated confidence interval')
    out.mkdir(parents=True)
    atomic(out/'config.json',config)
    atomic(out/'summary.json',summary)
    (out/'detections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in result))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
