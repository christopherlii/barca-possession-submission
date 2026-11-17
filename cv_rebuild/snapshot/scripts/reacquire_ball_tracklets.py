"""Reacquire missing ball tracks using candidate sequences and two-way SAM.

Provisional detector paths do not need an old accepted anchor. They must have
repeated strong image detections, camera-compensated motion and no ambiguity.
SAM is then independently prompted at both ends. Both propagations must reach
the opposite candidate and agree with each other and several interior detections.
Only actual mask positions are emitted. Estimates never become seed evidence.
No manual prompts, frame-specific rules or public labels are used.
"""
from cached_video_reader import CachedVideoCapture
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from stage1_cv_baseline import ROOT, digest, private_path
from local_cv_possession import atomic
from recover_blank_tracks import candidates, finite, window_bounds
from smooth_candidate_window import associate
from extend_sam_ball_tracks import is_target, propagate_direction
from sam2_gap_recovery import runtime, agreeing
from sam_confirmation_recovery import read_rows

CONFIG = dict(min_observations=8, strong_score=.4, min_strong=4,
              min_travel_px=16., max_alternatives=3, candidate_error_px=8.,
              min_verified_frames=8, min_interior_detector_matches=4,
              conflict_distance_px=12., max_seed_variants=3)


def penalty_overlap(row, p):
    """Retain active penalty-paint protection during automatic reacquisition."""
    tracked = row.get('active_penalty_spots') or {}
    for name in ('left_penalty_spot', 'right_penalty_spot'):
        track = tracked.get(name, {})
        if track.get('confirmed') and finite(track.get('center')):
            if np.linalg.norm(np.asarray(p)-track['center']) <= track['veto_radius_px']:
                return True
        else:
            for shape in row.get('field_geometry') or []:
                if shape['name'] == name and len(shape['points']) == 1:
                    if np.linalg.norm(np.asarray(p)-shape['points'][0]) <= 16.:
                        return True
    return False


def plan_tracklets(rows):
    """Propose multiple spatial paths; SAM validation is mandatory downstream."""
    plans, seen = [], set()
    # Include estimated spans even when the earlier stage drew a placeholder.
    targets = [dict(r, context_position=None) if is_target(r) else r for r in rows]
    for lo, hi in window_bounds(targets):
        poses = [np.eye(3)]
        for row in rows[lo+1:hi]:
            affine = np.eye(3)
            affine[:2] = row['camera_affine']
            poses.append(affine @ poses[-1])
        samples = []
        for row, pose in zip(rows[lo:hi], poses):
            inv = np.linalg.inv(pose)
            points = candidates(row) if is_target(row) else []
            samples.append([[*(inv @ np.r_[p[:2], 1])[:2], p[2]]
                            for p in points if not penalty_overlap(row, p[:2])])
        for _ in range(CONFIG['max_alternatives']):
            path, report = associate(samples)
            if not path:
                break
            # Remove just this spatial proposal to inspect alternatives rather
            # than letting a long static distractor hide a shorter moving path.
            for j, p in path:
                samples[j] = [q for q in samples[j] if np.linalg.norm(np.asarray(q[:2])-p[:2]) > 8]
            if report['status'] != 'associated':
                continue
            strong = [k for k, (_, p) in enumerate(path) if p[2] >= CONFIG['strong_score']]
            if len(strong) < CONFIG['min_strong']:
                continue
            path = path[strong[0]:strong[-1]+1]
            if len(path) < CONFIG['min_observations']:
                continue
            travel = float(np.linalg.norm(np.asarray(path[-1][1][:2])-path[0][1][:2]))
            if travel < CONFIG['min_travel_px']:
                continue
            points = {lo+j: [*(poses[j] @ np.r_[p[:2], 1])[:2], float(p[2])] for j, p in path}
            first, last = min(points), max(points)
            key = (first, last, *(round(float(v), 1) for v in points[first][:2]))
            if key in seen:
                continue
            seen.add(key)
            plans.append(dict(lo=first, hi=last, points=points, travel_px=travel,
                              observations=len(path), strong_observations=len(strong)))
    return plans


def seed_variants(plan):
    """Retry initialization using strong detections in the two outer thirds.

    Selection uses detector evidence only, never observed success on a particular
    match frame. Each retry must independently pass the same validation rules.
    """
    points = {int(k):v for k,v in plan['points'].items()}
    lo, hi = plan['lo'], plan['hi']
    width = (hi-lo)/3
    left = sorted((i for i,p in points.items() if i <= lo+width and p[2] >= CONFIG['strong_score']),
                  key=lambda i: (-points[i][2], i))
    right = sorted((i for i,p in points.items() if i >= hi-width and p[2] >= CONFIG['strong_score']),
                   key=lambda i: (-points[i][2], -i))
    pairs = [(lo,hi)] + list(zip(left, right))
    variants, seen = [], set()
    for a,b in pairs:
        subset = {i:p for i,p in points.items() if a <= i <= b}
        if ((a,b) in seen or len(subset) < CONFIG['min_observations']
                or b-a-1 < CONFIG['min_verified_frames']):
            continue
        seen.add((a,b))
        variants.append(dict(plan, lo=a, hi=b, points=subset))
        if len(variants) == CONFIG['max_seed_variants']:
            break
    return variants


def validate_tracklet(rows, plan, forward, backward):
    """Both endpoint tests plus interior agreement; no detector-only fallback.

    SAM and the detector are correlated evidence, so this is a consistency test,
    not a calibrated confidence or proof of object identity.
    """
    points = {int(k):v for k,v in plan['points'].items()}
    verified, matches = {}, 0
    for index in forward.keys() & backward.keys():
        a, b = forward[index], backward[index]
        if not a['endpoint_confirmed'] or not b['endpoint_confirmed']:
            continue
        if not agreeing(a['mask'], b['mask']):
            continue
        position = np.asarray(a['mask']['point'])
        if penalty_overlap(rows[index], position):
            continue
        if index in points:
            if np.linalg.norm(position-points[index][:2]) > CONFIG['candidate_error_px']:
                continue
            matches += 1
        verified[index] = dict(position=position.tolist(), mask=a['mask'],
                               basis='two_way_sam_reacquisition',
                               anchors=[rows[plan['lo']]['frame'], rows[plan['hi']]['frame']])
    accepted = (len(verified) >= CONFIG['min_verified_frames']
                and matches >= CONFIG['min_interior_detector_matches'])
    return (verified if accepted else {}), dict(accepted=accepted,
        agreeing_mask_frames=len(verified), interior_detector_matches=matches)


def merge_verified(rows, proposals):
    """Reject conflicting tracklets and preserve all previous image observations."""
    result = copy.deepcopy(rows)
    conflicts = 0
    for index, additions in proposals.items():
        if not is_target(rows[index]):
            continue
        first = additions[0]
        if any(np.linalg.norm(np.asarray(a['position'])-first['position']) > CONFIG['conflict_distance_px']
               for a in additions[1:]):
            conflicts += 1
            continue
        result[index].update(context_position=first['position'], context_source='sam2_reacquired',
                             sam2_mask=first['mask'], sam2_reacquisition=first)
    return result, conflicts


def run(args):
    import cv2
    import torch
    cache, out = private_path(args.cache), private_path(args.out)
    if out.exists():
        raise ValueError('Use a new output directory')
    config = json.loads((cache/'config.json').read_text())
    if config.get('continuous_trajectory'):
        raise ValueError('Use the unfilled evidence cache before continuous interpolation')
    rows = read_rows(cache/'detections.jsonl')
    plans = [p for p in plan_tracklets(rows) if seed_variants(p)]
    # These optional bounds limit compute for diagnostics; they never change
    # the detector path criteria, prompts or acceptance thresholds.
    plans = [p for p in plans if (args.start_frame is None or rows[p['hi']]['frame'] >= args.start_frame)
             and (args.end_frame is None or rows[p['lo']]['frame'] <= args.end_frame)]
    if args.max_tracklets is not None:
        plans = plans[:args.max_tracklets]
    out.mkdir(parents=True)
    pins = {name:digest(Path(__file__).with_name(name)) for name in
            ('reacquire_ball_tracklets.py', 'extend_sam_ball_tracks.py', 'smooth_candidate_window.py',
             'recover_blank_tracks.py', 'sam2_gap_recovery.py', 'sam2_gap_recovery_v2.py')}
    signature = dict(config=CONFIG, source_sha256=digest(cache/'detections.jsonl'),
                     code_sha256=pins, model_sha256=digest(ROOT/'outputs/sam2-models/sam2.1_hiera_tiny.pt'),
                     start_frame=args.start_frame, end_frame=args.end_frame, max_tracklets=args.max_tracklets,
                     device=args.device, seed=args.seed)
    atomic(out/'plan.json', dict(signature, tracklets=plans))
    started = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    predictor = runtime(args.device) if plans else None
    cap = CachedVideoCapture(config['source'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    cv2.setNumThreads(2)
    proposals, accepted_jobs = {}, 0
    try:
        with (out/'audit.jsonl').open('w') as audit:
            for n, plan in enumerate(plans):
                attempts = []
                for variant in seed_variants(plan):
                    lo, hi = variant['lo'], variant['hi']
                    local = copy.deepcopy(rows[lo:hi+1])
                    # Prompts exist only in this local validation copy.
                    for i in (lo, hi):
                        local[i-lo]['blank_recovery_detection'] = variant['points'][i]
                    job = dict(lo=1, hi=len(local)-2)
                    forward, fa = propagate_direction(predictor, local, job, False, cap, out)
                    backward, ba = propagate_direction(predictor, local, job, True, cap, out)
                    forward = {i+lo:v for i,v in forward.items()}
                    backward = {i+lo:v for i,v in backward.items()}
                    additions, report = validate_tracklet(rows, variant, forward, backward)
                    attempts.append(dict(seeds=[rows[lo]['frame'], rows[hi]['frame']],
                                         report=report, attempts=fa+ba))
                    if report['accepted']:
                        break
                for i, addition in additions.items():
                    proposals.setdefault(i, []).append(addition)
                accepted_jobs += int(report['accepted'])
                audit.write(json.dumps(dict(tracklet=n, report=report, variants=attempts, additions=additions))+'\n')
                audit.flush()
                status = dict(state='running', completed=n+1, total=len(plans), accepted_tracklets=accepted_jobs,
                              proposed_frames=len(proposals), elapsed_seconds=time.monotonic()-started)
                atomic(out/'status.json', status)
                print(json.dumps(status), flush=True)
                if args.device == 'mps':
                    torch.mps.empty_cache()
    finally:
        cap.release()
    result, conflicts = merge_verified(rows, proposals)
    changed = [b for a,b in zip(rows,result) if a.get('context_position') != b.get('context_position')]
    summary = dict(frames=len(rows), tracklets=len(plans), accepted_tracklets=accepted_jobs,
                   recovered_frames=len(changed), conflicting_frames=conflicts,
                   elapsed_seconds=time.monotonic()-started, accuracy_measured=False)
    config['sam2_reacquisition'] = signature
    atomic(out/'config.json', config)
    with (out/'detections.jsonl').open('w') as handle:
        for row in result:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    atomic(out/'summary.json', summary)
    atomic(out/'status.json', dict(state='complete', **summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--max-tracklets', type=int)
    parser.add_argument('--seed', type=int, default=0, help='Fixed initialization seed; GPU bitwise repeatability is not guaranteed')
    run(parser.parse_args())
