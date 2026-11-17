"""Audit cached SAM rejections and corroborate them with CPU optical flow.

Both measured endpoints must track through the whole gap, close on the opposite
measurement, and agree with the SAM proposal. No new SAM/YOLO inference or human
prompts are used. This writes a separate experiment cache from a frozen snapshot
of completed jobs, so the active SAM run can continue writing its own files.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import time

import numpy as np
from stage1_cv_baseline import ROOT, digest, private_path
from local_cv_possession import atomic
from sam2_gap_recovery import CONFIG as SAM_CONFIG, gap_jobs, observation, transport
from sam2_gap_recovery_v2 import anchors_for


CONFIG = dict(max_span=34, window_size=15, pyramid_levels=3,
              patch_size=11, min_patch_std=3., min_patch_correlation=.7,
              min_anchor_correlation=.55, max_roundtrip_error=1.5,
              max_endpoint_error=4., max_direction_error=4.,
              max_sam_error=6., max_step=60.)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def snapshot_jobs(path, count):
    """Ignore only an unfinished final line from an actively written JSONL."""
    data = Path(path).read_bytes()
    lines = data.split(b'\n')[:-1]
    return [json.loads(line) for line in lines if line][:count]


def candidate_points(row):
    return [p for key in ('candidates', 'player_context_candidates',
                         'crowded_zoom_candidates')
            for p in (row.get(key) or [])
            if len(p) == 3 and np.isfinite(p).all()
            and p[2] >= SAM_CONFIG['minimum_seed_confidence']]


def rejection_reason(records, row):
    if len(records) == 2:
        distance = np.linalg.norm(np.asarray(records[0]['point']) - records[1]['point'])
        return ('two_way_center_disagreement' if distance > SAM_CONFIG['agreement_px']
                else 'two_way_mask_iou_failure')
    return ('one_way_candidates_elsewhere' if candidate_points(row)
            else 'one_way_no_candidates')


def patch(image, point, size):
    import cv2
    x, y = point
    radius = size // 2 + 1
    if not (radius <= x < image.shape[1] - radius
            and radius <= y < image.shape[0] - radius):
        return None
    return cv2.getRectSubPix(image, (size, size), (float(x), float(y)))


def correlation(a, b, minimum_std):
    if a is None or b is None:
        return -1.
    a = a.astype(float).ravel()
    b = b.astype(float).ravel()
    if min(a.std(), b.std()) < minimum_std:
        return -1.
    a -= a.mean()
    b -= b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def track_direction(images, rows, anchor, target, config=None):
    """Track pixels without using SAM positions as flow initializations."""
    import cv2
    c = dict(CONFIG, **(config or {}))
    point = observation(rows[anchor])[:2].astype(np.float32)
    seed_patch = patch(images[anchor], point, c['patch_size'])
    if seed_patch is None or seed_patch.std() < c['min_patch_std']:
        return {}, 'uninformative_seed_patch'
    result = {anchor: point.copy()}
    step = 1 if target > anchor else -1
    velocity = np.zeros(2)
    for current in range(anchor, target, step):
        following = current + step
        expected = transport(point, rows, current, following)
        guess = (expected + velocity).astype(np.float32).reshape(1, 1, 2)
        options = dict(winSize=(c['window_size'], c['window_size']),
                       maxLevel=c['pyramid_levels'],
                       criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
        original = point.reshape(1, 1, 2)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            images[current], images[following], original, guess,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **options)
        if moved is None or not status[0, 0] or not np.isfinite(moved).all():
            return result, 'flow_lost'
        back, status, _ = cv2.calcOpticalFlowPyrLK(
            images[following], images[current], moved, None, **options)
        if (back is None or not status[0, 0] or not np.isfinite(back).all()
                or np.linalg.norm(back.reshape(2) - point) > c['max_roundtrip_error']):
            return result, 'roundtrip_failed'
        new_point = moved.reshape(2)
        if np.linalg.norm(new_point - expected) > c['max_step']:
            return result, 'motion_jump'
        previous_patch = patch(images[current], point, c['patch_size'])
        new_patch = patch(images[following], new_point, c['patch_size'])
        if correlation(previous_patch, new_patch, c['min_patch_std']) < c['min_patch_correlation']:
            return result, 'adjacent_appearance_failed'
        if correlation(seed_patch, new_patch, c['min_patch_std']) < c['min_anchor_correlation']:
            return result, 'anchor_appearance_failed'
        velocity = new_point - expected
        point = new_point
        result[following] = point.copy()
    endpoint = observation(rows[target])
    if endpoint is None or np.linalg.norm(point - endpoint[:2]) > c['max_endpoint_error']:
        return result, 'endpoint_failed'
    return result, 'confirmed'


def recover_job(rows, job, record, images, config=None):
    c = dict(CONFIG, **(config or {}))
    left = anchors_for(rows, job, False)
    right = anchors_for(rows, job, True)
    if not left or not right:
        return {}, dict(reason='missing_measured_endpoint')
    lo, hi = left[0], right[0]
    if hi - lo > c['max_span']:
        return {}, dict(reason='span_exceeds_limit')
    if any(rows[i].get('camera_affine') is None
           or rows[i]['frame'] != rows[i - 1]['frame'] + 1
           for i in range(lo + 1, hi + 1)):
        return {}, dict(reason='camera_discontinuity')
    forward, f_reason = track_direction(images, rows, lo, hi, c)
    backward, b_reason = track_direction(images, rows, hi, lo, c)
    detail = dict(forward=f_reason, backward=b_reason, reason='flow_unconfirmed')
    if f_reason != 'confirmed' or b_reason != 'confirmed':
        return {}, detail
    additions = {}
    reasons = Counter()
    for key, records in record['proposals'].items():
        i = int(key)
        if key in record['additions']:
            continue
        if np.linalg.norm(forward[i] - backward[i]) > c['max_direction_error']:
            reasons['flow_direction_disagreement'] += 1
            continue
        matches = [r for r in records
                   if max(np.linalg.norm(np.asarray(r['point']) - forward[i]),
                          np.linalg.norm(np.asarray(r['point']) - backward[i])) <= c['max_sam_error']]
        if len(matches) != 1:
            reasons['sam_ambiguous' if matches else 'sam_flow_disagreement'] += 1
            continue
        additions[key] = dict(position=matches[0]['point'], mask=matches[0],
                              source='sam2_optical_flow_supported',
                              forward=forward[i].tolist(), backward=backward[i].tolist(),
                              anchors=[rows[lo]['frame'], rows[hi]['frame']])
    detail.update(reason='confirmed_flow', frame_rejections=dict(reasons))
    return additions, detail


def run(args):
    import cv2
    cv2.setNumThreads(1)
    out = private_path(args.out)
    if out.exists():
        raise ValueError('Use a new output directory for a reproducible snapshot')
    rows = read_rows(args.cache / 'detections.jsonl')
    baseline = read_rows(args.baseline / 'detections.jsonl')
    if [r['frame'] for r in rows] != [r['frame'] for r in baseline]:
        raise ValueError('Baseline frame alignment mismatch')
    records = snapshot_jobs(args.sam_results / 'jobs.jsonl', args.snapshot_jobs)
    jobs = gap_jobs(rows)
    # Snapshot incorporates the accepted current-run results before adding flow.
    result = copy.deepcopy(baseline)
    for record in records:
        for key, proposals in record['proposals'].items():
            result[int(key)]['sam2_retry_proposals'] = proposals
        for key, addition in record['additions'].items():
            i = int(key)
            if observation(rows[i]) is None and not result[i].get('sam2_mask'):
                result[i].update(context_position=addition['position'],
                                 context_source=addition['source'], sam2_mask=addition['mask'])
    snapshot = copy.deepcopy(result)
    targets = [r for r in records if r['proposals'] and not r['additions']]
    config = json.loads((args.cache / 'config.json').read_text())
    cap = cv2.VideoCapture(config['source'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    audit = []
    started = time.monotonic()
    try:
        for number, record in enumerate(targets):
            job = jobs[record['job_index']]
            left, right = anchors_for(rows, job, False), anchors_for(rows, job, True)
            images = {}
            if left and right and right[0] - left[0] <= CONFIG['max_span']:
                cap.set(cv2.CAP_PROP_POS_FRAMES, rows[left[0]]['frame'])
                for i in range(left[0], right[0] + 1):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Video decode failed')
                    images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            additions, detail = recover_job(rows, job, record, images)
            detail.update(job_index=record['job_index'], additions=additions,
                          rejection_counts=dict(Counter(
                              rejection_reason(p, rows[int(key)])
                              for key, p in record['proposals'].items())))
            audit.append(detail)
            for key, addition in additions.items():
                i = int(key)
                if observation(rows[i]) is not None or result[i].get('sam2_mask'):
                    continue
                result[i].update(context_position=addition['position'],
                                 context_source=addition['source'], sam2_mask=addition['mask'],
                                 sam2_flow_evidence=addition)
            print(f'CPU confirmation {number + 1}/{len(targets)} gaps; '
                  f'{sum(len(a["additions"]) for a in audit)} corroborated frames', flush=True)
    finally:
        cap.release()
    summary = dict(snapshot_jobs=len(records), investigated_gaps=len(targets),
                   outcomes=dict(Counter(a['reason'] for a in audit)),
                   forward_outcomes=dict(Counter(a.get('forward', a['reason']) for a in audit)),
                   backward_outcomes=dict(Counter(a.get('backward', a['reason']) for a in audit)),
                   rejection_counts=dict(sum((Counter(a['rejection_counts']) for a in audit), Counter())),
                   additional_supported_frames=sum(a.get('sam2_mask') is None and b.get('sam2_mask') is not None
                                                   for a, b in zip(snapshot, result)),
                   newly_filled=sum(a.get('context_position') is None and b.get('context_position') is not None
                                    for a, b in zip(snapshot, result)),
                   elapsed_seconds=time.monotonic() - started, accuracy_measured=False)
    out.mkdir(parents=True)
    config['sam2_confirmation_recovery'] = dict(CONFIG, code_sha256=digest(__file__),
                                               snapshot_job_count=len(records))
    config['sam2_gap_recovery'] = dict(experimental=True, variant='SAM plus two-ended optical flow')
    atomic(out / 'config.json', config)
    atomic(out / 'summary.json', summary)
    (out / 'jobs_snapshot.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (out / 'confirmation_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit))
    (out / 'detections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in result))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--sam-results', type=Path, required=True)
    parser.add_argument('--snapshot-jobs', type=int, default=112)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
