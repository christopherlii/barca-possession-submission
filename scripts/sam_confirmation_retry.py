"""Retry unresolved SAM proposals using alternate flow seeds and fresh CPU CV.

Input is a completed confirmation snapshot. It never changes the active SAM
run or overwrites existing confirmed positions. Alternate flow paths keep the
same appearance, round-trip, endpoint and agreement checks as the first pass.
Fresh YOLO inference uses native-pixel crops and the original SAM/CV acceptance
thresholds. Every proposal and rejection remains in a private audit file.
"""
import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
from stage1_cv_baseline import ROOT, digest, private_path
from local_cv_possession import atomic
from sam2_gap_recovery import CONFIG as SAM_CONFIG, gap_jobs, observation
from sam2_gap_recovery_v2 import anchors_for
from sam_confirmation_recovery import CONFIG as FLOW_CONFIG, read_rows, track_direction


CONFIG = dict(anchor_lookaround=16, max_anchors=3, max_span=64,
              flow_scales=[(9, 7), (15, 11), (21, 15)],
              crop_sizes=[128, 64], crop_modes=['raw', 'clahe'], detector_size=640)


def select_mask(records, forward, backward):
    if np.linalg.norm(forward - backward) > FLOW_CONFIG['max_direction_error']:
        return None
    matches = [r for r in records if max(
        np.linalg.norm(np.asarray(r['point']) - forward),
        np.linalg.norm(np.asarray(r['point']) - backward)) <= FLOW_CONFIG['max_sam_error']]
    return matches[0] if len(matches) == 1 else None


def retry_flow(rows, job, record, images):
    left = anchors_for(rows, job, False, CONFIG)
    right = anchors_for(rows, job, True, CONFIG)
    attempts, additions = [], {}
    for lo in left:
        for hi in right:
            if hi - lo > CONFIG['max_span']:
                continue
            if any(rows[i].get('camera_affine') is None
                   or rows[i]['frame'] != rows[i - 1]['frame'] + 1
                   for i in range(lo + 1, hi + 1)):
                continue
            for window, size in CONFIG['flow_scales']:
                config = dict(window_size=window, patch_size=size)
                forward, f_reason = track_direction(images, rows, lo, hi, config)
                backward, b_reason = track_direction(images, rows, hi, lo, config)
                attempts.append(dict(anchors=[lo, hi], window_size=window,
                                     patch_size=size, forward=f_reason, backward=b_reason))
                if f_reason != 'confirmed' or b_reason != 'confirmed':
                    continue
                for key, masks in record['proposals'].items():
                    i = int(key)
                    if key in additions:
                        continue
                    mask = select_mask(masks, forward[i], backward[i])
                    if mask:
                        additions[key] = dict(position=mask['point'], mask=mask,
                                              source='sam2_multiscale_flow_supported',
                                              anchors=[rows[lo]['frame'], rows[hi]['frame']],
                                              window_size=window, patch_size=size)
                if len(additions) == len(record['proposals']):
                    return additions, attempts
    return additions, attempts


def detector_match(records, candidates):
    """Fresh detection must corroborate exactly one SAM hypothesis."""
    supported = []
    for mask in records:
        matches = [p for p in candidates if len(p) == 3 and np.isfinite(p).all()
                   and p[2] >= SAM_CONFIG['minimum_seed_confidence']
                   and np.linalg.norm(np.asarray(p[:2]) - mask['point'])
                   <= SAM_CONFIG['candidate_agreement_px']]
        if matches:
            supported.append((mask, max(matches, key=lambda p: p[2])))
    return supported[0] if len(supported) == 1 else None


def crop_box(point, size, width, height):
    x = int(np.clip(round(point[0] - size / 2), 0, max(0, width - size)))
    y = int(np.clip(round(point[1] - size / 2), 0, max(0, height - size)))
    return [x, y, min(width, x + size), min(height, y + size)]


def detector_retry(image, records, detect):
    """Evaluate tight crops in fixed order, stopping on unique corroboration."""
    from frame_preprocessing_experiment import preprocess
    height, width = image.shape[:2]
    attempts = []
    for size in CONFIG['crop_sizes']:
        for mask in records:
            box = crop_box(mask['point'], size, width, height)
            x, y, xx, yy = box
            crop = image[y:yy, x:xx]
            for mode in CONFIG['crop_modes']:
                processed = preprocess(crop, mode)
                local = detect(processed)
                points = [[float(p[0] + x), float(p[1] + y), float(p[2])] for p in local]
                attempts.append(dict(crop=box, mode=mode, candidates=points))
                match = detector_match(records, points)
                if match:
                    record, candidate = match
                    return dict(position=record['point'], mask=record,
                                source='sam2_tight_crop_cv_supported',
                                detector_candidate=candidate, crop=box, mode=mode), attempts
    return None, attempts


def run(args):
    import cv2
    cv2.setNumThreads(1)
    out = private_path(args.out)
    if out.exists():
        raise ValueError('Use a new experiment output directory')
    rows = read_rows(args.cache / 'detections.jsonl')
    result = read_rows(args.previous / 'detections.jsonl')
    if [r['frame'] for r in rows] != [r['frame'] for r in result]:
        raise ValueError('Snapshot frame alignment mismatch')
    original = copy.deepcopy(result)
    previous = read_rows(args.previous / 'confirmation_audit.jsonl')
    records = {r['job_index']: r for r in read_rows(args.previous / 'jobs_snapshot.jsonl')}
    targets = [r['job_index'] for r in previous if not r['additions']]
    jobs = gap_jobs(rows)
    config = json.loads((args.cache / 'config.json').read_text())
    cap = cv2.VideoCapture(config['source'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    audit = []
    model = None
    start = time.monotonic()
    try:
        for number, index in enumerate(targets):
            job, record = jobs[index], records[index]
            left = anchors_for(rows, job, False, CONFIG)
            right = anchors_for(rows, job, True, CONFIG)
            gray = {}
            pairs = [(a, b) for a in left for b in right if b - a <= CONFIG['max_span']]
            if pairs:
                lo, hi = min(a for a, _ in pairs), max(b for _, b in pairs)
                cap.set(cv2.CAP_PROP_POS_FRAMES, rows[lo]['frame'])
                for i in range(lo, hi + 1):
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Video decode failed')
                    gray[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            additions, flow_attempts = retry_flow(rows, job, record, gray)
            crop_attempts = {}
            for key, masks in record['proposals'].items():
                i = int(key)
                if key in additions or result[i].get('sam2_mask') or observation(rows[i]) is not None:
                    continue
                if model is None:
                    import torch
                    os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'outputs/two-stage-cv-settings')
                    os.environ['YOLO_AUTOINSTALL'] = 'False'
                    from ultralytics import YOLO
                    torch.set_num_threads(1)
                    model = YOLO(ROOT / 'outputs/two-stage-cv-models/yolo11x.pt')
                from frame_preprocessing_experiment import detect as yolo_detect
                cap.set(cv2.CAP_PROP_POS_FRAMES, rows[i]['frame'])
                ok, image = cap.read()
                if not ok:
                    raise ValueError('Video decode failed')
                def detect(crop):
                    return yolo_detect(model, crop, 'cpu', CONFIG['detector_size'])
                recovered, attempts = detector_retry(image, masks, detect)
                crop_attempts[key] = attempts
                if recovered:
                    additions[key] = recovered
            for key, recovered in additions.items():
                i = int(key)
                if observation(rows[i]) is None and not result[i].get('sam2_mask'):
                    result[i].update(context_position=recovered['position'],
                                     context_source=recovered['source'], sam2_mask=recovered['mask'],
                                     sam2_confirmation_retry=recovered)
            audit.append(dict(job_index=index, additions=additions,
                              flow_attempts=flow_attempts, crop_attempts=crop_attempts))
            print(f'Remaining-gap retry {number + 1}/{len(targets)}; '
                  f'{sum(len(a["additions"]) for a in audit)} corroborated positions', flush=True)
    finally:
        cap.release()
    changed = [i for i, (a, b) in enumerate(zip(original, result))
               if a.get('sam2_mask') is None and b.get('sam2_mask') is not None]
    summary = dict(target_gaps=len(targets), recovered_gaps=sum(bool(a['additions']) for a in audit),
                   additional_supported_frames=len(changed),
                   sources=dict(Counter(result[i]['context_source'] for i in changed)),
                   newly_filled=sum(original[i].get('context_position') is None for i in changed),
                   crop_inferences=sum(len(v) for a in audit for v in a['crop_attempts'].values()),
                   flow_attempts=sum(len(a['flow_attempts']) for a in audit),
                   elapsed_seconds=time.monotonic() - start, accuracy_measured=False)
    out.mkdir(parents=True)
    config = json.loads((args.previous / 'config.json').read_text())
    config['sam_confirmation_retry'] = dict(CONFIG, code_sha256=digest(__file__),
                                            detector_device='cpu', previous=str(args.previous))
    atomic(out / 'config.json', config)
    atomic(out / 'summary.json', summary)
    (out / 'retry_audit.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in audit))
    (out / 'detections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in result))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
