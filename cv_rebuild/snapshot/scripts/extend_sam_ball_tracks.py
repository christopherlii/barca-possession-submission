"""Extend image-supported SAM tracks into remaining estimated/missing intervals.

Prompts are automatic image observations. Small background-negative prompts and
endpoint-guided image crops address oversized seed masks. Crop centers are only
search hints. Every output location is the centroid of an actual predicted mask.
Independent reverse propagation, a measured endpoint, or a detector candidate
must corroborate a propagated mask before it replaces an estimated position.
"""
from cached_video_reader import CachedVideoCapture
import argparse
import copy
import gc
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stage1_cv_baseline import ROOT, digest, private_path
from local_cv_possession import atomic
from recover_blank_tracks import candidates, finite
from sam_confirmation_recovery import read_rows
from sam2_gap_recovery import runtime, transport, agreeing, CONFIG as MASK_CONFIG
from sam2_gap_recovery_v2 import mask_diagnostics


CONFIG = dict(min_anchor_confidence=.25, anchor_lookaround=12, max_anchors=2,
              max_direction_frames=96, crop_sizes=[128, 256], negative_radius_px=12.,
              prompt_box_radius_px=5., endpoint_error_px=8., candidate_error_px=8.,
              max_step_px=60., min_confirmed_track_frames=3)
ESTIMATED_SOURCES = {'missing', 'predicted', 'smoothed_gap', 'blank_recovery_interpolated',
                     'bidirectional_interpolated', 'aerial_interpolated_path', 'continuous_inferred'}


def image_measurement(row):
    for key in ('blank_recovery_detection', 'short_gap_detection', 'selected',
                'aerial_detection', 'bidirectional_detection'):
        point = row.get(key)
        if finite(point) and len(point) == 3 and point[2] >= CONFIG['min_anchor_confidence']:
            return np.asarray(point, float)
    if row.get('sam2_mask') and finite(row.get('context_position')):
        return np.r_[row['context_position'], .5]
    return None


def is_target(row):
    return not finite(row.get('context_position')) or row.get('context_source') in ESTIMATED_SOURCES


def missing_intervals(rows):
    target = [is_target(row) for row in rows]
    intervals = []
    i = 0
    while i < len(rows):
        if not target[i]:
            i += 1
            continue
        end = i + 1
        while (end < len(rows) and target[end] and rows[end]['camera_affine'] is not None
               and rows[end]['frame'] == rows[end-1]['frame']+1):
            end += 1
        intervals.append(dict(lo=i, hi=end-1))
        i = end
    merged = []
    for job in intervals:
        bridge = range(merged[-1]['hi']+1, job['lo']) if merged else []
        same_shot = merged and all(rows[k]['camera_affine'] is not None
                    and rows[k]['frame'] == rows[k-1]['frame']+1
                    for k in range(merged[-1]['hi']+1, job['lo']+1))
        if same_shot and not any(image_measurement(rows[k]) is not None for k in bridge):
            merged[-1]['hi'] = job['hi']
        else:
            merged.append(job)
    return merged


def anchors(rows, job, reverse):
    boundary = job['hi']+1 if reverse else job['lo']-1
    found = []
    for step in range(CONFIG['anchor_lookaround']+1):
        index = boundary+step if reverse else boundary-step
        if not 0 <= index < len(rows):
            break
        edge = index if reverse else index+1
        if step and (rows[edge]['camera_affine'] is None
                     or rows[edge]['frame'] != rows[edge-1]['frame']+1):
            break
        point = image_measurement(rows[index])
        if point is not None:
            found.append(index)
            if len(found) == CONFIG['max_anchors']:
                break
    return found


def crop_centers(rows, lo, hi, anchor, opposite):
    """Interpolate a search ROI, while all returned ball positions come from SAM."""
    start = image_measurement(rows[anchor])[:2]
    destination = transport(image_measurement(rows[opposite])[:2], rows, opposite, anchor) if opposite is not None else None
    velocity = np.zeros(2)
    if destination is None:
        neighbor = anchor+1 if anchor == hi else anchor-1
        if 0 <= neighbor < len(rows) and image_measurement(rows[neighbor]) is not None:
            velocity = (transport(image_measurement(rows[neighbor])[:2], rows, neighbor, anchor)-start)/(neighbor-anchor)
            velocity *= min(1., 45./max(np.linalg.norm(velocity), 1e-9))
    centers = []
    for index in range(lo, hi+1):
        offset = ((destination-start)*(index-anchor)/(opposite-anchor)
                  if destination is not None else velocity*(index-anchor))
        centers.append(transport(start+offset, rows, anchor, index))
    return centers


def prompt_points(center, size, ring):
    points = [center]
    labels = [1]
    if ring:
        for angle in np.arange(8)*np.pi/4:
            point = center + CONFIG['negative_radius_px']*np.array([np.cos(angle), np.sin(angle)])
            if (point > 0).all() and (point < size-1).all():
                points.append(point)
                labels.append(0)
    return np.asarray(points, np.float32), np.asarray(labels, np.int32)


def propagate_direction(predictor, rows, job, reverse, cap, output):
    import cv2
    import torch
    seeds = anchors(rows, job, reverse)
    other = anchors(rows, job, not reverse)
    best, attempts = {}, []
    for anchor in seeds:
        opposite = other[0] if other else None
        edge = max(job['lo'], anchor-CONFIG['max_direction_frames']) if reverse else min(job['hi'], anchor+CONFIG['max_direction_frames'])
        # Include the opposite observed endpoint when it fits this window.
        if opposite is not None and abs(opposite-anchor) <= CONFIG['max_direction_frames']:
            edge = opposite
        lo, hi = sorted((anchor, edge))
        if any(rows[i]['camera_affine'] is None or rows[i]['frame'] != rows[i-1]['frame']+1
               for i in range(lo+1, hi+1)):
            attempts.append(dict(anchor=rows[anchor]['frame'], reason='camera_discontinuity'))
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, rows[lo]['frame'])
        images = []
        for _ in range(lo, hi+1):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Video decode failed')
            images.append(image)
        height, width = images[0].shape[:2]
        centers = crop_centers(rows, lo, hi, anchor, opposite)
        for size in CONFIG['crop_sizes']:
            origins = [np.clip(np.round(center-size/2), [0, 0], [width-size, height-size]).astype(int)
                       for center in centers]
            with tempfile.TemporaryDirectory(dir=output, prefix='sam-extension-') as directory:
                for index, (image, origin) in enumerate(zip(images, origins)):
                    x, y = origin
                    if not cv2.imwrite(str(Path(directory)/f'{index:05d}.jpg'), image[y:y+size, x:x+size], [cv2.IMWRITE_JPEG_QUALITY, 100]):
                        raise ValueError('Crop write failed')
                with torch.inference_mode():
                    state = predictor.init_state(directory, offload_video_to_cpu=True, offload_state_to_cpu=True)
                    for ring in (True, False):
                        predictor.reset_state(state)
                        local = image_measurement(rows[anchor])[:2]-origins[anchor-lo]
                        points, labels = prompt_points(local, size, ring)
                        radius = CONFIG['prompt_box_radius_px']
                        box = np.r_[np.maximum(0, local-radius), np.minimum(size-1, local+radius)].astype(np.float32)
                        _, _, logits = predictor.add_new_points_or_box(state, frame_idx=anchor-lo, obj_id=1,
                            points=points, labels=labels, box=box)
                        seed, diagnostics = mask_diagnostics(logits[0].cpu().numpy(), origins[anchor-lo], image_measurement(rows[anchor])[:2])
                        note = dict(anchor=rows[anchor]['frame'], reverse=reverse, crop_size=size, negatives=ring, seed=diagnostics)
                        attempts.append(note)
                        if seed is None:
                            continue
                        propagated = {}
                        last, last_index = np.asarray(seed['point']), anchor
                        for frame, _, masks in predictor.propagate_in_video(state, start_frame_idx=anchor-lo,
                                max_frame_num_to_track=hi-lo, reverse=reverse):
                            index = lo+frame
                            mask, reason = mask_diagnostics(masks[0].cpu().numpy(), origins[frame])
                            if mask is None:
                                note['stop'] = reason['reason']
                                break
                            ratio = mask['area']/seed['area']
                            if not MASK_CONFIG['area_ratio_min'] <= ratio <= MASK_CONFIG['area_ratio_max']:
                                note['stop'] = 'area_change'
                                break
                            expected = transport(last, rows, last_index, index)
                            if np.linalg.norm(np.asarray(mask['point'])-expected) > CONFIG['max_step_px']*max(1, abs(index-last_index)):
                                note['stop'] = 'motion_jump'
                                break
                            propagated[index] = mask
                            last, last_index = np.asarray(mask['point']), index
                        endpoint_ok = (len(propagated) >= CONFIG['min_confirmed_track_frames']
                            and opposite is not None and opposite in propagated
                            and np.linalg.norm(np.asarray(propagated[opposite]['point'])-image_measurement(rows[opposite])[:2]) <= CONFIG['endpoint_error_px'])
                        note.update(propagated_frames=len(propagated), endpoint_confirmed=bool(endpoint_ok))
                        proposed = {i:dict(mask=mask, endpoint_confirmed=bool(endpoint_ok), anchor=rows[anchor]['frame'],
                                          track_length=len(propagated))
                                    for i, mask in propagated.items() if job['lo'] <= i <= job['hi']}
                        if endpoint_ok or len(proposed) > len(best):
                            best = proposed
                        if endpoint_ok:
                            del state
                            return best, attempts
                    del state
            gc.collect()
    return best, attempts


def confirmed_masks(rows, forward, backward):
    additions = {}
    for index in forward.keys() | backward.keys():
        directions = [source[index] for source in (forward, backward) if index in source]
        if len(directions) == 2 and agreeing(directions[0]['mask'], directions[1]['mask']):
            selected, basis = directions[0], 'independent_forward_backward_masks'
        else:
            endpoint = [item for item in directions if item['endpoint_confirmed']]
            if len(endpoint) == 2:
                continue
            if endpoint:
                selected, basis = endpoint[0], 'propagation_reaches_measured_endpoint'
            else:
                matches = [item for item in directions if item['track_length'] >= CONFIG['min_confirmed_track_frames'] and any(
                    np.linalg.norm(np.asarray(item['mask']['point'])-candidate[:2]) <= CONFIG['candidate_error_px']
                    for candidate in candidates(rows[index]))]
                if len(matches) != 1:
                    continue
                selected, basis = matches[0], 'independent_detector_candidate'
        additions[str(index)] = dict(position=selected['mask']['point'], mask=selected['mask'], basis=basis,
                                     anchors=[item['anchor'] for item in directions])
    return additions


def run(args):
    import cv2
    import torch
    cache, output = private_path(args.cache), private_path(args.out)
    rows = read_rows(cache/'detections.jsonl')
    jobs = missing_intervals(rows)
    if args.max_gaps is not None:
        jobs = jobs[:args.max_gaps]
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((cache/'config.json').read_text())
    signature = dict(source_sha256=digest(cache/'detections.jsonl'), code_sha256=digest(__file__), config=CONFIG,
                     max_gaps=args.max_gaps, model_sha256=digest(ROOT/'outputs/sam2-models/sam2.1_hiera_tiny.pt'))
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text()) != signature:
        raise ValueError('Resume plan differs')
    atomic(output/'plan.json', signature)
    records = []
    if (output/'jobs.jsonl').exists():
        lines = (output/'jobs.jsonl').read_bytes().splitlines(keepends=True)
        records = [json.loads(line) for line in lines if line.endswith(b'\n')]
        if lines and not lines[-1].endswith(b'\n'):
            (output/'jobs.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
    done = {record['job_index']:record for record in records}
    predictor = runtime(args.device)
    cap = CachedVideoCapture(config['source'])
    if not cap.isOpened():
        raise ValueError('Cannot open source video')
    cv2.setNumThreads(2)
    started = time.monotonic()
    try:
        with (output/'jobs.jsonl').open('a') as stream:
            for number, job in enumerate(jobs):
                if number in done:
                    continue
                forward, first = propagate_direction(predictor, rows, job, False, cap, output)
                backward, second = propagate_direction(predictor, rows, job, True, cap, output)
                additions = confirmed_masks(rows, forward, backward)
                additions = {key:value for key,value in additions.items() if is_target(rows[int(key)])}
                record = dict(job_index=number, gap=[rows[job['lo']]['frame'], rows[job['hi']]['frame']],
                              attempts=first+second, additions=additions)
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                done[number] = record
                status = dict(state='running', completed=len(done), total=len(jobs),
                              supported_frames=sum(len(r['additions']) for r in done.values()),
                              elapsed_seconds=time.monotonic()-started)
                atomic(output/'status.json', status)
                print(json.dumps(status), flush=True)
                gc.collect()
                if args.device == 'mps':
                    torch.mps.empty_cache()
    finally:
        cap.release()
    result = copy.deepcopy(rows)
    for record in done.values():
        for key, addition in record['additions'].items():
            index = int(key)
            result[index].update(context_position=addition['position'], context_source='sam2_extended_verified',
                                 sam2_mask=addition['mask'], sam2_extension=addition)
    summary = dict(frames=len(rows), intervals=len(jobs), supported_frames=sum(len(r['additions']) for r in done.values()),
                   newly_filled=int(sum(not finite(a.get('context_position')) and finite(b.get('context_position')) for a,b in zip(rows,result))),
                   replaced_estimates=sum(a.get('context_source') in ESTIMATED_SOURCES and b.get('context_source') == 'sam2_extended_verified' for a,b in zip(rows,result)),
                   confirmation=dict(Counter(a['basis'] for record in done.values() for a in record['additions'].values())),
                   seed_outcomes=dict(Counter(a.get('seed',{}).get('reason',a.get('reason')) for record in done.values() for a in record['attempts'])),
                   elapsed_seconds=time.monotonic()-started, accuracy_measured=False)
    config['sam2_extension'] = signature
    atomic(output/'config.json', config)
    atomic(output/'summary.json', summary)
    (output/'detections.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in result))
    atomic(output/'status.json', dict(state='complete', **summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-gaps', type=int)
    run(parser.parse_args())
