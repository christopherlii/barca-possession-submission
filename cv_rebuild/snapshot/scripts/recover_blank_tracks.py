"""Recover actual blank positions from image-supported offline tracklets.

Works on every blank in a completed cache. Cached/fresh detections are associated
in camera-compensated windows. A path needs an accepted observation or repeated
tactical support. Persistent stationary markings, ambiguous paths and camera cuts
remain excluded. Short interpolations have explicit estimated provenance.
"""
from cached_video_reader import CachedVideoCapture
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
from local_cv_tiling import merge_points
from smooth_candidate_window import associate
from sam2_gap_recovery import observation, transport
from sam_confirmation_recovery import read_rows
from sam_confirmation_retry import crop_box
from field_marking_evidence import closest_marking, is_field_boundary


CONFIG = dict(window=48, stride=24, min_confidence=.15,
              max_candidates=8, guide_support_px=40., guide_support_fraction=.7,
              guide_score_sigma=96., stationary_step_px=3., marking_px=10.,
              contact_px=40., min_unanchored_travel=8., max_interpolation=5,
              crop_size=256, detail_size=128, max_crops=3, detector_size=640)


def finite(point):
    return point is not None and len(point) >= 2 and np.isfinite(point[:2]).all()


def trusted(row):
    if finite(row.get('blank_recovery_detection')):
        return np.asarray(row['blank_recovery_detection'][:2])
    point = observation(row)
    if point is not None:
        return point[:2]
    if row.get('sam2_mask') and finite(row.get('context_position')):
        return np.asarray(row['context_position'])
    return None


def blanks(rows):
    return [i for i, row in enumerate(rows) if not finite(row.get('context_position'))]


def search_targets(rows):
    """Search missing image evidence even when a display prediction exists.

    Predictions may guide a crop but must not postpone fresh detection until
    they expire, which can break an otherwise recoverable observation chain.
    """
    return [i for i, row in enumerate(rows) if trusted(row) is None]


def coverage(rows):
    missing = blanks(rows)
    return dict(blank_frames=len(missing), blank_stretches=sum(
        i == 0 or missing[i] != missing[i - 1] + 1 for i in range(len(missing))))


def clear_boundary_vetoes(rows):
    """Keep past vetoes auditable while removing them from the active policy."""
    cleared = 0
    for row in rows:
        removed = 0
        for evidence in row.get('marking_evidence') or []:
            name = evidence.get('marking')
            if name and is_field_boundary(name):
                was_rejected = bool(evidence.get('rejected'))
                evidence['previous_boundary_rejection'] = was_rejected or bool(evidence.get('previous_boundary_rejection'))
                removed += int(was_rejected)
                evidence.update(rejected=False, reason='boundary_filter_disabled')
        if removed:
            row['gate_rejected'] = max(0, row.get('gate_rejected',0)-removed)
        cleared += removed
    return cleared


def candidates(row):
    groups = []
    for key in ['candidates', 'crowded_zoom_candidates', 'player_context_candidates',
                'recovery_candidates', 'blank_recovery_candidates']:
        points = [p for p in (row.get(key) or [])
                  if len(p) == 3 and np.isfinite(p).all() and p[2] >= CONFIG['min_confidence']]
        groups.append(points)
    return merge_points(groups).tolist()


def window_bounds(rows):
    bounds = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i < len(rows) and rows[i].get('camera_affine') is not None and rows[i]['frame'] == rows[i - 1]['frame'] + 1:
            continue
        for lo in range(start, i, CONFIG['stride']):
            hi = min(i, lo + CONFIG['window'])
            if any(not finite(r.get('context_position')) for r in rows[lo:hi]):
                bounds.append((lo, hi))
        start = i
    return bounds


def recover(rows, player_boxes=None, ignore_field_boundary=False):
    """Pure association pass. Source observations remain immutable."""
    result = copy.deepcopy(rows)
    reports = []
    observations = {}
    estimates = {}
    for lo, hi in window_bounds(rows):
        poses = [np.eye(3)]
        for row in rows[lo + 1:hi]:
            affine = np.eye(3)
            affine[:2] = row['camera_affine']
            poses.append(affine @ poses[-1])
        samples, raw, anchors = [], [], {}
        for offset, row in enumerate(rows[lo:hi]):
            inv = np.linalg.inv(poses[offset])
            measured = trusted(row)
            points = candidates(row)
            if measured is not None:
                anchors[offset] = measured
                points = [[*measured, 1.]]
            points = sorted(points, key=lambda p: -p[2])[:CONFIG['max_candidates']]
            raw.append(points)
            weighted = []
            for point in points:
                score = point[2]
                if measured is None and finite(row.get('guide')):
                    distance = np.linalg.norm(np.asarray(point[:2]) - row['guide'][:2])
                    score *= .5 + .5 * np.exp(-.5 * (distance / CONFIG['guide_score_sigma']) ** 2)
                weighted.append([*(inv @ np.r_[point[:2], 1])[:2], score])
            samples.append(weighted)
        path, report = associate(samples)
        note = dict(first=lo, stop=hi, association=report, accepted=0)
        reports.append(note)
        if report['status'] != 'associated':
            continue
        image_path = {j: (poses[j] @ np.r_[p[:2], 1])[:2] for j, p in path}
        attached = any(j in anchors and np.linalg.norm(image_path[j] - anchors[j]) <= 8 for j in image_path)
        guided = [np.linalg.norm(point - rows[lo + j]['guide'][:2]) <= CONFIG['guide_support_px']
                  for j, point in image_path.items() if finite(rows[lo + j].get('guide'))]
        guide_supported = len(guided) >= 4 and np.mean(guided) >= CONFIG['guide_support_fraction']
        if not attached and not guide_supported:
            note['rejected'] = 'no_anchor_or_repeated_guide_support'
            continue
        steps = [np.linalg.norm(np.asarray(b[1][:2]) - a[1][:2]) / (b[0] - a[0]) for a, b in zip(path, path[1:])]
        stationary = bool(steps) and np.median(steps) < CONFIG['stationary_step_px']
        marking, contact = [], []
        for j, point in image_path.items():
            shapes = rows[lo + j].get('field_geometry') or []
            normalized = [dict(s, points=np.asarray(s['points'])) for s in shapes]
            marking.append(closest_marking(point, normalized, ignore_field_boundary)[0] <= CONFIG['marking_px'])
            boxes = np.asarray(player_boxes[lo + j] if player_boxes is not None else []).reshape(-1, 4)
            feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
            contact.append(bool(len(feet)) and np.linalg.norm(feet - point, axis=1).min() <= CONFIG['contact_px'])
        if stationary and np.mean(marking) >= .5 and np.mean(contact) < .5:
            note['rejected'] = 'stationary_marking'
            continue
        if not attached and np.linalg.norm(np.asarray(path[-1][1][:2]) - path[0][1][:2]) < CONFIG['min_unanchored_travel'] and np.mean(contact) < .5:
            note['rejected'] = 'unanchored_static_candidate'
            continue
        accepted = []
        for j, point in image_path.items():
            i = lo + j
            if finite(rows[i].get('context_position')):
                continue
            confidence = min(raw[j], key=lambda p: np.linalg.norm(np.asarray(p[:2]) - point))[2]
            observations.setdefault(i, []).append([*point.tolist(), confidence])
            accepted.append(i)
        # Interpolation is restricted to two observed nodes within this accepted path.
        for (ja, _), (jb, _) in zip(path, path[1:]):
            if jb - ja - 1 > CONFIG['max_interpolation']:
                continue
            for j in range(ja + 1, jb):
                i = lo + j
                if finite(rows[i].get('context_position')):
                    continue
                u = (j - ja) / (jb - ja)
                a = np.linalg.solve(poses[ja], np.r_[image_path[ja], 1])[:2]
                b = np.linalg.solve(poses[jb], np.r_[image_path[jb], 1])[:2]
                point = (poses[j] @ np.r_[(1 - u) * a + u * b, 1])[:2]
                estimates.setdefault(i, []).append(dict(point=point.tolist(), anchors=[lo + ja, lo + jb]))
        note.update(accepted=len(accepted), attached=attached, guide_supported=bool(guide_supported))
    conflicting = set()
    for i, points in observations.items():
        best = max(points, key=lambda p: p[2])
        if any(np.linalg.norm(np.asarray(p[:2]) - best[:2]) > 12 for p in points):
            conflicting.add(i)
            continue
        result[i].update(context_position=best[:2], context_source='blank_recovery_observation',
                         blank_recovery_detection=best)
    for i, entries in estimates.items():
        if i in conflicting or finite(result[i].get('context_position')):
            continue
        valid = [e for e in entries if all(finite(result[k].get('context_position')) for k in e['anchors'])]
        if valid and all(np.linalg.norm(np.asarray(e['point']) - valid[0]['point']) <= 12 for e in valid):
            result[i].update(context_position=valid[0]['point'], context_source='blank_recovery_interpolated')
    # Resolve incompatible touching windows by abstaining on the newly added points.
    invalid = set()
    for i in range(1, len(result)):
        a, b = result[i - 1], result[i]
        if not finite(a.get('context_position')) or not finite(b.get('context_position')) or b.get('camera_affine') is None:
            continue
        expected = transport(a['context_position'], rows, i - 1, i)
        if np.linalg.norm(expected - b['context_position']) > 45:
            invalid.update(k for k in (i - 1, i) if not finite(rows[k].get('context_position')))
    for i in invalid:
        result[i] = copy.deepcopy(rows[i])
    return result, reports


def search_plans(rows, i, width, height):
    row = rows[i]
    centers = []
    if finite(row.get('guide')):
        centers.append(('tactical', np.asarray(row['guide'][:2])))
    if trusted(row) is None and finite(row.get('context_position')):
        centers.append(('prediction_search', np.asarray(row['context_position'][:2])))
    left = next((k for k in range(i - 1, max(-1, i - 151), -1) if trusted(rows[k]) is not None), None)
    right = next((k for k in range(i + 1, min(len(rows), i + 151)) if trusted(rows[k]) is not None), None)
    if left is not None and right is not None and all(rows[k].get('camera_affine') is not None for k in range(left + 1, right + 1)):
        a = transport(trusted(rows[left]), rows, left, i)
        b = transport(trusted(rows[right]), rows, right, i)
        u = (i - left) / (right - left)
        centers.append(('endpoint_search', (1 - u) * a + u * b))
    points = candidates(row)
    if points:
        centers.append(('cached_candidate', np.asarray(points[0][:2])))
    plans = []
    for source, center in centers:
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        box = crop_box(center, CONFIG['crop_size'], width, height)
        if not any(np.linalg.norm(center - p['center']) < 96 for p in plans):
            plans.append(dict(source=source, center=center.tolist(), box=box, mode='raw'))
        if len(plans) >= CONFIG['max_crops']:
            break
    if plans and len(plans) < CONFIG['max_crops']:
        center = plans[0]['center']
        plans.append(dict(source='detail', center=center, box=crop_box(center, CONFIG['detail_size'], width, height), mode='clahe'))
    return plans or [dict(source='full_frame', box=[0, 0, width, height], mode='raw')]


def run(args):
    import cv2
    from frame_preprocessing_experiment import preprocess, detect
    cv2.setNumThreads(2)
    out = private_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.cache / 'detections.jsonl')
    config = json.loads((args.cache / 'config.json').read_text())
    ignore_field_boundary = getattr(args,'ignore_field_boundary',False) or config.get('field_gate',{}).get('ignore_field_boundary',False)
    use_memory = getattr(args, 'distractor_memory', False)
    if use_memory and (config.get('distractor_memory') or config.get('continuous_trajectory')):
        raise ValueError('Apply memory once, on an unfilled detection cache')
    if config.get('distractor_memory') and (args.fresh or args.overlay):
        raise ValueError('Fresh searches and overlays require the unfiltered input with --distractor-memory')
    overlays = args.overlay or []
    for overlay in overlays:
        extra = read_rows(overlay / 'detections.jsonl')
        overlay_config = json.loads((overlay/'config.json').read_text())
        if overlay_config.get('sam2_extension'):
            config['sam2_extension'] = overlay_config['sam2_extension']
        if [r['frame'] for r in extra] != [r['frame'] for r in rows]:
            raise ValueError('Overlay frame alignment differs')
        for row, newer in zip(rows, extra):
            if newer.get('sam2_mask') and not row.get('sam2_mask') and trusted(row) is None:
                for key in ['context_position', 'context_source', 'sam2_mask',
                            'sam2_flow_evidence', 'sam2_confirmation_retry', 'sam2_extension']:
                    if key in newer:
                        row[key] = newer[key]
    with np.load(args.features, allow_pickle=False) as features:
        if features['frame_ids'].tolist() != [r['frame'] for r in rows]:
            raise ValueError('Feature alignment differs')
        offsets, boxes = features['player_offsets'], features['player_boxes']
        player_boxes = [boxes[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    pin = dict(source=digest(args.cache / 'detections.jsonl'), features=digest(args.features),
               overlays=[dict(path=str(p), sha256=digest(p / 'detections.jsonl')) for p in overlays],
               implementation=digest(__file__), fresh=args.fresh, config=CONFIG,
               ignore_field_boundary=ignore_field_boundary,
               distractor_memory=use_memory,
               memory_code_sha256=digest(Path(__file__).with_name('distractor_memory.py')) if use_memory else None)
    plan_path = out / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(pin)):
        raise ValueError('Resume plan differs')
    atomic(plan_path, pin)
    started = time.monotonic()
    targets = search_targets(rows)
    searches = {}
    log_path = out / 'searches.jsonl'
    if log_path.exists():
        searches = {r['index']: r for r in read_rows(log_path)}
    if args.fresh:
        import torch
        os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'outputs/two-stage-cv-settings')
        os.environ['YOLO_AUTOINSTALL'] = 'False'
        from ultralytics import YOLO
        torch.set_num_threads(2)
        model = YOLO(ROOT / 'outputs/two-stage-cv-models/yolo11x.pt')
        cap = CachedVideoCapture(config['source'])
        try:
            with log_path.open('a') as handle:
                for n, i in enumerate(targets):
                    if i in searches:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, rows[i]['frame'])
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Video decode failed')
                    plans = search_plans(rows, i, image.shape[1], image.shape[0])
                    found = []
                    for plan in plans:
                        x, y, xx, yy = plan['box']
                        crop = preprocess(image[y:yy, x:xx], plan['mode'])
                        size = 1280 if plan['source'] == 'full_frame' else CONFIG['detector_size']
                        points = detect(model, crop, args.device, size)
                        found.append([[p[0] + x, p[1] + y, p[2]] for p in points])
                    record = dict(index=i, plans=plans, candidates=merge_points(found).tolist())
                    handle.write(json.dumps(record) + '\n')
                    handle.flush()
                    searches[i] = record
                    atomic(out / 'status.json', dict(state='detecting', completed=len(searches), total=len(targets), elapsed_seconds=time.monotonic() - started))
                    if (n + 1) % 100 == 0:
                        print(f'Fresh blank-frame searches {n + 1}/{len(targets)}', flush=True)
        finally:
            cap.release()
    for i, record in searches.items():
        rows[i]['blank_recovery_candidates'] = record['candidates']
    if use_memory:
        from dataclasses import asdict
        from distractor_memory import MemoryConfig, filter_video, summarize
        memory_config = MemoryConfig()
        filtered = filter_video(rows, config['source'], memory_config)
        config['distractor_memory'] = dict(config=asdict(memory_config),
            code_sha256=pin['memory_code_sha256'], automatic_pseudolabels=True,
            summary=summarize(rows, filtered))
        rows = filtered
    result, reports = recover(rows, player_boxes, ignore_field_boundary)
    before, after = coverage(rows), coverage(result)
    added = [b for a, b in zip(rows, result) if not finite(a.get('context_position')) and finite(b.get('context_position'))]
    summary = dict(before=before, after=after, newly_filled_frames=len(added),
                   new_sources=dict(Counter(r['context_source'] for r in added)),
                   windows=len(reports), accepted_windows=sum(r['accepted'] > 0 for r in reports),
                   rejection_reasons=dict(Counter(r.get('rejected', r['association']['status']) for r in reports if not r['accepted'])),
                   elapsed_seconds=time.monotonic() - started, accuracy_measured=False)
    if ignore_field_boundary:
        summary['disabled_boundary_vetoes'] = clear_boundary_vetoes(result)
    if any(a.get('context_position') != b.get('context_position') for a, b in zip(rows, result) if finite(a.get('context_position'))):
        raise AssertionError('An existing position changed')
    config['blank_track_recovery'] = dict(pin, input_cache=str(args.cache), fresh=args.fresh)
    config['field_gate'] = dict(config.get('field_gate',{}), ignore_field_boundary=ignore_field_boundary)
    atomic(out / 'config.json', config)
    (out / 'detections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in result))
    (out / 'association.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in reports))
    atomic(out / 'summary.json', summary)
    atomic(out / 'status.json', dict(state='complete', **summary))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--distractor-memory', action='store_true',
                        help='Apply expiring image-based negative memory before association (also filters fresh detections)')
    parser.add_argument('--ignore-field-boundary', action='store_true', help='Ignore touchline/corner distractors while retaining interior marking and penalty-spot checks')
    run(parser.parse_args())
