"""YOLO + ByteTrack -> private, frame-aligned NPZ features and heuristic baseline.

Team IDs are color clusters, not home/away identities. Coordinates/distances use
image width/height, velocities use seconds. This is an offline, noncausal stage:
whole-video color clustering, bounded interpolation and centered median filtering.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from cv_pixel_preprocessing import detector_image

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_RUNTIME = ROOT/'outputs/two-stage-cv-runtime'
if OPTIONAL_RUNTIME.exists():
    sys.path.append(str(OPTIONAL_RUNTIME))

FEATURE_NAMES = ['ball_x', 'ball_y', 'ball_vx', 'ball_vy', 'ball_ax', 'ball_ay',
    'ball_speed', 'ball_angle_change', 'ball_visible', 'team0_d1', 'team0_d2',
    'team1_d1', 'team1_d2', 'team0_closing', 'team1_closing', 'team0_density',
    'team1_density', 'team0_centroid_x', 'team0_centroid_y',
    'team1_centroid_x', 'team1_centroid_y']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def private_path(path):
    import subprocess
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Generated data must remain under ignored outputs/')
    if subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd=ROOT).returncode:
        raise ValueError('Generated data must be Git-ignored')
    return path


def torso_color(image, box):
    """Median CIELAB of the inner 40% width/height, slightly above box center."""
    import cv2
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    xa, xb = max(0, int(x1+.3*w)), min(image.shape[1], int(x1+.7*w))
    ya, yb = max(0, int(y1+.2*h)), min(image.shape[0], int(y1+.6*h))
    if xb <= xa or yb <= ya:
        return None
    lab = cv2.cvtColor(image[ya:yb, xa:xb].astype(np.float32)/255., cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0)


def interpolate_ball(ball, visible, segments, max_gap=5):
    """Fill only bounded gaps with direct observations on both sides, within a shot."""
    result = np.asarray(ball, float).copy()
    for i in range(1, len(result)):
        if visible[i] or not visible[i-1]:
            continue
        j = i
        while j < len(result) and not visible[j] and segments[j] == segments[i-1]:
            j += 1
        if (j < len(result) and visible[j] and segments[j] == segments[i-1]
                and j-i <= max_gap):
            result[i:j] = np.linspace(result[i-1], result[j], j-i+2)[1:-1]
    return result


def derivative(values, fps, segments):
    out = np.full_like(values, np.nan, dtype=float)
    good = np.isfinite(values[1:]).all(axis=1) & np.isfinite(values[:-1]).all(axis=1)
    good &= segments[1:] == segments[:-1]
    out[1:][good] = (values[1:][good]-values[:-1][good])*fps
    return out


def cluster_teams(colors, outlier_distance=None):
    if len(colors) < 2:
        return {key: -1 for key in colors}, np.full((2, 3), np.nan)
    ids = sorted(colors)
    x = np.array([np.median(colors[k], axis=0) for k in ids])
    if len(np.unique(x, axis=0)) < 2:
        return {key: -1 for key in colors}, np.full((2, 3), np.nan)
    model = KMeans(n_clusters=2, random_state=0, n_init=10).fit(x)
    # Stable, explicitly arbitrary IDs; no match labels determine this ordering.
    order = sorted(range(2), key=lambda i: tuple(model.cluster_centers_[i]))
    mapping = {old: new for new, old in enumerate(order)}
    teams = {key: mapping[int(label)] for key, label in zip(ids, model.labels_)}
    if outlier_distance is not None:
        for key, value, label in zip(ids, x, model.labels_):
            if np.linalg.norm(value-model.cluster_centers_[label]) > outlier_distance:
                teams[key] = -1
    return teams, model.cluster_centers_[order]


def extract_features(rows, ball, visible, teams, width, height, fps, segments,
                     radius_px=100, control_px=60, carry_frames=5, median_window=5):
    scale = np.array([width, height], float)
    xy = ball/scale
    vel = derivative(xy, fps, segments)
    acc = derivative(vel, fps, segments)
    speed = np.linalg.norm(vel, axis=1)
    angle = np.full(len(ball), np.nan)
    for i in range(1, len(ball)):
        if (segments[i] == segments[i-1] and np.isfinite(vel[i-1:i+1]).all()
                and speed[i] > 1e-8 and speed[i-1] > 1e-8):
            delta = np.arctan2(vel[i, 1], vel[i, 0])-np.arctan2(vel[i-1, 1], vel[i-1, 0])
            angle[i] = np.arctan2(np.sin(delta), np.cos(delta))
    features = np.full((len(rows), len(FEATURE_NAMES)), np.nan, np.float32)
    features[:, :9] = np.column_stack([xy, vel, acc, speed, angle, visible])
    baseline = np.full(len(rows), 2, np.int64)
    previous = {}
    last_owner, last_contact = 2, -10**9
    for i, people in enumerate(rows):
        if i == 0 or segments[i] != segments[i-1]:
            previous, last_owner, last_contact = {}, 2, -10**9
        current, nearest = {}, []
        for team in (0, 1):
            team_rows = [(tid, np.array([(box[0]+box[2])/2, box[3]]))
                         for tid, box in people if teams.get(tid, -1) == team]
            for tid, foot in team_rows:
                current[tid] = foot
            if team_rows:
                features[i, 17+team*2:19+team*2] = np.mean([p for _, p in team_rows], axis=0)/scale
            if not np.isfinite(ball[i]).all():
                features[i, 15+team] = np.nan
                continue
            features[i, 15+team] = 0
            if not team_rows:
                continue
            distances = sorted((float(np.linalg.norm((foot-ball[i])/scale)), tid, foot)
                               for tid, foot in team_rows)
            for k, (distance, _, _) in enumerate(distances[:2]):
                features[i, 9+team*2+k] = distance
            features[i, 15+team] = sum(np.linalg.norm(foot-ball[i]) <= radius_px for _, foot in team_rows)
            _, tid, foot = distances[0]
            delta = (ball[i]-foot)/scale
            distance = np.linalg.norm(delta)
            if tid in previous and np.isfinite(vel[i]).all() and distance > 1e-8:
                pv = (foot-previous[tid])/scale*fps
                features[i, 13+team] = np.dot(pv-vel[i], delta/distance)
            # Pixel threshold and nearest player are computed in the same units.
            closest_px = min(float(np.linalg.norm(foot-ball[i])) for _, foot in team_rows)
            nearest.append((closest_px, team))
        previous = current
        if nearest and min(nearest)[0] <= control_px:
            last_owner, last_contact = min(nearest)[1], i
            baseline[i] = last_owner
        elif i-last_contact <= carry_frames:
            baseline[i] = last_owner
    smoothed = baseline.copy()
    half = median_window//2
    for i in range(len(baseline)):
        lo, hi = max(0, i-half), min(len(baseline), i+half+1)
        local = baseline[lo:hi][segments[lo:hi] == segments[i]]
        # Pick an actual ordinal median, including for even edge windows.
        smoothed[i] = np.sort(local)[(len(local)-1)//2]
    return features, baseline, smoothed


def load_labels(path, frame_ids, supplied):
    from path_guard import check_paths
    labels = np.full(len(frame_ids), -100, np.int64)
    if path is None:
        return labels
    if supplied:
        raise ValueError('Supplied-match labels cannot enter the feature/training pipeline')
    check_paths([path])
    lookup = {int(frame): i for i, frame in enumerate(frame_ids)}
    seen = set()
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['frame', 'label']:
            raise ValueError('Labels must have frame,label columns with cluster-aligned 0/1/2 labels')
        for row in reader:
            frame, label = int(row['frame']), int(row['label'])
            if frame in seen or label not in (0, 1, 2, -100):
                raise ValueError('Invalid or duplicate ground-truth label')
            seen.add(frame)
            if frame in lookup:
                labels[lookup[frame]] = label
    return labels


def run(args):
    settings_dir = ROOT/'outputs/two-stage-cv-settings'
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings_dir)
    os.environ['YOLO_AUTOINSTALL'] = 'False'
    import cv2
    import supervision as sv
    from ultralytics import YOLO
    from ultralytics import settings
    settings.update({'sync': False})
    from path_guard import forbidden_reason
    source = args.video.resolve()
    supplied = forbidden_reason(source) is not None
    out = private_path(args.out)
    if out.suffix != '.npz' or out.exists():
        raise ValueError('Use a new .npz output under outputs/')
    if args.labels and supplied:
        raise ValueError('No supplied-match labels are allowed')
    cap = cv2.VideoCapture(str(source))
    fps, width, height, total = (cap.get(k) for k in
        [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT])
    if not cap.isOpened() or not np.isfinite([fps, width, height, total]).all() or min(fps, width, height, total) <= 0:
        cap.release()
        raise ValueError('Video metadata is invalid')
    width, height, total = int(width), int(height), int(total)
    count = total-args.start_frame if args.max_frames is None else args.max_frames
    if args.start_frame < 0 or count < 1 or args.start_frame+count > total:
        raise ValueError('Requested frames are outside video')
    frame_ids = np.arange(args.start_frame, args.start_frame+count)
    labels = load_labels(args.labels, frame_ids, supplied)
    model_path = Path(args.model)
    if len(model_path.parts) == 1 and not model_path.exists():
        model_path = ROOT/'outputs/two-stage-cv-models'/model_path.name
        model_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    if model.names.get(0) != 'person' or model.names.get(32) != 'sports ball':
        raise ValueError('Expected a COCO model with person=0, sports ball=32')
    player_tracker = sv.ByteTrack(frame_rate=max(1, round(fps)), minimum_consecutive_frames=1)
    ball_tracker = sv.ByteTrack(frame_rate=max(1, round(fps)), minimum_consecutive_frames=1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    rows, colors, raw_ball, visible, segments, ball_ids = [], defaultdict(list), [], [], [], []
    previous_thumb, segment = None, 0
    try:
        for i in range(count):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Decode ended before the requested last frame')
            thumb = cv2.resize(image, (64, 36)).astype(float)
            if previous_thumb is not None and np.mean(np.abs(thumb-previous_thumb))/255 > args.cut_threshold:
                segment += 1
                player_tracker.reset()
                ball_tracker.reset()
            previous_thumb = thumb
            result = model.predict(detector_image(image, getattr(args, 'preprocess', 'none')), classes=[0, 32], conf=args.confidence,
                imgsz=args.imgsz, device=args.device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            people = player_tracker.update_with_detections(detections[detections.class_id == 0])
            balls_raw = detections[detections.class_id == 32]
            balls = ball_tracker.update_with_detections(balls_raw)
            frame_people = []
            for box, tid in zip(people.xyxy, people.tracker_id):
                if tid < 0:
                    continue
                # Track IDs can restart at scene cuts; namespace them by shot.
                key = segment*1_000_000_000+int(tid)
                frame_people.append((key, box.astype(float)))
                color = torso_color(image, box)
                if color is not None:
                    colors[key].append(color)
            rows.append(frame_people)
            # Only direct observations count as visible; tracker predictions do not.
            if len(balls_raw):
                j = int(np.argmax(balls_raw.confidence))
                box = balls_raw.xyxy[j]
                raw_ball.append([(box[0]+box[2])/2, (box[1]+box[3])/2])
                visible.append(1)
                matching = np.where(np.all(np.isclose(balls.xyxy, box), axis=1))[0]
                ball_ids.append(segment*1_000_000_000+int(balls.tracker_id[matching[0]]) if len(matching) else -1)
            else:
                raw_ball.append([np.nan, np.nan])
                visible.append(0)
                ball_ids.append(-1)
            segments.append(segment)
            if (i+1) % 100 == 0:
                print(f'{i+1}/{count} frames', flush=True)
    finally:
        cap.release()
    segments, visible = np.array(segments), np.array(visible, np.uint8)
    teams, centers = cluster_teams(colors, args.color_outlier_distance)
    ball = interpolate_ball(raw_ball, visible, segments, args.max_ball_gap)
    features, raw_baseline, baseline = extract_features(rows, ball, visible, teams,
        width, height, fps, segments, args.density_radius, args.control_threshold,
        args.carry_frames, args.median_window)
    offsets = np.cumsum([0]+[len(row) for row in rows])
    flat = [(i, tid, box) for i, row in enumerate(rows) for tid, box in row]
    metadata = dict(schema=1, source=str(source), source_sha256=digest(source),
        supplied_match=supplied, training_eligible=not supplied, fps=fps,
        width=width, height=height, start_frame=args.start_frame,
        model=str(model.ckpt_path), model_sha256=digest(model.ckpt_path),
        code_sha256=digest(__file__), team_semantics='CIELAB centers sorted lexicographically; not H/A',
        detector_preprocessing=getattr(args, 'preprocess', 'none'),
        preprocessing_code_sha256=digest(Path(__file__).with_name('cv_pixel_preprocessing.py')),
        torso_color_source='original pixels', scene_cut_source='original pixels',
        none_class=2, unlabeled=-100, offline=True,
        configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        versions={name: __import__('importlib.metadata', fromlist=['version']).version(name)
                  for name in ['ultralytics', 'supervision', 'scikit-learn', 'opencv-python', 'torch']})
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=features, feature_names=np.array(FEATURE_NAMES),
        frame_ids=frame_ids, segments=segments, ball_xy=ball, ball_visible=visible,
        ball_track_ids=np.array(ball_ids), labels=labels, baseline=baseline,
        baseline_raw=raw_baseline, player_offsets=offsets,
        player_track_ids=np.array([tid for _, tid, _ in flat], np.int64),
        player_boxes=np.array([box for _, _, box in flat], np.float32).reshape(-1, 4),
        player_teams=np.array([teams.get(tid, -1) for _, tid, _ in flat], np.int8),
        team_centers_lab=centers, metadata=np.array(json.dumps(metadata)))
    print(json.dumps(dict(output=str(out), frames=count, features=len(FEATURE_NAMES),
        directly_visible_ball_frames=int(visible.sum()), training_eligible=not supplied)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='yolo11x.pt')
    p.add_argument('--preprocess', choices=['none','zero-green'], default='none',
                   help='Applied before YOLO; zero-green sets G=0 and preserves R/B')
    p.add_argument('--device', default=None, help='Ultralytics device, e.g. cpu, mps, 0')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--confidence', type=float, default=.1, help='Low-confidence candidates for ByteTrack')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--max-ball-gap', type=int, default=5)
    p.add_argument('--control-threshold', type=float, default=60, help='Pixels')
    p.add_argument('--density-radius', type=float, default=100, help='Pixels')
    p.add_argument('--carry-frames', type=int, default=5)
    p.add_argument('--median-window', type=int, default=5)
    p.add_argument('--cut-threshold', type=float, default=.22)
    p.add_argument('--color-outlier-distance', type=float)
    p.add_argument('--labels', type=Path, help='Other-match CSV: frame,label, labels aligned to color clusters')
    args = p.parse_args()
    if (args.max_ball_gap < 0 or args.carry_frames < 0 or args.median_window < 1
            or args.median_window % 2 == 0 or args.imgsz < 32
            or not 0 < args.confidence <= 1 or not 0 < args.cut_threshold <= 1
            or args.control_threshold <= 0 or args.density_radius <= 0
            or (args.color_outlier_distance is not None and args.color_outlier_distance <= 0)):
        p.error('Invalid threshold, gap, image size or odd median window')
    run(args)


if __name__ == '__main__':
    main()
