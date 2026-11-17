"""Label-blind, same-camera ATD/CV fusion. Never fits or reads answers.

Keep the cached CV observations unchanged; attach ATD team evidence frame by
frame and add metric tracking features, independent source masks and residuals.
ATD is primary on disagreement, which remains explicitly flagged, not averaged.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from stage1_cv_baseline import ROOT, FEATURE_NAMES, digest, private_path, extract_features

sys.path.insert(0, str(ROOT/'challenge'))
from games import GAME, GAMES, asset_paths, half_video_blocks
from sequence_decoder import team_of_each_track

EXTRA_NAMES = ['tracking_'+n for n in FEATURE_NAMES] + [
    'cv_ball_direct', 'atd_ball_direct', 'fused_ball_observed',
    'homography_usable', 'ball_sources_disagree', 'ball_residual_m',
    'cv_team_matched_fraction', 'atd_player_count', 'cv_player_count',
    'fused_ball_from_atd', 'fused_ball_from_cv', 'ball_interpolated']
FUSED_NAMES = FEATURE_NAMES + EXTRA_NAMES


def transform(points, matrix):
    p = np.asarray(points, float).reshape(-1, 2)
    out = np.full_like(p, np.nan)
    h = np.c_[p, np.ones(len(p))] @ matrix.T
    valid = np.isfinite(h).all(1) & (np.abs(h[:, 2]) > 1e-9)
    out[valid] = h[valid, :2]/h[valid, 2, None]
    return out


def read_atd(raw, frames, slots):
    players = np.full((len(frames), slots, 2), np.nan)
    ball = np.full((len(frames), 2), np.nan)
    count = 0
    with Path(raw).open() as stream:
        for line in stream:
            parts = line.strip().split(':')
            frame = int(parts[0])-1
            if frame < frames[0]: continue
            if frame > frames[-1]: break
            if frame != frames[count] or len(parts) != 3:
                raise ValueError('ATD frame alignment failed')
            groups = parts[1].split(';')
            if len(groups) != slots: raise ValueError('ATD slot mismatch')
            players[count] = [[float(v) for v in g.split(',')[:2]] for g in groups]
            ball[count] = [float(v) for v in parts[2].split(',')[:2]]
            count += 1
    if count != len(frames) or np.isinf(players).any() or np.isinf(ball).any():
        raise ValueError('Incomplete or invalid ATD window')
    return players, ball


def match_teams(boxes, projected, teams, gate_heights=.75):
    """One-to-one current-frame geometric match; no slot identity persistence."""
    assigned = np.full(len(boxes), -1, np.int8)
    slots = np.full(len(boxes), -1, np.int64)
    if not len(boxes) or not len(projected): return assigned, slots
    feet = np.c_[(boxes[:, 0]+boxes[:, 2])/2, boxes[:, 3]]
    heights = np.maximum(10., boxes[:, 3]-boxes[:, 1])
    cost = np.linalg.norm(feet[:, None]-projected[None], axis=2)/heights[:, None]
    cost[~np.isfinite(cost)] = 1e6
    a, b = linear_sum_assignment(cost)
    for i, j in zip(a, b):
        if cost[i, j] <= gate_heights:
            assigned[i], slots[i] = teams[j], j
    return assigned, slots


def fuse_ball(atd, cv, agreement_m=2.):
    has_atd, has_cv = np.isfinite(atd).all(), np.isfinite(cv).all()
    residual = float(np.linalg.norm(atd-cv)) if has_atd and has_cv else np.nan
    # Avoid uncertain geometric averages, especially for airborne CV detections.
    return (atd.copy() if has_atd else cv.copy()), dict(
        atd=has_atd, cv=has_cv, disagree=bool(has_atd and has_cv and residual>agreement_m),
        residual=residual, from_atd=has_atd, from_cv=bool(has_cv and not has_atd))


def run(args):
    out = private_path(args.out)
    if out.exists(): raise ValueError('Preserve previous outputs; use a new path')
    with np.load(args.features, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    meta = json.loads(str(data['metadata']))
    paths = asset_paths(data_dir=str(ROOT/'data'), source='tactical')
    if Path(meta['source']).resolve() != Path(paths['video']).resolve():
        raise ValueError('Fusion requires the same tactical camera; broadcast is not synchronized')
    if np.any(data['labels'] != -100): raise ValueError('Fusion accepts label-free observations only')
    frames = data['frame_ids']
    if not len(frames) or not np.all(np.diff(frames)==1): raise ValueError('Nonconsecutive CV frames')
    teams = np.array([0 if t=='H' else 1 for t in team_of_each_track(paths['atd_meta'])])
    p, b = read_atd(paths['atd_raw'], frames, len(teams))
    pitch = np.array(GAMES[GAME]['pitch'])
    p, b = p*pitch, b*pitch
    matrices = np.full((len(frames), 3, 3), np.nan)
    errors = np.full(len(frames), np.nan)
    with Path(paths['homography']).open() as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame'])
            if frame < frames[0]: continue
            if frame > frames[-1]: break
            i = frame-int(frames[0])
            errors[i] = float(row['reprojection_error'])
            if int(row['valid']) == 1 and errors[i] <= 2.:
                matrices[i] = [[float(row[f'h{a}{c}']) for c in range(3)] for a in range(3)]
    segments = data['segments'].copy()
    # Enforce half boundaries in addition to cached camera cuts.
    halves = np.zeros(len(frames), int)
    for half, lo, hi in half_video_blocks(GAME):
        halves[(frames>=lo)&(frames<=hi)] = half
    segments = np.cumsum(np.r_[0, (np.diff(segments)!=0)|(np.diff(halves)!=0)])
    fused = np.full_like(b, np.nan)
    cv_metric = np.full_like(b, np.nan)
    tracking_screen = np.full_like(b, np.nan)
    cv_teams = np.full(len(data['player_boxes']), -1, np.int8)
    cv_slots = np.full(len(cv_teams), -1, np.int64)
    flags, screen_rows, metric_rows, metric_teams = [], [], [], {}
    direct_cv = data['ball_visible'].astype(bool)
    for i in range(len(frames)):
        a, end = data['player_offsets'][i:i+2]
        boxes = data['player_boxes'][a:end]
        h = matrices[i]
        usable = np.isfinite(h).all() and abs(np.linalg.det(h))>1e-12
        projected = transform(p[i]/pitch, h) if usable else np.full_like(p[i], np.nan)
        assigned, slots = match_teams(boxes, projected, teams)
        cv_teams[a:end], cv_slots[a:end] = assigned, slots
        if usable and direct_cv[i]:
            cv_metric[i] = transform([data['ball_xy'][i]], np.linalg.inv(h))[0]*pitch
            if not ((cv_metric[i]>=-3).all() and (cv_metric[i]<=pitch+3).all()):
                cv_metric[i] = np.nan
        fused[i], evidence = fuse_ball(b[i], cv_metric[i])
        if usable and np.isfinite(b[i]).all():
            tracking_screen[i] = transform([b[i]/pitch], h)[0]
        flags.append([direct_cv[i], evidence['atd'], np.isfinite(fused[i]).all(), usable,
            evidence['disagree'], evidence['residual'],
            float(np.mean(assigned>=0)) if len(assigned) else 0.,
            np.isfinite(p[i]).all(1).sum(), len(boxes), evidence['from_atd'], evidence['from_cv'], False])
        screen_row = []
        for j, (box, team) in enumerate(zip(boxes, assigned)):
            key = int(a+j)
            metric_teams[key+1_000_000_000] = int(team)
            screen_row.append((key+1_000_000_000, box))
        screen_rows.append(screen_row)
        metric_row = []
        for slot, pos in enumerate(p[i]):
            if np.isfinite(pos).all():
                # Zero-height internal boxes encode foot points for feature extraction only.
                metric_row.append((slot, [pos[0],pos[1],pos[0],pos[1]]))
                metric_teams[slot] = int(teams[slot])
        # Unmatched CV objects have unknown teams and cannot invent metric owners.
        metric_rows.append(metric_row)
    observed = np.isfinite(fused).all(1)
    # Recover only bounded missing runs, and keep interpolations explicitly masked.
    from stage1_cv_baseline import interpolate_ball
    recovered = interpolate_ball(fused, observed, segments, 5)
    flags = np.asarray(flags, float)
    flags[:, -1] = ~observed & np.isfinite(recovered).all(1)
    fused_screen = np.full_like(recovered, np.nan)
    for i, pos in enumerate(recovered):
        if flags[i, 3] and np.isfinite(pos).all():
            fused_screen[i] = transform([pos/pitch], matrices[i])[0]
    image_features, _, _ = extract_features(screen_rows, fused_screen, observed,
        metric_teams, meta['width'], meta['height'], meta['fps'], segments)
    # Scale=1 produces positions in metres and velocities in m/s, not pixel proxies.
    metric_features, raw, baseline = extract_features(metric_rows, recovered, observed,
        metric_teams, 1, 1, meta['fps'], segments, radius_px=5, control_px=2,
        carry_frames=5, median_window=5)
    features = np.c_[image_features, metric_features, flags].astype(np.float32)
    meta.update(schema=2, feature_mode='tactical_cv_fusion', supplied_match=True,
        training_eligible=False, team_semantics='Team 0 = Barcelona (H); Team 1 = Athletic Club (A)',
        label_names=['Barcelona (H)', 'Athletic Club (A)', 'Uncertain'],
        feature_source_sha256=digest(args.features), fusion_code_sha256=digest(__file__),
        fusion_input_sha256={k:digest(paths[k]) for k in ['atd_raw','atd_meta','homography']},
        fusion_config=dict(ball_agreement_m=2., match_gate_player_heights=.75,
            max_reprojection_error=2., max_ball_gap=5, control_m=2., density_m=5.),
        baseline_kind='metric nearest-team heuristic; uncertainty is not dead ball',
        answers_read=False, fusion_policy='ATD primary; CV fallback; disagreements flagged',
        tracking_slot_identity='slot only, may change identity; never a named player')
    data.update(features=features, feature_names=np.array(FUSED_NAMES), segments=segments,
        cv_ball_xy=data['ball_xy'], cv_ball_visible=data['ball_visible'],
        cv_cluster_teams=data['player_teams'], cv_baseline=data['baseline'],
        player_teams=cv_teams, matched_atd_slots=cv_slots,
        ball_xy=fused_screen, ball_visible=observed.astype(np.uint8),
        baseline=baseline, baseline_raw=raw, atd_ball_xy=tracking_screen,
        fused_ball_m=recovered, atd_ball_m=b, cv_ball_m=cv_metric,
        atd_players_m=p, atd_slot_teams=teams, source_flags=flags,
        metadata=np.array(json.dumps(meta)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    print(json.dumps(dict(frames=len(frames), features=len(FUSED_NAMES),
        cv_ball_direct=int(direct_cv.sum()), atd_ball_direct=int(flags[:,1].sum()),
        fused_ball_observed=int(observed.sum()), interpolated=int(flags[:,-1].sum()),
        source_disagreements=int(flags[:,4].sum()),
        matched_cv_players=int((cv_teams>=0).sum()), cv_players=len(cv_teams),
        accuracy_measured=False)), flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
