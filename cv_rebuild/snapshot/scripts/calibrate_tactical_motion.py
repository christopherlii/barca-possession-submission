"""Estimate empirical motion envelopes from consecutive automatic tactical tracks.

No possession labels are read. A local quadratic fit differentiates observed
metric positions only. The resulting limits describe the automatic 2D track,
including its measurement errors and ground-projection errors for aerial balls.
They are statistical envelopes, not physical maxima or confidence intervals.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from stage1_cv_baseline import ROOT, digest, private_path
from local_cv_possession import atomic

sys.path.insert(0, str(ROOT / 'challenge'))
from games import GAME, GAMES, half_video_blocks


def estimate_motion(points_m, eligible, fps, window=7, quantile=.995, margin=1.25):
    """Differentiate complete windows and robustly reject poor quadratic fits."""
    points = np.asarray(points_m, float)
    eligible = np.asarray(eligible, bool) & np.isfinite(points).all(axis=1)
    if points.ndim != 2 or points.shape[1] != 2 or eligible.shape != (len(points),):
        raise ValueError('Expected aligned (N, 2) metric positions and (N,) mask')
    if fps <= 0 or window < 5 or window % 2 != 1 or len(points) < window:
        raise ValueError('Need positive fps and an odd window of at least five frames')
    if not 0 < quantile < 1 or margin < 1:
        raise ValueError('Quantile must be in (0, 1) and margin at least one')
    windows = np.lib.stride_tricks.sliding_window_view(points, window, axis=0).transpose(0, 2, 1)
    complete = np.lib.stride_tricks.sliding_window_view(eligible, window).all(axis=1)
    windows = windows[complete]
    if not len(windows):
        raise ValueError('No complete observed windows for calibration')
    times = np.arange(-(window // 2), window // 2 + 1) / fps
    design = np.column_stack((np.ones(window), times, times**2))
    coefficients = np.einsum('ij,njk->nik', np.linalg.pinv(design), windows)
    fitted = np.einsum('ij,njk->nik', design, coefficients)
    residual = np.sqrt(np.mean((windows - fitted)**2, axis=(1, 2)))
    median = np.median(residual)
    residual_limit = max(.15, float(median + 6 * 1.4826 * np.median(abs(residual - median))))
    keep = residual <= residual_limit
    speed = np.linalg.norm(coefficients[keep, 1], axis=1)
    acceleration = np.linalg.norm(2 * coefficients[keep, 2], axis=1)
    percentiles = [.5, .95, .99, quantile, .999, 1.]
    return dict(
        observed_frames=int(eligible.sum()), complete_windows=int(complete.sum()),
        retained_windows=int(keep.sum()), rejected_fit_windows=int((~keep).sum()),
        residual_limit_m=residual_limit, percentile_levels=percentiles,
        speed_percentiles_mps=np.quantile(speed, percentiles).tolist(),
        acceleration_percentiles_mps2=np.quantile(acceleration, percentiles).tolist(),
        max_speed_mps=float(np.quantile(speed, quantile) * margin),
        max_acceleration_mps2=float(np.quantile(acceleration, quantile) * margin),
        policy=dict(window_frames=window, polynomial_degree=2, quantile=quantile,
                    safety_margin=margin, residual_floor_m=.15, residual_mad_multiplier=6))


def projection_scales(matrix, positions_m, pitch):
    """Largest singular value of the local homography Jacobian, in pixels/metre.

    This conservative scalar accommodates either direction of image motion.
    Positions only select local scale. They never set the displayed ball position.
    """
    points = np.asarray(positions_m, float).reshape(-1, 2) / pitch
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    d = homogeneous[:, 2]
    valid = np.isfinite(homogeneous).all(axis=1) & (abs(d) > 1e-6)
    scales = np.full(len(points), np.nan)
    h = homogeneous[valid]
    jacobian = ((matrix[:2, :2][None] * h[:, 2, None, None]
                 - h[:, :2, None] * matrix[2, :2][None, None])
                / h[:, 2, None, None]**2 / np.asarray(pitch)[None, None])
    scales[valid] = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
    return scales


def load_homographies(path, frame_ids):
    wanted = set(map(int, frame_ids))
    matrices = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            frame = int(row['frame'])
            if frame in wanted and int(row['valid']) and float(row['reprojection_error']) <= 2.:
                matrix = np.array([float(row[f'h{i}{j}']) for i in range(3) for j in range(3)]).reshape(3, 3)
                if np.isfinite(matrix).all():
                    matrices[frame] = matrix
    return matrices


def frame_motion_limits(rows, ball_m, players_m, matrices, calibration, zoom, width, height):
    """Build camera-aware limits with conservative smoothing of scale changes."""
    pitch = np.asarray(calibration['pitch_m'], float)
    speed, acceleration = calibration['max_speed_mps'], calibration['max_acceleration_mps2']
    fps = calibration['fps']
    if min(speed, acceleration, fps) <= 0:
        raise ValueError('Calibration needs positive speed, acceleration and fps')
    raw_scales, sources = [], []
    for i, row in enumerate(rows):
        matrix = matrices.get(row['frame'])
        scale, source = np.nan, 'camera_scale_fallback'
        if matrix is not None:
            if np.isfinite(ball_m[i]).all():
                scale = projection_scales(matrix, [ball_m[i]], pitch)[0]
                source = 'tactical_ball_local_jacobian'
            if not np.isfinite(scale) or scale <= 0:
                players = np.asarray(players_m[i])
                finite = np.isfinite(players).all(axis=1)
                players = players[finite]
                projected = np.column_stack((players / pitch, np.ones(len(players)))) @ matrix.T
                projected = projected[:, :2] / projected[:, 2, None]
                visible = ((projected >= 0).all(axis=1)
                           & (projected < [width, height]).all(axis=1))
                locations = players[visible] if visible.any() else [pitch / 2]
                values = projection_scales(matrix, locations, pitch)
                values = values[np.isfinite(values) & (values > 0)]
                if len(values):
                    scale = float(np.quantile(values, .9))
                    source = 'visible_player_scale_p90' if visible.any() else 'pitch_center_scale'
        raw_scales.append(scale)
        sources.append(source)
    available = np.asarray(raw_scales)
    available = available[np.isfinite(available) & (available > 0)]
    if not len(available):
        raise ValueError('No usable homography scale in this clip')
    reference = None
    result = []
    # Restrict depth/scale contraction so a previous valid velocity can decelerate.
    max_fraction = min(.02, .5 * acceleration / (speed * fps))
    for i, raw in enumerate(raw_scales):
        new_shot = i == 0 or rows[i].get('camera_affine') is None or rows[i]['frame'] != rows[i-1]['frame'] + 1
        if new_shot:
            reference = (raw if np.isfinite(raw) and raw > 0 else float(np.median(available))) / zoom[i]
        elif np.isfinite(raw) and raw > 0:
            target = .9 * reference + .1 * raw / zoom[i]
            reference = float(np.clip(target, reference * (1-max_fraction), reference * (1+max_fraction)))
        scale = reference * zoom[i]
        result.append(dict(pixels_per_meter=scale, raw_pixels_per_meter=float(raw) if np.isfinite(raw) else None,
                           scale_source=sources[i], max_speed_mps=speed,
                           max_acceleration_mps2=acceleration,
                           max_speed_px_frame=speed * scale / fps,
                           max_acceleration_px_frame2=acceleration * scale / fps**2))
    return result


def run(args):
    game = GAMES[args.game]
    pitch = np.asarray(game['pitch'], float)
    positions = []
    with args.atd.open() as stream:
        for i, line in enumerate(stream):
            head, _, tail = line.partition(':')
            if int(head) - 1 != i:
                raise ValueError('ATD must be consecutive, one-based tactical frame IDs')
            positions.append([float(x) for x in tail.rsplit(':', 1)[1].split(',')[:2]])
    points = np.asarray(positions) * pitch
    if len(points) != game['n_video_frames']:
        raise ValueError('Unexpected tactical frame count')
    eligible = np.zeros(len(points), bool)
    for _, first, last in half_video_blocks(args.game):
        eligible[first:last+1] = True
    eligible &= (points >= -3).all(axis=1) & (points <= pitch+3).all(axis=1)
    summary = estimate_motion(points, eligible, game['fps'], args.window, args.quantile, args.margin)
    summary.update(schema=1, game=args.game, source='automatic_tactical_positions',
                   source_sha256=digest(args.atd), code_sha256=digest(__file__),
                   pitch_m=pitch.tolist(), fps=game['fps'], answers_read=False,
                   interpretation='Empirical 2D automatic-track envelope; aerial projection and tracking errors remain.')
    output = private_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic(output, summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atd', type=Path, required=True)
    parser.add_argument('--game', choices=sorted(GAMES), default=GAME)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--window', type=int, default=7)
    parser.add_argument('--quantile', type=float, default=.995)
    parser.add_argument('--margin', type=float, default=1.25)
    run(parser.parse_args())
