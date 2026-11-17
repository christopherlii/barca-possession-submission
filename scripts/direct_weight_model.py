"""Frozen submission inference: direct weight model."""
import math


from collections import deque


import numpy as np


FEATURE_NAMES = (
    'ball_observed', 'home_proximity', 'away_proximity', 'home_distance_advantage',
    'ball_speed', 'speed_known', 'ball_stillness', 'recent_observed_fraction',
    'outside_distance', 'boundary_proximity', 'centre_proximity', 'missing_age',
    'home_own_half_fraction', 'away_own_half_fraction', 'outward_speed', 'player_crowding',
)


def _blocks(blocks, n):
    result = []
    for block in blocks:
        if len(block) != 3 or any(type(x) is not int for x in block):
            raise ValueError('half blocks must contain integer half, first, last')
        half, first, last = block
        if half not in (1, 2) or not 0 <= first <= last < n:
            raise ValueError('invalid half block')
        result.append((half, first, last))
    result.sort(key=lambda x: x[1])
    if any(a[2] >= b[1] for a, b in zip(result, result[1:])):
        raise ValueError('half blocks overlap')
    return result


def features_from_arrays(players, ball, teams, pitch, blocks, fps=25):
    """Identical observed-only features for local inference, with no file reads.

    Training file APIs retain their path guards. This pure API lets explicit
    supplied-match inference reuse the feature calculation without copying
    private inputs into a development directory or weakening those guards.
    """
    players=np.asarray(players,dtype=float);ball=np.asarray(ball,dtype=float)
    if (players.ndim!=3 or players.shape[2]!=2 or ball.shape!=(len(players),2) or
        len(teams)!=players.shape[1] or not teams or any(t not in ('H','A') for t in teams) or
        np.isinf(players).any() or np.isinf(ball).any() or not math.isfinite(fps) or fps<=0 or
        len(pitch)!=2 or not np.isfinite(pitch).all() or min(pitch)<=0):
        raise ValueError('invalid observed feature arrays')
    n_frames=len(ball);blocks=_blocks(blocks,n_frames)
    features = np.zeros((n_frames, len(FEATURE_NAMES)))
    features[:, 11] = 1.0
    home = np.array(teams) == 'H'
    length, width = pitch
    window = max(1, round(2 * fps))
    max_speed_gap = max(1, round(.4 * fps))
    normals = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    for half, first, last in blocks:
        previous, previous_t = None, None
        recent = deque()
        total = np.zeros(2)
        total_square = np.zeros(2)
        for t in range(first, last + 1):
            row = features[t]
            while recent and recent[0][0] < t - window + 1:
                _, point = recent.popleft()
                total -= point
                total_square -= point * point
            current = ball[t]
            observed = bool(np.isfinite(current).all())
            velocity = None
            if observed:
                row[0] = 1.0
                recent.append((t, current))
                total += current
                total_square += current * current
                if previous is not None and 0 < t - previous_t <= max_speed_gap:
                    velocity = (current - previous) * fps / (t - previous_t)
                    row[4] = np.clip(np.linalg.norm(velocity) / 20., 0., 1.)
                    row[5] = 1.0
                previous, previous_t = current, t
            count = len(recent)
            row[7] = count / min(window, t - first + 1)
            if count >= 2:
                variance = np.maximum(total_square / count - (total / count) ** 2, 0.)
                row[6] = np.clip(1. - np.sqrt(variance.sum()) / 2., 0., 1.)
            valid_players = np.isfinite(players[t]).all(axis=1)
            for mask, column, direction in ((home, 12, 1 if half == 1 else -1),
                                             (~home, 13, -1 if half == 1 else 1)):
                points = players[t, mask & valid_players]
                if len(points):
                    own = points[:, 0] <= length / 2. if direction == 1 else points[:, 0] >= length / 2.
                    row[column] = own.mean()
            if previous is None:
                continue
            row[11] = np.clip((t - previous_t) / fps / 5., 0., 1.)
            distances = np.linalg.norm(players[t] - previous, axis=1)
            nearest = []
            for mask, column in ((home, 1), (~home, 2)):
                values = distances[mask & valid_players]
                value = values.min() if len(values) else None
                nearest.append(value)
                if value is not None:
                    row[column] = np.clip(1. - value / 10., 0., 1.)
            if all(x is not None for x in nearest):
                row[3] = np.clip((nearest[1] - nearest[0]) / 5., -1., 1.)
            x, y = previous
            gaps = np.array([x, length - x, y, width - y])
            outside_x, outside_y = max(0., -x, x - length), max(0., -y, y - width)
            row[8] = np.clip(math.hypot(outside_x, outside_y) / 3., 0., 1.)
            row[9] = np.clip(1. - gaps.min() / 5., 0., 1.)
            row[10] = np.clip(1. - np.linalg.norm(previous - np.array([length / 2., width / 2.])) / 10., 0., 1.)
            if velocity is not None:
                row[14] = np.clip(np.dot(velocity, normals[int(gaps.argmin())]) / 15., 0., 1.)
            row[15] = np.clip(np.count_nonzero(valid_players & (distances <= 3.)) / 6., 0., 1.)
    return features

