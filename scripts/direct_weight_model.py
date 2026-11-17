"""Data-free neural-weight experiment: tracking features and frozen matrix inference.

This module never reads labels, latent causes, decoder predictions or restart tags.
All predictive coefficients live in the model JSON. No fitting or label rewriting.
"""
import json
import math
import os
from collections import deque

import numpy as np

from path_guard import check_paths

FEATURE_NAMES = (
    'ball_observed', 'home_proximity', 'away_proximity', 'home_distance_advantage',
    'ball_speed', 'speed_known', 'ball_stillness', 'recent_observed_fraction',
    'outside_distance', 'boundary_proximity', 'centre_proximity', 'missing_age',
    'home_own_half_fraction', 'away_own_half_fraction', 'outward_speed', 'player_crowding',
)
LABELS = np.array(['H', 'A', 'D'])


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


def _point(group, pitch):
    parts = group.split(',')
    if len(parts) != 3:
        raise ValueError('tracking group needs x,y,speed')
    x, y = float(parts[0]), float(parts[1])
    if math.isinf(x) or math.isinf(y):
        raise ValueError('infinite tracking coordinates')
    if not math.isfinite(x) or not math.isfinite(y):
        return (np.nan, np.nan)
    point = (x * pitch[0], y * pitch[1])
    if not all(math.isfinite(v) for v in point):
        raise ValueError('tracking coordinate overflow')
    return point


def features_from_tracking(raw_path, teams, n_frames, pitch, blocks, fps=25):
    """Causal 16-column features from observed ATD only, resetting history each half."""
    check_paths([raw_path])
    if type(n_frames) is not int or n_frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError('invalid frame count or frame rate')
    if len(pitch) != 2 or any(not math.isfinite(x) or x <= 0 for x in pitch):
        raise ValueError('invalid pitch dimensions')
    if not teams or any(team not in ('H', 'A') for team in teams):
        raise ValueError('track teams must be H or A')
    blocks = _blocks(blocks, n_frames)
    players = np.full((n_frames, len(teams), 2), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    seen = set()
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(':')
            if len(parts) != 3:
                raise ValueError('tracking row needs frame:players:ball')
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames or frame in seen:
                raise ValueError('out-of-range or repeated tracking frame')
            seen.add(frame)
            groups = parts[1].split(';')
            if len(groups) != len(teams):
                raise ValueError('tracking slot count differs from metadata')
            players[frame] = [_point(group, pitch) for group in groups]
            ball[frame] = _point(parts[2], pitch)
    return features_from_arrays(players, ball, teams, pitch, blocks, fps)


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


def _numeric(value):
    if isinstance(value, list):
        return all(_numeric(x) for x in value)
    return type(value) in (int, float) and math.isfinite(value)


def load_weights(path):
    check_paths([path])
    if os.path.getsize(path) > 2 * 1024 * 1024:
        raise ValueError('weight file exceeds 2 MiB')
    with open(path) as handle:
        data = json.load(handle)
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise ValueError('unsupported weight schema')
    if data.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('weight feature order does not match extractor')
    large = 'W3' in data
    shapes = ({'W1': (128, 16), 'b1': (128,), 'W2': (136, 128), 'b2': (136,),
               'W3': (3, 136), 'b3': (3,)} if large else
              {'W1': (12, 16), 'b1': (12,), 'W2': (3, 12), 'b2': (3,)})
    shapes.update(transition_logits=(3, 3), initial_logits=(3,))
    if set(data) != set(shapes) | {'schema_version', 'feature_names'}:
        raise ValueError('unexpected or missing weight fields')
    result = {}
    for name, shape in shapes.items():
        if not _numeric(data[name]):
            raise ValueError('weights must contain finite numeric values')
        try:
            array = np.asarray(data[name], dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise ValueError('invalid matrix shape') from error
        if array.shape != shape:
            raise ValueError('wrong shape for ' + name)
        result[name] = array
    return result


def parameter_count(weights):
    return sum(value.size for value in weights.values())


def frame_logits(features, weights):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('features must be a finite N x 16 matrix')
    result = np.empty((len(x), 3))
    try:
        with np.errstate(over='raise', invalid='raise'):
            for first in range(0, len(x), 4096):
                h = np.maximum(0., x[first:first + 4096] @ weights['W1'].T + weights['b1'])
                h = h @ weights['W2'].T + weights['b2']
                if 'W3' in weights:
                    h = np.maximum(0., h) @ weights['W3'].T + weights['b3']
                result[first:first + len(h)] = h
    except FloatingPointError as error:
        raise ValueError('non-finite neural inference') from error
    return result


def _log_softmax(values, axis=-1):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def predict(features, weights, blocks, temporal=True):
    """Only model emissions and matrix transitions decide H/A/D; half mask supplies X."""
    return predict_from_logits(frame_logits(features, weights), weights, blocks, temporal)


def predict_from_logits(logits, weights, blocks, temporal=True):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError('logits must be a finite N x 3 matrix')
    blocks = _blocks(blocks, len(logits))
    result = np.full(len(logits), 'X')
    transitions = _log_softmax(weights['transition_logits'], axis=1)
    initial = _log_softmax(weights['initial_logits'])
    for _, first, last in blocks:
        emissions = _log_softmax(logits[first:last + 1])
        if not temporal:
            result[first:last + 1] = LABELS[emissions.argmax(axis=1)]
            continue
        pointers = np.zeros((len(emissions), 3), dtype=np.uint8)
        state = emissions[0] + initial
        for t in range(1, len(emissions)):
            candidates = state[:, None] + transitions
            pointers[t] = candidates.argmax(axis=0)
            state = candidates[pointers[t], np.arange(3)] + emissions[t]
        winner = int(state.argmax())
        result[last] = LABELS[winner]
        for t in range(len(emissions) - 1, 0, -1):
            winner = int(pointers[t, winner])
            result[first + t - 1] = LABELS[winner]
    return result.tolist()
