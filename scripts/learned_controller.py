"""Development-only supervised controller; inputs are tracking, never label sidecars.

Offline temporal context is allowed. Windows reset at half boundaries. Model fitting
and evaluation live in learned_controller_experiment.py, outside this input API.
"""
import numpy as np

from direct_weight_model import features_from_tracking, _blocks


CONTEXT_SECONDS = (.2, .5, 1., 2., 4.)


def contextualize(features, blocks, fps):
    """Current values plus preceding/following window means; no half leakage.

    Windows are clipped, not zero-padded. Empty periods remain zero. This is
    bidirectional offline inference, not a claim of causal/streaming operation.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('need finite matrix and positive fps')
    blocks = _blocks(blocks, len(x))
    out = np.zeros((len(x), x.shape[1] * (1 + 2 * len(CONTEXT_SECONDS))), np.float32)
    for _, first, last in blocks:
        local = x[first:last + 1]
        n = len(local)
        positions = np.arange(n)
        summed = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(local, axis=0, dtype=np.float64)])
        out[first:last + 1, :x.shape[1]] = local
        column = x.shape[1]
        for seconds in CONTEXT_SECONDS:
            width = max(1, round(seconds * fps))
            for left, right in ((np.maximum(0, positions - width + 1), positions + 1),
                                (positions, np.minimum(n, positions + width))):
                out[first:last + 1, column:column + x.shape[1]] = (
                    summed[right] - summed[left]) / (right - left)[:, None]
                column += x.shape[1]
    return out


def raw_tracking_features(raw_path, teams, n_frames, pitch, blocks, fps=25, dynamics=False, touches=False):
    base = features_from_tracking(raw_path, teams, n_frames, pitch, blocks, fps)
    if dynamics or touches:
        from controller_dynamics import tracking_arrays,motion_features
        players,ball=tracking_arrays(raw_path,teams,n_frames,pitch)
        if dynamics:
            base=np.concatenate([base,motion_features(players,ball,teams,blocks,fps)],axis=1)
        if touches:
            from ball_control_features import ball_control_features
            base=np.concatenate([base,ball_control_features(ball,blocks,fps,.04),
                                 ball_control_features(ball,blocks,fps,.2)],axis=1)
    return np.asarray(base,dtype=np.float32)


def tracking_features(raw_path, teams, n_frames, pitch, blocks, fps=25, dynamics=False, touches=False):
    return contextualize(raw_tracking_features(raw_path,teams,n_frames,pitch,blocks,fps,
                                              dynamics=dynamics,touches=touches),blocks,fps)


def decode_probabilities(probabilities, blocks, switch_cost):
    """Maximum-log-probability H/A/D path, with one cost per state change."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('invalid H/A/D probabilities')
    if not np.allclose(p.sum(axis=1), 1.) or not np.isfinite(switch_cost) or switch_cost < 0:
        raise ValueError('invalid probabilities or switch cost')
    blocks = _blocks(blocks, len(p))
    labels = np.full(len(p), 'X', dtype='<U1')
    transition = -switch_cost * (1 - np.eye(3))
    emissions = np.log(np.maximum(p, 1e-12))
    for _, first, last in blocks:
        values = emissions[first].copy()
        back = np.zeros((last - first + 1, 3), dtype=np.int8)
        for t in range(first + 1, last + 1):
            candidates = values[:, None] + transition
            back[t - first] = candidates.argmax(axis=0)
            values = candidates.max(axis=0) + emissions[t]
        state = values.argmax()
        for t in range(last, first - 1, -1):
            labels[t] = ('H', 'A', 'D')[state]
            state = back[t - first, state]
    return labels
