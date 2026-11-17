"""Frozen submission inference: learned controller."""
import numpy as np


from direct_weight_model import _blocks


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

