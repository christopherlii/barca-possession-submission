"""Fixed observation-only gap generator; copied unchanged from the audited development helper."""
import numpy as np
from direct_weight_model import _blocks

def gap_mask(n, blocks, run_frames, seed, share=.4):
    """Exact hidden count per half, full fixed-size gaps plus one remainder.

At least one observed frame separates gaps and protects each half endpoint.
Only clock, RNG and declared configuration enter this function, never targets.
"""
    if type(n) is not int or n < 1 or type(run_frames) is not int or run_frames < 1:
        raise ValueError('Positive integer frame counts required')
    if not np.isfinite(share) or not 0 < share < 1:
        raise ValueError('Missing share must lie strictly between zero and one')
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, bool)
    for _, first, last in _blocks(blocks, n):
        length = last-first+1
        hidden = int(np.floor(length*share))
        if not hidden:
            continue
        runs = [run_frames]*(hidden//run_frames)
        if hidden % run_frames:
            runs.append(hidden % run_frames)
        visible = length-hidden
        if visible < len(runs)+1:
            raise ValueError('Insufficient visible frames to separate gaps')
        rng.shuffle(runs)
        spaces = 1+rng.multinomial(visible-len(runs)-1, np.full(len(runs)+1, 1/(len(runs)+1)))
        position = first+int(spaces[0])
        for size, space in zip(runs, spaces[1:]):
            mask[position:position+size] = True
            position += size+int(space)
        if position != last+1 or mask[first:last+1].sum() != hidden:
            raise AssertionError('Gap construction changed coverage')
    return mask
