"""Frozen submission inference: control motion."""
from pathlib import Path


import numpy as np


def tracking_arrays(raw_path, teams, n_frames, pitch, dtype=np.float32):
    """Strict one-based ATD loader, including supplied inputs for local inference.

Experiments must apply path_guard before calling. Unlike training-only loaders,
this input API intentionally supports supplied-match inference, never targets.
"""
    if (type(n_frames) is not int or n_frames < 1 or not teams or
        any(t not in ('H', 'A') for t in teams) or len(pitch) != 2 or
        not np.isfinite(pitch).all() or min(pitch) <= 0):
        raise ValueError('Invalid tracking dimensions')
    if np.dtype(dtype) not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Tracking coordinates require float32 or float64')
    p = np.full((n_frames, len(teams), 2), np.nan, dtype)
    b = np.full((n_frames, 2), np.nan, dtype)
    seen = np.zeros(n_frames, bool)
    try:
        with Path(raw_path).open() as handle:
            for line in handle:
                parts = line.strip().split(':')
                if len(parts) != 3:
                    raise ValueError
                frame = int(parts[0])-1
                groups = parts[1].split(';')
                if not 0 <= frame < n_frames or seen[frame] or len(groups) != len(teams):
                    raise ValueError
                xy = np.array([[float(v) for v in g.split(',')[:2]] for g in groups])
                ball = np.array([float(v) for v in parts[2].split(',')[:2]])
                if xy.shape != (len(teams), 2) or ball.shape != (2,) or np.isinf(xy).any() or np.isinf(ball).any():
                    raise ValueError
                p[frame] = xy*np.asarray(pitch)
                b[frame] = ball*np.asarray(pitch)
                seen[frame] = True
    except (ValueError, IndexError, OverflowError):
        raise ValueError('Invalid ATD input; private row suppressed') from None
    if not seen.all():
        raise ValueError('ATD frames must be consecutive and complete')
    return p, b

