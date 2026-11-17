"""Observed co-motion proxy, not an official or sufficient definition of control.

Same-slot motion is trusted only over a bounded observed interval with plausible
player speed. Slots are not identities. Half resets and explicit missingness are
mandatory. No labels, latent states or event targets are accepted.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    radius_m: float = 2.0
    max_relative_speed_mps: float = 5.0
    lag_frames: int = 5
    minimum_run_frames: int = 3
    opponent_margin_m: float = 0.5
    max_player_speed_mps: float = 12.0

    def __post_init__(self):
        for x in (self.radius_m, self.max_relative_speed_mps, self.max_player_speed_mps):
            if not math.isfinite(x) or x <= 0:
                raise ValueError('Motion limits must be finite and positive')
        if not math.isfinite(self.opponent_margin_m) or self.opponent_margin_m < 0:
            raise ValueError('Invalid opponent margin')
        if any(type(x) is not int or x < 1 for x in (self.lag_frames, self.minimum_run_frames)):
            raise ValueError('Motion lengths must be positive integers')


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


def control_support(players, ball, teams, blocks, fps=25., config=MotionConfig()):
    from direct_weight_model import _blocks
    p, b = np.asarray(players, np.float32), np.asarray(ball, np.float32)
    if (p.ndim != 3 or p.shape[2] != 2 or b.shape != (len(p), 2) or
        len(teams) != p.shape[1] or any(t not in ('H', 'A') for t in teams) or
        not math.isfinite(fps) or fps <= 0 or np.isinf(p).any() or np.isinf(b).any()):
        raise ValueError('Invalid motion inputs')
    support = np.full(len(p), '', dtype='<U1')
    side = np.asarray(teams)
    lag = config.lag_frames
    for _, first, last in _blocks(blocks, len(p)):
        xy, bb = p[first:last+1], b[first:last+1]
        valid = np.isfinite(xy).all(2) & np.isfinite(bb).all(1)[:, None]
        distance = np.where(valid, np.linalg.norm(xy-bb[:, None], axis=2), np.inf)
        close = np.zeros(valid.shape, bool)
        if lag < len(xy):
            # Every intermediate observation must exist; do not bridge occlusion
            # or a re-acquired slot using just its endpoints.
            missing = np.concatenate([np.zeros((1, p.shape[1]), int), np.cumsum(~valid, axis=0)])
            observed_span = missing[lag+1:]-missing[:-(lag+1)] == 0
            vp = (xy[lag:]-xy[:-lag])*fps/lag
            vb = (bb[lag:]-bb[:-lag])*fps/lag
            error = np.linalg.norm(vp-vb[:, None], axis=2)
            close[lag:] = (observed_span & (error <= config.max_relative_speed_mps) &
                (np.linalg.norm(vp, axis=2) <= config.max_player_speed_mps) &
                (distance[lag:] <= config.radius_m))
        nearest = np.column_stack([distance[:, side == t].min(1) if np.any(side == t)
                                   else np.full(len(xy), np.inf) for t in ('H', 'A')])
        raw = np.full(len(xy), '', dtype='<U1')
        for ti, team in enumerate(('H', 'A')):
            hit = close[:, side == team].any(1) if np.any(side == team) else np.zeros(len(xy), bool)
            hit &= nearest[:, ti]+config.opponent_margin_m <= nearest[:, 1-ti]
            raw[hit] = team
        # Offline confirmation can backfill a *confirmed* short run, not gaps.
        edges = np.r_[0, np.flatnonzero(raw[1:] != raw[:-1])+1, len(raw)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            if raw[lo] and hi-lo >= config.minimum_run_frames:
                support[first+lo:first+hi] = raw[lo]
    return support


def gate_evidence(evidence, support):
    if len(evidence) != len(support):
        raise ValueError('Motion evidence length mismatch')
    # Positive existing dead-ball evidence survives; missingness never becomes D.
    return [label if label not in ('H', 'A') or label == support[i] else None
            for i, label in enumerate(evidence)]
