"""Label-free motion evidence for a controller near a noisy tracked ball.

Team-pooled distances penalize implausibly jumpy player tracks. They supplement,
never overwrite, observed positions. No knowledge of synthetic player identities,
carrier IDs, formation slots, labels, or latent causes is used.
"""
import numpy as np

from direct_weight_model import _blocks, _point
from path_guard import check_paths


def tracking_arrays(raw_path, teams, n_frames, pitch):
    check_paths([raw_path])
    if not teams or any(t not in ('H','A') for t in teams) or n_frames < 1:
        raise ValueError('invalid tracking dimensions or teams')
    players = np.full((n_frames,len(teams),2), np.nan, dtype=np.float32)
    ball = np.full((n_frames,2), np.nan, dtype=np.float32)
    seen = set()
    with open(raw_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(':')
            if len(parts) != 3:
                raise ValueError('invalid tracking row')
            frame = int(parts[0])-1
            groups = parts[1].split(';')
            if frame in seen or not 0 <= frame < n_frames or len(groups) != len(teams):
                raise ValueError('invalid frame or slot count')
            seen.add(frame)
            players[frame] = [_point(g,pitch) for g in groups]
            ball[frame] = _point(parts[2],pitch)
    return players,ball


def motion_features(players, ball, teams, blocks, fps=25.):
    """24 dimensionless team-motion features, resetting at each half.

    Per team: first/second nearest distances, near-player fraction, and for each
    0.2/0.5/1-second backward interval: motion availability, minimum distance plus
    speed cost, and speed of the current nearest track. Costs are metres plus
    0.2 seconds * metres/second. Values are clipped for robustness, not fitted.
    Missing ball rows have no invented distance evidence.
    """
    players,ball = np.asarray(players),np.asarray(ball)
    if players.ndim != 3 or players.shape[2] != 2 or ball.shape != (len(players),2):
        raise ValueError('invalid position shapes')
    if len(teams)!=players.shape[1] or any(t not in ('H','A') for t in teams):
        raise ValueError('invalid team slots')
    if not np.isfinite(fps) or fps <= 0 or np.isinf(players).any() or np.isinf(ball).any():
        raise ValueError('invalid positions or fps')
    blocks = _blocks(blocks,len(ball))
    output = np.zeros((len(ball),24),np.float32)
    team_array = np.array(teams)
    for _,first,last in blocks:
        xy,b = players[first:last+1],ball[first:last+1]
        known = np.isfinite(xy).all(axis=2) & np.isfinite(b).all(axis=1)[:,None]
        distances = np.linalg.norm(xy-b[:,None,:],axis=2)
        distances = np.where(known,distances,np.inf)
        for ti,team in enumerate(('H','A')):
            mask = team_array==team
            if not mask.any():
                raise ValueError('need both teams')
            local = distances[:,mask]
            valid = known[:,mask]
            offset = 12*ti
            nearest = local.argmin(axis=1)
            sorted_distances = np.sort(local,axis=1)
            output[first:last+1,offset] = np.minimum(sorted_distances[:,0]/10.,1.)
            output[first:last+1,offset+1] = np.minimum(sorted_distances[:,min(1,mask.sum()-1)]/10.,1.)
            output[first:last+1,offset+2] = (local<2.).sum(axis=1)/mask.sum()
            for j,seconds in enumerate((.2,.5,1.)):
                lag = max(1,round(seconds*fps))
                speed = np.full(local.shape,np.nan)
                if lag<len(xy):
                    speed[lag:] = np.linalg.norm(xy[lag:,mask]-xy[:-lag,mask],axis=2)*fps/lag
                speed_known = np.isfinite(speed)&valid
                cost = np.where(speed_known,local+.2*speed,np.inf)
                col=offset+3+3*j
                output[first:last+1,col] = speed_known.any(axis=1)
                output[first:last+1,col+1] = np.minimum(cost.min(axis=1)/10.,1.)
                selected = speed[np.arange(len(xy)),nearest]
                selected = np.where(np.isfinite(selected)&valid.any(axis=1),selected,0.)
                output[first:last+1,col+2] = np.minimum(selected/15.,1.)
    return output


def dynamics_from_tracking(raw_path, teams, n_frames, pitch, blocks, fps=25):
    players,ball=tracking_arrays(raw_path,teams,n_frames,pitch)
    return motion_features(players,ball,teams,blocks,fps)
