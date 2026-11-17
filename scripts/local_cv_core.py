"""Conservative image-space sequence evidence. No answers, training or API clients.

Distances are in player heights, NOT calibrated pitch metres. Scores are detector
scores, not possession probabilities. Unknown live/dead is never converted to D.
"""
from collections import deque
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


class PlayerTracks:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, players, t, cut=False):
        if cut:
            self.tracks.clear()
        self.tracks = {k: v for k, v in self.tracks.items() if t-v['t'] < .5}
        ids = list(self.tracks)
        costs = np.full((len(ids), len(players)), 1e6)
        for i, key in enumerate(ids):
            old = self.tracks[key]
            for j, p in enumerate(players):
                a, b = np.array(old['box']), np.array(p['box'])
                height = max(1., (a[3]-a[1]+b[3]-b[1])/2)
                d = np.linalg.norm((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)/height
                compatible = old['team'] == p['team'] or 'unknown' in (old['team'], p['team'])
                if compatible and d < 1.2:
                    costs[i, j] = d
        matched = {}
        if costs.size:
            ai, bi = linear_sum_assignment(costs)
            matched = {int(j): ids[i] for i, j in zip(ai, bi) if costs[i, j] < 1.2}
        result = []
        for j, player in enumerate(players):
            key = matched.get(j)
            if key is None:
                key = self.next_id
                self.next_id += 1
            row = dict(player, track_id=key, t=t)
            self.tracks[key] = row
            result.append(row)
        return result


class BallTrack:
    def __init__(self):
        self.history = deque(maxlen=3)

    def update(self, candidates, t, cut=False):
        if cut or (self.history and t-self.history[-1][0] > .4):
            self.history.clear()
        if not candidates:
            return None
        if not all(len(c) == 3 and all(math.isfinite(x) for x in c) and 0 <= c[2] <= 1 for c in candidates):
            raise ValueError('Invalid ball observations')
        # Ambiguous isolated hypotheses are left unresolved. Continuity is only
        # a ranking cue; predicted positions are never emitted as detections.
        scored = []
        for c in candidates:
            rank = c[2]
            if self.history:
                old_t, old = self.history[-1]
                dt = t-old_t
                expected = np.array(old[:2])
                if len(self.history) > 1:
                    tt, pp = self.history[-2]
                    velocity = (np.array(old[:2])-pp[:2])/max(old_t-tt, .001)
                    expected += velocity*dt
                distance = np.linalg.norm(np.array(c[:2])-expected)
                if distance > 50+600*dt:
                    continue
                rank -= min(.6, distance/250)
            scored.append((float(rank), c))
        if not scored:
            return None
        scored.sort(key=lambda v: -v[0])
        if len(scored) > 1 and scored[0][0]-scored[1][0] < .12:
            return None
        chosen = list(map(float, scored[0][1]))
        self.history.append((t, chosen))
        return chosen


class ControlSequence:
    """Do not award a passing ball to a nearby defender on one observation."""
    def __init__(self, dwell=.2, radius=.45, relative_speed=2., hold=2.):
        self.dwell, self.radius = dwell, radius
        self.relative_speed, self.hold = relative_speed, hold
        self.reset()

    def reset(self):
        self.owner = None
        self.last_control = -math.inf
        self.candidate = None
        self.relative = {}

    def update(self, ball, players, t, cut=False):
        if cut:
            self.reset()
        label = self.owner if t-self.last_control <= self.hold else None
        result = dict(label=label, live_dead='unknown', control=None,
                      reason='Short ownership carry' if label else 'No confirmed control',
                      nearest=None, anchor_start=None)
        contacts, current = [], {}
        if ball is not None:
            for p in players:
                x1, y1, x2, y2 = p['box']
                h = max(1., y2-y1)
                offset = (np.array(ball[:2])-[(x1+x2)/2, y2-.04*h])/h
                distance = float(np.linalg.norm(offset))
                prior = self.relative.get(p['track_id'])
                speed = None
                if prior is not None and 0 < t-prior[0] <= .25:
                    speed = float(np.linalg.norm(offset-prior[1])/(t-prior[0]))
                current[p['track_id']] = (t, offset)
                contacts.append((distance, p, speed))
        self.relative = current
        contacts.sort(key=lambda r: r[0])
        if contacts:
            distance, p, speed = contacts[0]
            result['nearest'] = dict(track_id=p['track_id'], team=p['team'],
                                     distance_heights=distance, relative_speed=speed)
            uncontested = not any(q['team'] != p['team'] and d < self.radius+.15 for d, q, _ in contacts[1:])
            valid = p['team'] in ('H', 'A') and distance <= self.radius and uncontested
            valid = valid and speed is not None and speed <= self.relative_speed
            if valid:
                key = p['track_id']
                if self.candidate is None or self.candidate[0] != key:
                    self.candidate = (key, t, p['team'])
                if t-self.candidate[1]+1e-8 >= self.dwell:
                    self.owner, self.last_control = p['team'], t
                    result.update(label=self.owner, control=key,
                                  anchor_start=self.candidate[1], reason='Repeated nearby co-motion')
                else:
                    result['reason'] = 'Contact candidate awaiting temporal evidence'
                return result
            if distance <= self.radius:
                result['reason'] = 'Nearby ball: ambiguous team, contest, or fast relative motion'
        self.candidate = None
        if ball is None:
            # Missing is an observation gap, never dead-ball evidence. Avoid
            # claiming continuous possession over an unseen turnover.
            result.update(label=None, reason='Ball missing or ambiguous')
        return result


def nearest_baseline(ball, players, radius=.45):
    if ball is None or not players:
        return None
    distances = [(np.linalg.norm(np.array(ball[:2])-[(p['box'][0]+p['box'][2])/2, p['box'][3]]) /
                  max(1., p['box'][3]-p['box'][1]), p['team']) for p in players]
    distance, team = min(distances)
    return team if distance <= radius and team in ('H', 'A') else None
