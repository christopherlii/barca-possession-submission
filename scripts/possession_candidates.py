"""Label-blind candidate windows from tactical tracking, tags and local audio.

This is the first stage of a funnel, not a possession classifier. It never reads
ground truth, assigns H/A/D labels, or treats absence of a cue as proof of stability.
All frame IDs in outputs are zero-based tactical-video IDs; ends are exclusive.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from candidate_audio import mask_runs
from sequence_decoder import team_of_each_track


@dataclass(frozen=True)
class CandidateConfig:
    fps: int = 25
    proximity_radius_m: float = 3.0
    proximity_lead_m: float = 0.5
    minimum_team_run_frames: int = 3
    contested_radius_m: float = 3.0
    contested_margin_m: float = 1.0
    minimum_contested_frames: int = 3
    missing_ball_frames: int = 5
    weak_evidence_frames: int = 25
    out_of_bounds_margin_m: float = 0.5
    padding_frames: int = 25
    merge_gap_frames: int = 5

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Candidate parameters must be finite and nonnegative')
        if min(self.fps, self.minimum_team_run_frames, self.minimum_contested_frames,
               self.missing_ball_frames, self.weak_evidence_frames) < 1:
            raise ValueError('Sampling rate and run lengths must be positive')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_tracking(raw_path, meta_path, start_frame, end_frame, pitch=(105.3, 68.0)):
    if not 0 <= start_frame < end_frame:
        raise ValueError('Invalid tracking window')
    teams = team_of_each_track(meta_path)
    side = np.array([0 if team == 'H' else 1 for team in teams])
    count = end_frame-start_frame
    ball = np.full((count, 2), np.nan)
    distance = np.full((count, 2), np.inf)
    seen = np.zeros(count, dtype=bool)
    scale = np.asarray(pitch)
    with Path(raw_path).open() as handle:
        for line in handle:
            try:
                parts = line.rstrip().split(':')
                frame = int(parts[0])-1
                if frame < start_frame:
                    continue
                if frame >= end_frame:
                    break
                if len(parts) != 3 or seen[frame-start_frame]:
                    raise ValueError
                groups = parts[1].split(';')
                if len(groups) != len(teams):
                    raise ValueError
                i = frame-start_frame
                seen[i] = True
                xy = np.array([float(v) for v in parts[2].split(',')[:2]])
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    continue
                ball[i] = xy * scale
                players = np.array([[float(v) for v in group.split(',')[:2]] for group in groups]) * scale
                if players.shape != (len(teams), 2):
                    raise ValueError
                valid = np.isfinite(players).all(axis=1)
                for team in (0, 1):
                    selected = players[valid & (side == team)]
                    if len(selected):
                        distance[i, team] = np.linalg.norm(selected-ball[i], axis=1).min()
            except (ValueError, IndexError, OverflowError):
                raise ValueError('Invalid tracking structure; private row suppressed') from None
    if not seen.all():
        raise ValueError('Tracking window is not consecutive')
    return dict(start_frame=start_frame, end_frame=end_frame, ball=ball,
                distance=distance, pitch=tuple(pitch))


def read_tags(path, fps=25):
    events = []
    for instance in ET.parse(path).getroot().iter('instance'):
        try:
            start = float(instance.findtext('start'))
            end = float(instance.findtext('end'))
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError
            events.append(dict(start_frame=round(start*fps), end_frame=round(end*fps),
                code=(instance.findtext('code') or '').strip().upper(),
                tags={label.findtext('group'): label.findtext('text')
                      for label in instance.findall('label')}))
        except (TypeError, ValueError):
            raise ValueError('Invalid tag interval; private event suppressed') from None
    return events


def event(source, kind, start, end, anchor=None):
    return dict(source=source, kind=kind, start_frame=int(start), end_frame=int(end),
                anchor_frame=int(start if anchor is None else anchor))


def tracking_events(tracking, config=CandidateConfig()):
    ball, distances = tracking['ball'], tracking['distance']
    first = tracking['start_frame']
    observed = np.isfinite(ball).all(axis=1)
    finite_both = np.isfinite(distances).all(axis=1)
    lead = np.full(len(ball), np.inf)
    lead[finite_both] = np.abs(distances[finite_both, 0]-distances[finite_both, 1])
    proximity = np.argmin(distances, axis=1)
    strong = observed & (distances.min(axis=1) <= config.proximity_radius_m) & (lead >= config.proximity_lead_m)
    result = []
    # Keep the previous supported side through missing/contested intervals. These
    # are proximity changes, not labels or claims of controlled possession.
    supported = None
    current = None
    run_start = 0
    for i in range(len(ball)):
        team = int(proximity[i]) if strong[i] else None
        if team is None:
            current = None
            continue
        if team != current:
            current, run_start = team, i
        if i-run_start+1 == config.minimum_team_run_frames:
            if supported is not None and supported != team:
                result.append(event('tracking', 'proximity_team_change', first+run_start, first+i+1))
            supported = team
    contested = (observed & finite_both &
                 (distances.max(axis=1) <= config.contested_radius_m) &
                 (lead <= config.contested_margin_m))
    masks = [('contested_ball', contested, config.minimum_contested_frames),
             ('missing_ball', ~observed, config.missing_ball_frames),
             ('weak_control_evidence', observed & ~strong & ~contested, config.weak_evidence_frames)]
    for kind, mask, minimum in masks:
        for lo, hi in mask_runs(mask):
            if hi-lo >= minimum:
                result.append(event('tracking', kind, first+lo, first+hi))
    length, width = tracking['pitch']
    margin = config.out_of_bounds_margin_m
    outside = observed & ((ball[:, 0] < -margin) | (ball[:, 0] > length+margin) |
                          (ball[:, 1] < -margin) | (ball[:, 1] > width+margin))
    for lo, hi in mask_runs(outside):
        result.append(event('tracking', 'possible_out_of_play', first+lo, first+lo+1))
        if hi < len(ball) and observed[hi]:
            result.append(event('tracking', 'return_in_bounds', first+hi, first+hi+1))
    return result


def tagging_events(tags):
    result = []
    for tag in tags:
        code = tag['code'].strip().upper()
        lo, hi = tag['start_frame'], tag['end_frame']
        if code in ('SET PIECES', 'GOALS'):
            kind = 'restart_tag' if code == 'SET PIECES' else 'goal_tag'
            result.append(event('smart_tagging', kind, lo, hi))
        elif code in ('ATTACKING TRANSITION', 'DEFENSIVE TRANSITION'):
            result.extend([event('smart_tagging', 'transition_tag_start', lo, lo+1),
                           event('smart_tagging', 'transition_tag_end', hi, hi+1)])
    return result


def audio_events(cues, source, fps=25):
    if source not in ('tactical_audio', 'broadcast_audio'):
        raise ValueError('Audio clock/source must be explicit')
    return [event(source, cue['kind'], math.floor(cue['start_seconds']*fps),
                  max(math.floor(cue['start_seconds']*fps)+1, math.ceil(cue['end_seconds']*fps)))
            for cue in cues]


def combine(events, start_frame, end_frame, config=CandidateConfig()):
    selected = []
    for row in events:
        lo = max(start_frame, row['start_frame']-config.padding_frames)
        hi = min(end_frame, row['end_frame']+config.padding_frames)
        if lo >= hi:
            continue
        selected.append(dict(row, window_start_frame=lo, window_end_frame=hi))
    selected.sort(key=lambda row: (row['window_start_frame'], row['anchor_frame'], row['source'], row['kind']))
    for i, row in enumerate(selected):
        row['id'] = i
    windows = []
    for row in selected:
        lo, hi = row['window_start_frame'], row['window_end_frame']
        if windows and lo <= windows[-1]['end_frame']+config.merge_gap_frames:
            windows[-1]['end_frame'] = max(windows[-1]['end_frame'], hi)
            windows[-1]['event_ids'].append(row['id'])
        else:
            windows.append(dict(start_frame=lo, end_frame=hi, event_ids=[row['id']]))
    for i, window in enumerate(windows):
        window['id'] = i
        window['sources'] = sorted({selected[j]['source'] for j in window['event_ids']})
        window['reasons'] = sorted({selected[j]['kind'] for j in window['event_ids']})
    covered = sum(w['end_frame']-w['start_frame'] for w in windows)
    return dict(events=selected, windows=windows, summary=dict(event_count=len(selected),
        window_count=len(windows), coverage_fraction=covered/(end_frame-start_frame),
        by_source=dict(Counter(row['source'] for row in selected)),
        by_reason=dict(Counter(row['kind'] for row in selected))))
