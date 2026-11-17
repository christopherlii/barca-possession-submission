"""Causal, expiring negative memory for ball-like background objects.

Learn only from detector candidates corroborated by same-frame automatic tactical
tracking. Repeated YOLO crops and SAM propagation are not independent votes.
This is algorithmic pseudo-labeling, not ground truth. No human match labels are
read. Memory uses raw-image appearance and camera-compensated coordinates.

Run before association and continuous filling, on an unfilled detection cache.
A rejection removes evidence, never substitutes a made-up ball position. Original
values and decisions remain in each row for audit. Stationary distractors are the
initial scope; moving lookalikes are deliberately not extrapolated.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from local_cv_possession import atomic
from stage1_cv_baseline import digest, private_path
from sam_confirmation_recovery import read_rows
import cv2

CANDIDATE_KEYS = ('candidates', 'crowded_zoom_candidates', 'player_context_candidates',
                  'recovery_candidates', 'blank_recovery_candidates')
DETECTION_KEYS = ('selected', 'short_gap_detection', 'aerial_detection',
                  'blank_recovery_detection', 'bidirectional_detection', 'crowded_zoom_selected')


@dataclass(frozen=True)
class MemoryConfig:
    confirmations: int = 3
    ttl_frames: int = 75
    max_entries: int = 128
    positive_confidence: float = .4
    positive_agreement_px: float = 8.
    positive_protection_px: float = 24.
    candidate_confidence: float = .15
    single_detector_override: float = .85
    match_radius_px: float = 6.
    patch_size: int = 21
    minimum_patch_std: float = 8.
    minimum_correlation: float = .88
    maximum_color_distance: float = 35.


def point(value, confidence=False):
    try:
        p = np.asarray(value, dtype=float)
        return p if p.shape == ((3,) if confidence else (2,)) and np.isfinite(p).all() else None
    except (TypeError, ValueError):
        return None


def collect_candidates(row, config):
    """Deduplicate crop detections so duplicate boxes cannot confirm memory."""
    values = [p for key in CANDIDATE_KEYS for p in (row.get(key) or [])]
    values += [row[k] for k in DETECTION_KEYS if row.get(k) is not None]
    clean = [p for value in values if (p := point(value, True)) is not None
             and p[2] >= config.candidate_confidence]
    result = []
    for p in sorted(clean, key=lambda p: -p[2]):
        if not any(np.linalg.norm(p[:2]-q[:2]) <= 3 for q in result):
            result.append(p)
    return result


def consensus(row, candidates, config):
    """A unique spatial cluster of detector votes agrees with automatic ATD.

    The guide is same-frame projected tracking, never an interpolated display
    location. Two distant agreeing clusters are ambiguous and teach no negatives.
    """
    guide = point(row.get('guide'))
    if guide is None:
        return None
    votes = [p for p in candidates if p[2] >= config.positive_confidence
             and np.linalg.norm(p[:2]-guide) <= config.positive_agreement_px]
    if not votes or any(np.linalg.norm(a[:2]-b[:2]) > config.positive_agreement_px
                        for a in votes for b in votes):
        return None
    return max(votes, key=lambda p: p[2])[:2].copy()


class DistractorMemory:
    """Frozen negative templates refreshed only by new independent positives.

    Ambiguous matches never extend TTL or increase confirmation count. Coordinates
    and patch scale follow the camera affine, not the candidate being vetoed.
    """
    def __init__(self, config=None):
        self.config = config or MemoryConfig()
        self.entries = []
        self.previous_frame = None
        self.next_id = 1

    def _appearance(self, image, center, scale=1.):
        c = self.config
        side = max(5, int(round(c.patch_size*scale)))
        h, w = image.shape[:2]
        x, y = center
        if side > min(h, w) or x-side/2 < 0 or y-side/2 < 0 or x+side/2 >= w or y+side/2 >= h:
            return None
        patch = cv2.getRectSubPix(image, (side, side), (float(x), float(y)))
        patch = cv2.resize(patch, (c.patch_size, c.patch_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(float)
        if gray.std() < c.minimum_patch_std:
            return None
        normalized = (gray-gray.mean()).ravel()
        normalized /= np.linalg.norm(normalized)
        return normalized, patch.mean(axis=(0, 1))

    def _match(self, image, p, entry):
        distance = float(np.linalg.norm(p-entry['center']))
        if distance > self.config.match_radius_px*entry['scale']:
            return None
        appearance = self._appearance(image, p, entry['scale'])
        if appearance is None:
            return None
        correlation = float(appearance[0] @ entry['template'])
        color_distance = float(np.linalg.norm(appearance[1]-entry['color']))
        if correlation < self.config.minimum_correlation or color_distance > self.config.maximum_color_distance:
            return None
        return dict(memory_id=entry['id'], distance_px=distance,
                    correlation=correlation, color_distance=color_distance,
                    confirmations=entry['hits'], last_confirmed_frame=entry['last'])

    def update(self, row, image):
        c = self.config
        frame = int(row['frame'])
        affine = np.asarray(row.get('camera_affine'), dtype=float)
        valid = affine.shape == (2, 3) and np.isfinite(affine).all()
        if valid:
            singular = np.linalg.svd(affine[:, :2], compute_uv=False)
            valid = bool(.8 <= singular.min() <= singular.max() <= 1.25
                         and np.linalg.det(affine[:, :2]) > 0)
        cut = self.previous_frame is None or frame != self.previous_frame+1 or not valid
        if cut:
            self.entries.clear()
        else:
            scale = np.sqrt(np.linalg.det(affine[:, :2]))
            for entry in self.entries:
                entry['center'] = affine[:, :2] @ entry['center']+affine[:, 2]
                entry['scale'] *= scale
        self.previous_frame = frame
        self.entries = [e for e in self.entries if frame-e['last'] <= c.ttl_frames
                        and .5 <= e['scale'] <= 2.]
        candidates = collect_candidates(row, c)
        positive = consensus(row, candidates, c)
        # Seeing a supported ball here invalidates negative memory regardless of
        # appearance. A stationary real ball must not be permanently blacklisted.
        removed = []
        if positive is not None:
            removed = [e['id'] for e in self.entries if np.linalg.norm(e['center']-positive)
                       <= c.positive_protection_px*e['scale']]
            self.entries = [e for e in self.entries if e['id'] not in removed]
        rejected, overrides = [], []
        for p in candidates:
            matches = [(e, match) for e in self.entries
                       if e['hits'] >= c.confirmations and (match := self._match(image, p[:2], e)) is not None]
            if not matches:
                continue
            entry, match = max(matches, key=lambda item: item[1]['correlation'])
            match.update(point=p.tolist(), age_frames=frame-entry['last'])
            protected = positive is not None and np.linalg.norm(p[:2]-positive) <= c.positive_protection_px
            if protected or p[2] >= c.single_detector_override:
                overrides.append(dict(match, reason='positive_evidence_override'))
            else:
                rejected.append(dict(match, reason='confirmed_distractor'))
        learned = 0
        if positive is not None:
            used = set()
            for p in candidates:
                if np.linalg.norm(p[:2]-positive) <= c.positive_protection_px or p[2] >= c.single_detector_override:
                    continue
                matches = [(e, match) for e in self.entries
                           if (match := self._match(image, p[:2], e)) is not None]
                if matches:
                    entry, _ = max(matches, key=lambda item: item[1]['correlation'])
                    if entry['id'] not in used and entry['last'] != frame:
                        entry['hits'] += 1
                        entry['last'] = frame
                        used.add(entry['id'])
                        learned += 1
                    continue
                appearance = self._appearance(image, p[:2])
                if appearance is not None:
                    self.entries.append(dict(id=self.next_id, center=p[:2].copy(), scale=1.,
                                             template=appearance[0], color=appearance[1], hits=1, last=frame))
                    used.add(self.next_id)
                    self.next_id += 1
                    learned += 1
            self.entries = sorted(self.entries, key=lambda e: (e['last'], e['hits']), reverse=True)[:c.max_entries]
        return dict(positive=positive.tolist() if positive is not None else None,
                    positive_sources=['detector', 'automatic_tactical'] if positive is not None else [],
                    rejected=rejected, overrides=overrides, invalidated_memory_ids=removed,
                    learned=learned, entries=len(self.entries), camera_reset=cut)


def apply_decision(row, decision):
    """Prevent rejected points from re-entering association as old anchors."""
    result = copy.deepcopy(row)
    rejected = [np.asarray(item['point'][:2]) for item in decision['rejected']]
    def denied(value):
        return value is not None and any(np.linalg.norm(np.asarray(value[:2])-p) <= 3 for p in rejected)
    original = {}
    for key in CANDIDATE_KEYS:
        if key in result:
            kept = [p for p in result[key] or [] if not denied(p)]
            if kept != result[key]:
                original[key] = result[key]
                result[key] = kept
    for key in DETECTION_KEYS:
        if denied(result.get(key)):
            original[key] = result[key]
            result[key] = None
    if denied(result.get('context_position')):
        for key in ('context_position', 'context_source', 'sam2_mask', 'sam2_extension'):
            if key in result:
                original[key] = result[key]
                result[key] = None
        result['context_source'] = 'distractor_rejected'
    result['distractor_memory'] = dict(decision, removed_evidence=original)
    return result


def filter_video(rows, source, config=None, progress=None):
    """Decode each raw frame once; retain source timeline including any gaps."""
    memory = DistractorMemory(config)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open source video: {source}')
    result, previous = [], None
    try:
        for index, row in enumerate(rows):
            if previous is None or row['frame'] != previous+1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame'])
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {row['frame']}")
            result.append(apply_decision(row, memory.update(row, image)))
            previous = row['frame']
            if progress and (index+1) % 500 == 0:
                progress(index+1)
    finally:
        cap.release()
    return result


def summarize(rows, result):
    decisions = [r['distractor_memory'] for r in result]
    return dict(frames=len(rows), positive_frames=sum(d['positive'] is not None for d in decisions),
                rejected_candidates=sum(len(d['rejected']) for d in decisions),
                rejection_frames=sum(bool(d['rejected']) for d in decisions),
                removed_positions=sum(a.get('context_position') is not None and b.get('context_position') is None
                                      for a, b in zip(rows, result)),
                positive_overrides=sum(len(d['overrides'])+len(d['invalidated_memory_ids']) for d in decisions),
                maximum_memory_entries=max((d['entries'] for d in decisions), default=0),
                accuracy_measured=False)


def run(args):
    out = private_path(args.out)
    if out.exists():
        raise ValueError('Use a new output directory')
    rows = read_rows(args.cache/'detections.jsonl')
    metadata = json.loads((args.cache/'config.json').read_text())
    if metadata.get('continuous_trajectory') or metadata.get('distractor_memory'):
        raise ValueError('Use an unfilled cache without existing distractor filtering')
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    started = time.monotonic()
    def progress(completed):
        status = dict(state='running', completed=completed, total=len(rows), elapsed_seconds=time.monotonic()-started)
        atomic(out/'status.json', status)
        print(json.dumps(status), flush=True)
    config = MemoryConfig(ttl_frames=args.ttl_frames)
    if config.ttl_frames < config.confirmations:
        raise ValueError('TTL must permit repeated confirmations')
    result = filter_video(rows, metadata['source'], config, progress)
    summary = dict(summarize(rows, result), elapsed_seconds=time.monotonic()-started)
    metadata['distractor_memory'] = dict(config=asdict(config), code_sha256=digest(__file__),
                                       source_sha256=digest(args.cache/'detections.jsonl'),
                                       automatic_pseudolabels=True)
    atomic(out/'config.json', metadata)
    with (out/'detections.jsonl').open('w') as handle:
        for row in result:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
    atomic(out/'summary.json', summary)
    atomic(out/'status.json', dict(state='complete', **summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ttl-frames', type=int, default=75, help='Expiry since last independently confirmed negative (75 = 3s at 25fps)')
    run(parser.parse_args())
