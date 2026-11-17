"""Smart Tagging restart evidence for the sequence decoder (opt-in: --smart-tagging).

Every set piece is a restart, and every restart ends a dead-ball stretch. The tagging only says
roughly when (a window of a few seconds) and sometimes who; it is automatic, so some tags are
wrong. This module turns each restart tag into evidence that tracking has to confirm:

  1. Restarts: SET PIECES instances whose Type is a throw-in, corner, goal kick, free kick or a
     kick off (either variant). Codes, tag groups and values are matched case-insensitively.
  2. Locate the restart inside the tag window from the ball track: the first strike (ball speed
     rises to `kick_speed` after `settle_frames` without in-play movement), else the first
     reappearance after `reappear_gap` missing frames, else the point `fallback_position` of the
     way through the window.
  3. Walk back from the restart to the stoppage onset: just after the last run of
     `in_play_frames` frames with the ball seen moving in bounds, bounded by `max_stoppage`
     (`max_kickoff_stoppage` for kick offs, which follow a break or a goal), the half start and
     the previous restart. Those frames become D evidence. If in-play evidence ends less than
     `min_stoppage` frames before the restart, tracking contradicts the tag and it adds nothing.
  4. The Team tag is a prior for the restart frame only: it fills that frame when its evidence
     is missing, unless the first observed team within `prior_frames` disagrees, in which case
     the observation wins. A `N/A` team (every throw-in, per DATA.md) uses that observation.
     Vote and gap filling then run as usual; phase intervals are never copied as labels.

Team tags name clubs; they are mapped to H/A against the club names in the ATD metadata, after
normalising accents, case, punctuation and generic club words (FC, Club, ...). A value that is
not an exact normalised match must be at least `team_match_threshold` similar to exactly one
side, by `team_match_margin` over the other, or loading stops with a readable error.

Defaults are a-priori football timescales and speeds, checked on synthetic cases only.
Standard library only, like the decoder.
"""

from dataclasses import dataclass
import difflib
import math
import os
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "challenge", "starter"))
import load_smart_tagging  # noqa: E402
from sequence_decoder import DEAD, FIELD_SEP, MISSING, TEAMS, _xy  # noqa: E402

SET_PIECES = "setpieces"
KICKOFF = "kick off"
# Compact (lower case, alphanumerics only) Type value -> restart kind.
RESTART_TYPES = {"throwin": "throw-in", "cornerkick": "corner kick", "goalkick": "goal kick",
                 "freekick": "free kick"}
UNKNOWN_TEAM_VALUES = {"", "na", "none", "unknown"}
# Words that describe what a club is rather than which one; dropped before comparing names.
GENERIC_CLUB_WORDS = {"fc", "cf", "afc", "club", "football", "futbol", "sc", "cd", "ud", "sd",
                      "de", "the"}


@dataclass(frozen=True)
class TaggingConfig:
    kick_speed: float = 5.0          # m/s; a struck or thrown ball moves at least this fast
    settle_frames: int = 12          # frames without in-play movement before a strike
    reappear_gap: int = 25           # missing-ball frames before a reappearance counts
    fallback_position: float = 0.5   # restart point as a fraction of the window if unlocated
    speed_max_gap: int = 5           # widest frame gap between ball observations for a speed
    in_play_speed: float = 2.0       # m/s; ball seen moving in bounds at least this fast
    in_play_frames: int = 25         # consecutive such frames that confidently mean live play
    min_stoppage: int = 50           # less dead time than this before a restart contradicts it
    max_stoppage: int = 1500         # longest D stretch added before a restart (60 s)
    max_kickoff_stoppage: int = 7500  # same before a kick off (5 min)
    prior_frames: int = 50           # horizon after a restart for the team prior / observation
    team_match_threshold: float = 0.8  # similarity a non-exact Team tag needs to map to a club
    team_match_margin: float = 0.1   # and its lead over the other club

    def __post_init__(self):
        counts = (self.settle_frames, self.reappear_gap, self.speed_max_gap, self.in_play_frames,
                  self.min_stoppage, self.max_stoppage, self.max_kickoff_stoppage,
                  self.prior_frames)
        if min(counts) < 0 or self.speed_max_gap < 1 or self.in_play_frames < 1:
            raise ValueError("frame counts must be >= 0; speed_max_gap, in_play_frames >= 1")
        if not 0.0 <= self.fallback_position <= 1.0:
            raise ValueError("fallback_position must be between 0 and 1")


@dataclass(frozen=True)
class Restart:
    start: int        # first frame of the tag window
    end: int          # first frame after it
    kind: str
    team: object      # H, A or None when the tag names no team


def _fold(text):
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def _compact(text):
    return re.sub(r"[^0-9a-z]", "", _fold(text))


def _name_tokens(name):
    words = re.sub(r"[^0-9a-z]+", " ", _fold(name)).split()
    kept = [word for word in words if word not in GENERIC_CLUB_WORDS]
    return kept or words


def _tag(tags, group):
    """A tag group's value, matching the group name case-insensitively."""
    for key, value in tags.items():
        if _compact(key) == group:
            return value
    return None


def team_names(meta_path):
    """(home, away) club names from the ATD metadata, first team home.

    Same as challenge/starter/baseline.py team_names(), which cannot be imported here without
    numpy; tests check the two agree when numpy is installed.
    """
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def _similarity(value, name):
    a, b = _name_tokens(value), _name_tokens(name)
    if set(a) <= set(b) or set(b) <= set(a):
        return 1.0
    return difflib.SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()


def map_team(value, names, config=TaggingConfig()):
    """H, A, or None for an unknown team. Raises ValueError when a club cannot be told apart."""
    if _compact(value) in UNKNOWN_TEAM_VALUES:
        return None
    if len(names) != 2 or not all(names):
        raise ValueError("ATD metadata must name two teams to map Team tags, got {!r}"
                         .format(names))
    exact = [side for side, name in zip(TEAMS, names) if _name_tokens(value) == _name_tokens(name)]
    if len(exact) == 1:
        return exact[0]
    scores = [_similarity(value, name) for name in names]
    best = max(range(2), key=lambda i: scores[i])
    if scores[best] >= config.team_match_threshold and \
            scores[best] - scores[1 - best] >= config.team_match_margin:
        return TEAMS[best]
    raise ValueError(
        "Smart Tagging Team value {!r} does not map to one ATD team: similarity {:.2f} to "
        "{!r} (home), {:.2f} to {!r} (away); need >= {} with a lead of {}. Check the club "
        "names or adjust --tag-team-match-threshold.".format(
            value, scores[0], names[0], scores[1], names[1], config.team_match_threshold,
            config.team_match_margin))


def restart_kind(type_value):
    compact = _compact(type_value)
    if compact.startswith(_compact(KICKOFF)):
        return KICKOFF
    return RESTART_TYPES.get(compact)


def restart_events(events, names, config=TaggingConfig()):
    """Restarts from events shaped like load_smart_tagging.load() output, in window order.

    Every distinct Team value in `events` is mapped first, so an unmappable club name fails
    loudly even when it only appears on a phase tag.
    """
    teams = {}
    for event in events:
        value = _tag(event.get("tags", {}), "team")
        if value not in teams:
            teams[value] = map_team(value, names, config)
    out = []
    for event in events:
        if _compact(event.get("code")) != SET_PIECES:
            continue
        kind = restart_kind(_tag(event.get("tags", {}), "type"))
        if kind is not None:
            out.append(Restart(int(event["start_frame"]), int(event["end_frame"]), kind,
                               teams[_tag(event["tags"], "team")]))
    return sorted(out, key=lambda restart: (restart.start, restart.end))


def read_ball(raw_path, n_frames, pitch):
    """Per-frame ball position in metres, or None where the tracking has no ball."""
    length, width = pitch
    ball = [None] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            x, y = _xy(parts[2])
            ball[int(parts[0]) - 1] = (x * length, y * width)
    return ball


def ball_speeds(ball, fps, max_gap):
    """m/s from consecutive observations at most `max_gap` frames apart, else None."""
    speeds = [None] * len(ball)
    previous = None
    for t, position in enumerate(ball):
        if position is None:
            continue
        if previous is not None and t - previous <= max_gap:
            (x0, y0), (x1, y1) = ball[previous], position
            speeds[t] = math.hypot(x1 - x0, y1 - y0) * fps / (t - previous)
        previous = t
    return speeds


def _locate(lo, hi, first, ball, speeds, in_play, config):
    """(restart frame, how) within [lo, hi) of one half starting at `first`."""
    for t in range(lo, hi):
        if speeds[t] is not None and speeds[t] >= config.kick_speed and \
                not any(in_play[max(first, t - config.settle_frames):t]):
            return t, "kick"
    for t in range(max(lo, first + config.reappear_gap), hi):
        if ball[t] is not None and all(ball[u] is None for u in range(t - config.reappear_gap, t)):
            return t, "reappearance"
    return lo + int(round((hi - 1 - lo) * config.fallback_position)), "window"


def apply_restarts(evidence, ball, blocks, restarts, config=TaggingConfig(), fps=25):
    """Evidence with restart D stretches and team priors added, plus aggregate counts only."""
    out = list(evidence)
    speeds = ball_speeds(ball, fps, config.speed_max_gap)
    in_play = [ball[t] is not None and speeds[t] is not None and
               speeds[t] >= config.in_play_speed and evidence[t] != DEAD
               for t in range(len(ball))]
    stats = {"restart_events": len(restarts), "used": 0, "ignored": 0, "contradicted": 0,
             "team_prior_overridden": 0, "team_from_tracking": 0, "d_frames_added": 0,
             "located_by": {"kick": 0, "reappearance": 0, "window": 0}}
    previous = {}
    for restart in restarts:
        block = next(((first, last) for _, first, last in blocks
                      if max(restart.start, first) < min(restart.end, last + 1)), None)
        if block is None:
            stats["ignored"] += 1  # outside both halves
            continue
        first, last = block
        lo, hi = max(restart.start, first), min(restart.end, last + 1)
        frame, how = _locate(lo, hi, first, ball, speeds, in_play, config)
        if frame <= previous.get(block, first - 1):
            stats["ignored"] += 1  # duplicate of a restart already used
            continue

        limit = config.max_kickoff_stoppage if restart.kind == KICKOFF else config.max_stoppage
        floor = max(first, frame - limit, previous.get(block, first - 1) + 1)
        onset, run, live = floor, 0, False
        for t in range(frame - 1, floor - 1, -1):
            run = run + 1 if in_play[t] else 0
            if run >= config.in_play_frames:
                onset, live = t + run, True
                break
        if live and frame - onset < config.min_stoppage:
            stats["contradicted"] += 1
            continue

        stats["used"] += 1
        stats["located_by"][how] += 1
        previous[block] = frame
        stats["d_frames_added"] += sum(label != DEAD for label in out[onset:frame])
        out[onset:frame] = [DEAD] * (frame - onset)

        observed = next((evidence[t] for t in range(frame, min(frame + config.prior_frames,
                                                                last + 1))
                         if evidence[t] in TEAMS), None)
        seed = restart.team
        if seed is None:
            seed = observed
            stats["team_from_tracking"] += observed is not None
        elif observed is not None and observed != seed:
            seed = observed
            stats["team_prior_overridden"] += 1
        if seed is not None and out[frame] not in TEAMS:
            out[frame] = seed
    return out, stats


def load_events(game, data_dir=None):
    """Smart Tagging events through the official starter loader."""
    return load_smart_tagging.load(game, data_dir)


def apply_bundle(evidence, meta_path, raw_path, blocks, pitch, fps, config=TaggingConfig(),
                 game=load_smart_tagging.GAME, data_dir=None):
    """apply_restarts() on a data bundle's tagging and ball track."""
    restarts = restart_events(load_events(game, data_dir), team_names(meta_path), config)
    ball = read_ball(raw_path, len(evidence), pitch)
    return apply_restarts(evidence, ball, blocks, restarts, config, fps)
