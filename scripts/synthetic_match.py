"""Deterministic synthetic match bundles for possession experiments.

Writes a short fake match with its own ground truth, so decoders can be compared on labelled data
that has nothing to do with the supplied match. Three stages, kept apart on purpose:

  1. Latent: a possession process per half -- spells of H or A (very short to very long), passes
     inside a spell, turnovers with a contested hand-over window, dead balls (ball observed out of
     bounds, ball out but never observed, or a stationary foul restart) and a graded pre-kickoff
     dead stretch at the start of the second half. Players and ball move on crude kinematics.
     `carry_model` sets where a carried ball sits: "jitter" (default, the v1 bundles) redraws it
     0.3-1.2 m from the carrier in a random direction every frame; "dribble" keeps one offset per
     carrier that random-walks by about `dribble_drift_m` a frame inside `dribble_reach_m`, and
     with `keeper_hold_prob` a goalkeeper who gets the ball holds it still (within
     `keeper_hold_reach_m`) for `keeper_hold_s`. Those holds keep the keeper's team label: a
     live-ball stress case for stationary-ball dead evidence. Select it with a SyntheticConfig
     override file (`--config` here, `--generator-config` in scripts/decoder_experiments.py),
     e.g. {"carry_model": "dribble"}.
     Two optional loose-ball events, off by default: with `interception_prob` a turnover is a
     pass from the losing carrier that ends at an opponent -- the flight keeps the passer's team
     label (a loose ball stays with the last team) and the label changes on the receiver's first
     frame on the ball; with `deflection_prob` an opponent comes within touching distance of a
     within-team pass for a few mid-flight frames, and the label stays with the passing team.
     Both draw from their own random stream, so default bundles are unchanged.
     Two optional kick-off sources, off by default, so kick-offs can grow with match length
     instead of being one fixed stoppage per file: `first_half_kickoff` adds a graded pre-kickoff
     dead stretch at the start of the first half as well, and with `goal_rate_per_90` (goals per
     90 minutes of live play) a spell can end in a goal. A goal is followed by a kick-off dead
     stretch of `goal_kickoff_s` (both teams in their own half, ball on the centre spot, cause
     "kickoff") and the conceding team kicks off. Goals draw from their own random stream, and
     the manifest lists these fields only when one of them is on, so default bundles are unchanged.
  2. Observation: what a tracker would report from that truth -- ball dropout that is more likely
     in contested moments and after the ball leaves play, positional noise, an opponent closest to
     the ball around hand-overs, tracklet slots that vanish and come back elsewhere, slots that
     swap identity (possibly across teams), and occasional duplicate tracks.
  3. Tagging: Smart Tagging-like restart tags from the latent truth -- a window of a few seconds
     around the first live frame after each graded dead stretch (the second-half kickoff
     included), with jitter, missed restarts, false tags, wrong and N/A teams (always N/A for
     throw-ins, per DATA.md). It draws from its own random stream, so the files above it are the
     same whatever the tagging config.

Output, all in one directory and byte-identical for the same seed and config:

    SYN-<seed>_atd.txt              raw tracking, `frame:players:ball`, the ATD format the official
                                    baseline parses (one-based frame IDs, NaN, normalised coords)
    SYN-<seed>_atd.xml              metadata team_of_each_track() can read
    SYN-<seed>_possession_gt.csv    zero-based frame,label with H/A/D, X outside the halves
    SYN-<seed>_latent.csv           zero-based frame,cause,ball_missing: the latent reason for each
                                    label (CAUSES) and whether the ball was dropped, for aggregate
                                    error diagnostics only -- decoders must never read it
    SYN-<seed>_tagging.json         restart tags [{start, end, kind, team}], zero-based frames,
                                    end exclusive, kind as in tagging_evidence, team H/A/null
    SYN-<seed>_manifest.json        seed, config, frame count, half blocks, pitch, file names

Default rates come from published dataset facts in challenge/DATA.md (ball missing on 41% of graded
frames, 46 track slots, ~20.5 tracked objects per graded frame, pitch size, half layout). Spell and
dead-ball timescales and tagging error rates are a-priori values. Nothing is derived from any
match labels.

    python scripts/synthetic_match.py --seed 7 --out outputs/synthetic/seed7
"""

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from path_guard import check_paths, guard_paths  # noqa: E402

HOME, AWAY, DEAD, UNKNOWN = "H", "A", "D", "X"
PLAYERS_PER_TEAM = 11
FORMAT = "synthetic-match-v2"
# Latent cause behind each frame's label, in report order. New causes are appended so existing
# tables keep their layout.
CAUSES = ("ungraded", "kickoff", "dead_out_observed", "dead_out_hidden", "dead_foul", "contested",
          "flight", "hold", "interception", "deflection")
# Restart kinds, spelled as tagging_evidence.Restart.kind.
THROW_IN, CORNER, GOAL_KICK, FREE_KICK, KICK_OFF = (
    "throw-in", "corner kick", "goal kick", "free kick", "kick off")
FALSE_TAG_KINDS = (THROW_IN, CORNER, GOAL_KICK, FREE_KICK)
CARRY_MODELS = ("jitter", "dribble")

# Normalised 4-4-2 for a team attacking towards x = 1: goalkeeper first.
FORMATION = ((0.05, 0.50),
             (0.20, 0.15), (0.20, 0.38), (0.20, 0.62), (0.20, 0.85),
             (0.40, 0.15), (0.40, 0.38), (0.40, 0.62), (0.40, 0.85),
             (0.60, 0.35), (0.60, 0.65))


@dataclass(frozen=True)
class SyntheticConfig:
    n_frames: int = 7500                  # whole file, graded and ungraded; 25 fps -> 5 minutes
    fps: int = 25
    pitch_length: float = 105.3           # metres, DATA.md / games.py
    pitch_width: float = 68.0
    n_slots: int = 46                     # tracklet slots, DATA.md
    # Ungraded share of the file before, between and after the halves (DATA.md half table).
    pre_match_fraction: float = 0.056
    halftime_fraction: float = 0.101
    post_match_fraction: float = 0.046

    # Latent possession process (a-priori timescales, seconds).
    spell_median_s: float = 8.0
    spell_sigma: float = 0.7              # lognormal spread of a normal spell
    short_spell_prob: float = 0.12
    short_spell_s: tuple = (0.2, 1.0)
    long_spell_prob: float = 0.06
    long_spell_s: tuple = (40.0, 90.0)
    hold_s: tuple = (0.8, 4.0)            # time on the ball between passes
    pass_s: tuple = (0.25, 0.9)           # ball in flight
    dead_ball_prob: float = 0.3           # chance a spell ends in a stoppage rather than a turnover
    dead_ball_s: tuple = (2.0, 20.0)
    dead_out_observed_weight: float = 0.45  # ball seen beyond a line
    dead_out_hidden_weight: float = 0.30    # ball out of play but never seen out
    dead_foul_weight: float = 0.25          # stationary in-bounds restart, ball visible
    prekickoff_s: tuple = (3.0, 10.0)     # graded dead stretch before the second-half kickoff
    contest_s: float = 0.6                # +/- window around a turnover counted as contested

    # Carried ball (a-priori). "jitter" keeps the v1 output; the rest only affects "dribble".
    carry_model: str = "jitter"
    dribble_reach_m: tuple = (0.3, 1.2)   # ball's distance from the carrier
    dribble_drift_m: float = 0.05         # sd of the per-frame offset step, per axis
    keeper_hold_prob: float = 0.3         # chance a goalkeeper's carry starts with the ball held
    keeper_hold_s: tuple = (1.0, 3.0)     # how long a held ball stays put, unless passed sooner
    keeper_hold_reach_m: float = 0.2      # held ball's distance from where the keeper stood

    # Loose-ball events (a-priori; 0 keeps the default output byte-identical).
    interception_prob: float = 0.0        # chance a turnover is a pass intercepted by an opponent
    deflection_prob: float = 0.0          # chance an opponent touches a within-team pass mid-flight
    deflection_s: tuple = (0.04, 0.16)    # how long the opponent stays within touching distance

    # Kick-offs beyond the second half's (a-priori; off keeps the default output byte-identical).
    first_half_kickoff: bool = False      # graded pre-kickoff dead stretch before the first half too
    goal_rate_per_90: float = 0.0         # goals per 90 minutes of live play
    goal_kickoff_s: tuple = (10.0, 30.0)  # dead stretch between a goal and its kick-off

    # Observation model.
    ball_missing_rate: float = 0.41       # graded frames, DATA.md
    ball_missing_run_s: float = 0.8       # mean dropout run
    contested_missing_boost: float = 1.8  # dropout rate multiplier in contested frames
    dead_hidden_missing_rate: float = 0.9  # dropout once the ball has left the camera's view
    ball_noise_m: float = 0.15
    player_noise_m: float = 0.2
    handover_confusion_prob: float = 0.5  # chance a turnover has an opponent nearer the ball
    track_vanish_per_min: float = 1.0     # per player; ~7% missing gives DATA.md's ~20.5 tracked
    track_vanish_s: tuple = (0.5, 4.0)
    track_swap_per_min: float = 0.5       # whole match; nearby players exchange slots
    duplicate_per_min: float = 0.5        # whole match; a ghost copy of a player
    duplicate_s: tuple = (0.5, 3.0)
    duplicate_other_team_prob: float = 0.3

    # Restart tagging (a-priori error rates for an automatic tagger).
    tag_window_s: tuple = (2.0, 6.0)      # tag window length
    tag_jitter_s: float = 1.0             # sd of the window centre around the true restart
    tag_miss_prob: float = 0.1            # true restart with no tag
    tag_false_per_min: float = 0.2        # graded minutes; tags with no restart behind them
    tag_wrong_team_prob: float = 0.1      # tag names the other team
    tag_na_team_prob: float = 0.05        # tag names no team (throw-ins always do)

    def __post_init__(self):
        for name in ("pre_match_fraction", "halftime_fraction", "post_match_fraction",
                     "short_spell_prob", "long_spell_prob", "dead_ball_prob", "ball_missing_rate",
                     "dead_hidden_missing_rate", "handover_confusion_prob",
                     "duplicate_other_team_prob", "tag_miss_prob", "tag_wrong_team_prob",
                     "tag_na_team_prob", "keeper_hold_prob", "interception_prob",
                     "deflection_prob"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))
        if self.short_spell_prob + self.long_spell_prob > 1.0:
            raise ValueError("short_spell_prob + long_spell_prob must be <= 1")
        if self.n_slots < 2 * PLAYERS_PER_TEAM + 2:
            raise ValueError("n_slots must leave at least one spare slot per team")
        graded = 1.0 - self.pre_match_fraction - self.halftime_fraction - self.post_match_fraction
        if self.n_frames * graded < 4 * self.fps * max(self.prekickoff_s):
            raise ValueError("n_frames too small for two halves")
        if self.fps <= 0 or self.ball_missing_run_s <= 0:
            raise ValueError("fps and ball_missing_run_s must be positive")
        if min(self.tag_window_s) <= 0 or self.tag_jitter_s < 0 or self.tag_false_per_min < 0:
            raise ValueError("tag_window_s must be positive; tag jitter and false rate >= 0")
        if self.carry_model not in CARRY_MODELS:
            raise ValueError("carry_model must be one of {}".format(CARRY_MODELS))
        if not 0.0 <= self.dribble_reach_m[0] <= self.dribble_reach_m[1]:
            raise ValueError("dribble_reach_m must be (low, high) with 0 <= low <= high")
        if self.dribble_drift_m < 0 or self.keeper_hold_reach_m < 0 or min(self.keeper_hold_s) <= 0:
            raise ValueError("dribble_drift_m and keeper_hold_reach_m must be >= 0, "
                             "keeper_hold_s positive")
        if not 0.0 < self.deflection_s[0] <= self.deflection_s[1]:
            raise ValueError("deflection_s must be (low, high) with 0 < low <= high")
        if not isinstance(self.first_half_kickoff, bool):
            raise ValueError("first_half_kickoff must be true or false")
        if not 0.0 <= self.goal_rate_per_90 < math.inf:
            raise ValueError("goal_rate_per_90 must be finite and >= 0")
        if not 0.0 < self.goal_kickoff_s[0] <= self.goal_kickoff_s[1]:
            raise ValueError("goal_kickoff_s must be (low, high) with 0 < low <= high")


def load_config(path=None, **overrides):
    """SyntheticConfig from an optional JSON file of overrides; unknown keys are an error."""
    values = {}
    if path:
        check_paths([path])
        with open(path) as handle:
            values.update(json.load(handle))
    values.update(overrides)
    known = {field.name for field in fields(SyntheticConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError("unknown synthetic config keys: {}".format(unknown))
    values = {key: tuple(value) if isinstance(value, list) else value
              for key, value in values.items()}
    return SyntheticConfig(**values)


def half_blocks(config):
    """[(half, first, last)] zero-based graded frames, the same shape as games.half_video_blocks."""
    n = config.n_frames
    pre = max(1, round(n * config.pre_match_fraction))
    post = max(1, round(n * config.post_match_fraction))
    halftime = max(1, round(n * config.halftime_fraction))
    first_len = (n - pre - post - halftime) // 2
    second_len = n - pre - post - halftime - first_len
    first = (1, pre, pre + first_len - 1)
    second_start = pre + first_len + halftime
    return [first, (2, second_start, second_start + second_len - 1)]


# -- latent process ------------------------------------------------------------------------------

def _frames(rng, config, span_s):
    return max(1, int(round(rng.uniform(*span_s) * config.fps)))


def _spell_frames(rng, config):
    draw = rng.random()
    if draw < config.short_spell_prob:
        return _frames(rng, config, config.short_spell_s)
    if draw < config.short_spell_prob + config.long_spell_prob:
        return _frames(rng, config, config.long_spell_s)
    seconds = config.spell_median_s * math.exp(rng.gauss(0.0, config.spell_sigma))
    return max(config.fps, int(round(min(seconds, max(config.long_spell_s)) * config.fps)))


def _team_players(team):
    return range(PLAYERS_PER_TEAM) if team == HOME else range(PLAYERS_PER_TEAM,
                                                              2 * PLAYERS_PER_TEAM)


def _other(team):
    return AWAY if team == HOME else HOME


def _outfield(rng, team):
    return rng.choice(list(_team_players(team))[1:])


def _deflect(event_rng, config, frames, start, flight, team):
    """(deflection_prob > 0 only) an opponent within touching distance mid-flight; label kept."""
    if flight < 3 or event_rng.random() >= config.deflection_prob:
        return
    toucher = _outfield(event_rng, _other(team))
    length = min(flight - 2, _frames(event_rng, config, config.deflection_s))
    first = start + 1 + event_rng.randrange(flight - 1 - length)
    for f in range(first, first + length):
        frames[f].update(presser=toucher, event="deflection")


def _intercept(event_rng, config, frames, start, stop, passer, winner):
    """(interception_prob > 0 only) turn the passer's last hold before the turnover at `stop` into
    a pass that ends at `winner`. Flight frames keep the passer's label. True if it happened."""
    if event_rng.random() >= config.interception_prob:
        return False
    run = 0  # passer's uninterrupted hold up to the turnover; the pass leaves from inside it
    while (stop - run - 1 >= start and frames[stop - run - 1]["carrier"] == passer
           and frames[stop - run - 1]["flight"] is None):
        run += 1
    flight = min(run - 1, _frames(event_rng, config, config.pass_s))
    if flight < 1:
        return False
    for step, f in enumerate(range(stop - flight, stop)):
        # Also overrides the tail of the previous turnover's chase when the spell is very short.
        frames[f].update(carrier=None, flight=(passer, winner, (step + 1) / float(flight + 1)),
                         event="interception", contested=False, confused=False, presser=None)
    return True


def _schedule_spell(rng, config, frames, start, end, team, first_carrier, event_rng=None):
    """Carrier and passes for one H/A spell over [start, end)."""
    carrier = first_carrier
    t = start
    while t < end:
        hold_end = min(end, t + _frames(rng, config, config.hold_s))
        for f in range(t, hold_end):
            frames[f]["carrier"] = carrier
        t = hold_end
        flight = _frames(rng, config, config.pass_s)
        if t + flight >= end:
            continue
        receiver = rng.choice([p for p in _team_players(team) if p != carrier])
        for step, f in enumerate(range(t, t + flight)):
            frames[f]["flight"] = (carrier, receiver, (step + 1) / float(flight + 1))
        if config.deflection_prob > 0:
            _deflect(event_rng, config, frames, t, flight, team)
        t += flight
        carrier = receiver
    return carrier


def _kickoff(frames, start, stop, carrier):
    for f in range(start, stop):
        frames[f].update(label=DEAD, dead="kickoff", restart=carrier, contested=False,
                         presser=None, confused=False)


def simulate_latent(config, rng, event_rng=None, goal_rng=None):
    """Per-frame truth: label, who has the ball, dead-ball kind, contested flag, presser.

    event_rng (rng if not given) drives interceptions and deflections and goal_rng (event_rng if
    not given) goals; each is only drawn from when its rate is above zero, so the default process
    is unchanged."""
    event_rng = event_rng or rng
    goal_rng = goal_rng or event_rng
    blank = {"label": UNKNOWN, "carrier": None, "flight": None, "dead": None, "contested": False,
             "presser": None, "confused": False, "restart": None, "event": None}
    frames = [dict(blank) for _ in range(config.n_frames)]
    contest = max(1, int(round(config.contest_s * config.fps)))
    weights = (config.dead_out_observed_weight, config.dead_out_hidden_weight,
               config.dead_foul_weight)
    goal_per_frame = config.goal_rate_per_90 / (90.0 * 60.0 * config.fps)
    kickoff_team = rng.choice((HOME, AWAY))

    for half, first, last in half_blocks(config):
        end_of_half = last + 1
        t = first
        team = kickoff_team if half == 1 else _other(kickoff_team)
        carrier = _outfield(rng, team)
        if half == 2 or config.first_half_kickoff:
            length = _frames(rng, config, config.prekickoff_s)
            _kickoff(frames, t, t + length, carrier)
            t += length
        while t < end_of_half:
            stop = min(end_of_half, t + _spell_frames(rng, config))
            for f in range(t, stop):
                frames[f]["label"] = team
            last_carrier = _schedule_spell(rng, config, frames, t, stop, team, carrier, event_rng)
            spell_start, t = t, stop
            if t >= end_of_half:
                break
            if goal_per_frame > 0 and goal_rng.random() < -math.expm1(
                    -goal_per_frame * (t - spell_start)):
                # The spell ends in a goal; the conceding team kicks off from the centre spot.
                stop = min(end_of_half, t + _frames(goal_rng, config, config.goal_kickoff_s))
                team = _other(team)
                carrier = _outfield(goal_rng, team)
                _kickoff(frames, t, stop, carrier)
                t = stop
            elif rng.random() < config.dead_ball_prob:
                kind = rng.choices(("out_observed", "out_hidden", "foul"), weights)[0]
                stop = min(end_of_half, t + _frames(rng, config, config.dead_ball_s))
                team = rng.choice((HOME, AWAY))
                carrier = _outfield(rng, team)
                for f in range(t, stop):
                    frames[f].update(label=DEAD, dead=kind, restart=carrier, presser=None,
                                     confused=False)
                t = stop
            else:
                # Turnover: an opponent closes in, wins it, and the loser chases.
                winner = _outfield(rng, _other(team))
                confused = rng.random() < config.handover_confusion_prob
                intercepted = config.interception_prob > 0 and _intercept(
                    event_rng, config, frames, spell_start, t, last_carrier, winner)
                # Nobody presses an intercepted pass in flight; the loser still chases after it.
                contest_start = t if intercepted else max(first, t - contest)
                for f in range(contest_start, min(end_of_half, t + contest)):
                    if frames[f]["dead"] is None:
                        frames[f].update(contested=True, confused=confused, event=None,
                                         presser=winner if f < t else last_carrier)
                team, carrier = _other(team), winner
    return frames


def _clamp(value, low, high):
    return min(high, max(low, value))


def _dribble(config, rng, carry, carrier, position):
    """("dribble" only) ball position for a carried frame and the updated carry state.

    carry is None or [carrier, dx, dy, held_frames_left, held_spot]; a new carrier starts afresh.
    """
    low, high = config.dribble_reach_m
    if carry is None or carry[0] != carrier:
        reach = rng.uniform(low, high)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        carry = [carrier, reach * math.cos(angle), reach * math.sin(angle), 0, None]
        if carrier % PLAYERS_PER_TEAM == 0 and rng.random() < config.keeper_hold_prob:
            reach = rng.uniform(0.0, config.keeper_hold_reach_m)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            carry[3] = _frames(rng, config, config.keeper_hold_s)
            carry[4] = (position[0] + reach * math.cos(angle),
                        position[1] + reach * math.sin(angle))
    if carry[3] > 0:
        carry[3] -= 1
        return carry[4], carry
    dx = carry[1] + rng.gauss(0.0, config.dribble_drift_m)
    dy = carry[2] + rng.gauss(0.0, config.dribble_drift_m)
    reach = math.hypot(dx, dy)
    if reach < 1e-9:
        dx, dy, reach = 1.0, 0.0, 1.0
    scale = _clamp(reach, low, high) / reach
    carry[1], carry[2] = dx * scale, dy * scale
    return (position[0] + carry[1], position[1] + carry[2]), carry


def simulate_kinematics(config, rng, frames):
    """True player positions (metres) and ball position per graded frame; None elsewhere."""
    length, width = config.pitch_length, config.pitch_width
    n_players = 2 * PLAYERS_PER_TEAM
    blocks = half_blocks(config)
    players = [None] * config.n_frames
    balls = [None] * config.n_frames
    wander = [[0.0, 0.0] for _ in range(n_players)]
    shift = {HOME: 0.0, AWAY: 0.0}
    centre = 0.5
    ball = (length / 2, width / 2)
    out_spot = None
    carry = None  # "dribble" only: the current carrier's ball offset

    for half, first, last in blocks:
        # Home attacks +x in the first half, -x in the second.
        home_dir = 1 if half == 1 else -1
        for t in range(first, last + 1):
            frame = frames[t]
            label = frame["label"]
            for team in (HOME, AWAY):
                target = 0.08 if label == team else -0.05 if label in (HOME, AWAY) else 0.0
                shift[team] = 0.97 * shift[team] + 0.03 * target
            centre = 0.98 * centre + 0.02 * (ball[0] / length)

            positions = []
            for p in range(n_players):
                team = HOME if p < PLAYERS_PER_TEAM else AWAY
                direction = home_dir if team == HOME else -home_dir
                ax, ay = FORMATION[p % PLAYERS_PER_TEAM]
                ax = ax + shift[team]
                if frame["dead"] == "kickoff":
                    ax = min(ax, 0.45)  # both teams in their own half before kickoff
                x = ax if direction == 1 else 1.0 - ax
                if p % PLAYERS_PER_TEAM:
                    x += 0.5 * (centre - 0.5)
                wander[p][0] = 0.98 * wander[p][0] + rng.gauss(0.0, 0.15)
                wander[p][1] = 0.98 * wander[p][1] + rng.gauss(0.0, 0.15)
                positions.append([_clamp(x * length + wander[p][0], 0.5, length - 0.5),
                                  _clamp(ay * width + wander[p][1], 0.5, width - 0.5)])

            dead = frame["dead"]
            if dead is not None or frame["flight"]:
                carry = None
            if dead is None:
                out_spot = None
                if frame["flight"]:
                    passer, receiver, progress = frame["flight"]
                    (px, py), (rx, ry) = positions[passer], positions[receiver]
                    ball = (px + (rx - px) * progress, py + (ry - py) * progress)
                elif config.carry_model == "dribble":
                    carrier = frame["carrier"]
                    ball, carry = _dribble(config, rng, carry, carrier, positions[carrier])
                    if ball == carry[4]:
                        # Goalkeeper standing over a held ball, still live and still theirs.
                        positions[carrier] = [ball[0] - 0.1, ball[1]]
                else:
                    cx, cy = positions[frame["carrier"]]
                    reach = rng.uniform(0.3, 1.2)
                    angle = rng.uniform(0.0, 2.0 * math.pi)
                    ball = (cx + reach * math.cos(angle), cy + reach * math.sin(angle))
            elif dead == "kickoff":
                ball = (length / 2, width / 2)
                positions[frame["restart"]] = [length / 2 - 0.8 * home_dir, width / 2 + 0.3]
            elif dead == "foul":
                positions[frame["restart"]] = [ball[0] + 0.6, ball[1] + 0.4]
            else:
                if out_spot is None:
                    # Over the nearest touch line or goal line, by one to four metres.
                    beyond = rng.uniform(1.0, 4.0)
                    gaps = ((ball[1], (ball[0], -beyond)),
                            (width - ball[1], (ball[0], width + beyond)),
                            (ball[0], (-beyond, ball[1])),
                            (length - ball[0], (length + beyond, ball[1])))
                    out_spot = min(gaps)[1]
                ball = (ball[0] + 0.25 * (out_spot[0] - ball[0]),
                        ball[1] + 0.25 * (out_spot[1] - ball[1]))

            presser = frame["presser"]
            if presser is not None:
                # Contested: the challenger is within a stride; confused hand-overs put them
                # nearer the ball than whoever actually has it, and deflections within touching
                # distance of a pass.
                close = frame["confused"] or frame["event"] == "deflection"
                distance = rng.uniform(0.1, 0.4) if close else rng.uniform(0.4, 2.0)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                positions[presser] = [ball[0] + distance * math.cos(angle),
                                      ball[1] + distance * math.sin(angle)]
            players[t] = positions
            balls[t] = ball
    return players, balls


# -- observation model ---------------------------------------------------------------------------

def _ball_rates(config, frames):
    """Per-frame dropout target, calibrated so graded frames average config.ball_missing_rate."""
    graded = [f for f in frames if f["label"] != UNKNOWN]

    def rate(frame, base):
        if frame["dead"] == "out_hidden" or (frame["dead"] == "out_observed" and frame["late"]):
            return config.dead_hidden_missing_rate
        if frame["contested"]:
            return min(0.95, base * config.contested_missing_boost)
        return base

    low, high = 0.0, 1.0
    for _ in range(40):  # the mean is monotone in the base rate, so bisect it
        base = (low + high) / 2
        mean = sum(rate(f, base) for f in graded) / max(1, len(graded))
        low, high = (base, high) if mean < config.ball_missing_rate else (low, base)
    base = min(0.95, (low + high) / 2)
    return [rate(f, base) if f["label"] != UNKNOWN else 1.0 for f in frames]


def _mark_late_dead(config, frames):
    """Out-of-bounds stoppages: the ball is observed beyond the line for a while, then lost."""
    seen = 0
    for frame in frames:
        frame["late"] = False
        if frame["dead"] == "out_observed":
            seen += 1
            frame["late"] = seen > 2 * config.fps
        else:
            seen = 0


def observe(config, rng, frames, players, balls):
    """Turn latent truth into per-frame (slot positions, ball) observations in metres."""
    _mark_late_dead(config, frames)
    rates = _ball_rates(config, frames)
    leave = 1.0 / (config.ball_missing_run_s * config.fps)
    n_players = 2 * PLAYERS_PER_TEAM

    # Slots belong to a team for good; which person a slot shows does not.
    slot_team = [HOME] * (config.n_slots // 2) + [AWAY] * (config.n_slots - config.n_slots // 2)
    rng.shuffle(slot_team)
    free = {team: [s for s, owner in enumerate(slot_team) if owner == team] for team in (HOME, AWAY)}
    slot_of = []
    for p in range(n_players):
        slot_of.append(free[HOME if p < PLAYERS_PER_TEAM else AWAY].pop(0))
    gone_until = [0] * n_players
    duplicates = []  # (slot, person, until, dx, dy)

    per_frame = 1.0 / (60.0 * config.fps)
    vanish = config.track_vanish_per_min * per_frame
    swap = config.track_swap_per_min * per_frame
    duplicate = config.duplicate_per_min * per_frame

    observations = [None] * config.n_frames
    missing = False
    for t, frame in enumerate(frames):
        if players[t] is None:
            observations[t] = ([None] * config.n_slots, None)
            continue
        positions = players[t]

        for p in range(n_players):
            if slot_of[p] is None and t >= gone_until[p]:
                team = HOME if p < PLAYERS_PER_TEAM else AWAY
                if free[team]:
                    slot_of[p] = free[team].pop(rng.randrange(len(free[team])))
            elif slot_of[p] is not None and rng.random() < vanish:
                free[slot_team[slot_of[p]]].append(slot_of[p])
                slot_of[p] = None
                gone_until[p] = t + _frames(rng, config, config.track_vanish_s)
        if rng.random() < swap:
            a = rng.randrange(n_players)
            others = [p for p in range(n_players) if p != a and slot_of[p] is not None]
            if slot_of[a] is not None and others:
                b = min(others, key=lambda p: (positions[p][0] - positions[a][0]) ** 2
                        + (positions[p][1] - positions[a][1]) ** 2)
                slot_of[a], slot_of[b] = slot_of[b], slot_of[a]
        for expired in [d for d in duplicates if d[2] <= t]:
            free[slot_team[expired[0]]].append(expired[0])
        duplicates = [d for d in duplicates if d[2] > t]
        if rng.random() < duplicate:
            person = rng.randrange(n_players)
            team = HOME if person < PLAYERS_PER_TEAM else AWAY
            if rng.random() < config.duplicate_other_team_prob:
                team = _other(team)
            if free[team]:
                slot = free[team].pop(rng.randrange(len(free[team])))
                duplicates.append((slot, person, t + _frames(rng, config, config.duplicate_s),
                                   rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)))

        slots = [None] * config.n_slots
        for p in range(n_players):
            if slot_of[p] is not None:
                slots[slot_of[p]] = (positions[p][0] + rng.gauss(0.0, config.player_noise_m),
                                     positions[p][1] + rng.gauss(0.0, config.player_noise_m), p)
        for slot, person, _, dx, dy in duplicates:
            slots[slot] = (positions[person][0] + dx, positions[person][1] + dy, person)

        # Two-state dropout chain whose stationary missing rate tracks rates[t].
        target = rates[t]
        enter = min(1.0, leave * target / max(1e-9, 1.0 - target))
        missing = rng.random() >= leave if missing else rng.random() < enter
        ball = None
        if not missing:
            ball = (balls[t][0] + rng.gauss(0.0, config.ball_noise_m),
                    balls[t][1] + rng.gauss(0.0, config.ball_noise_m))
        observations[t] = (slots, ball)
    return observations, slot_team


# -- tagging -------------------------------------------------------------------------------------

def _restart_kind(config, dead, ball, team, home_dir):
    """Set piece for a stoppage, from where the ball last was and who restarts."""
    if dead == "kickoff":
        return KICK_OFF
    if dead == "foul":
        return FREE_KICK
    x, y = ball
    gaps = (y, config.pitch_width - y, x, config.pitch_length - x)
    line = gaps.index(min(gaps))
    if line < 2:
        return THROW_IN
    attacks_plus_x = (team == HOME) == (home_dir == 1)
    return CORNER if attacks_plus_x == (line == 3) else GOAL_KICK


def true_restarts(config, frames, balls):
    """[(frame, kind, team)]: the first live frame after each graded dead stretch."""
    restarts = []
    for half, first, last in half_blocks(config):
        home_dir = 1 if half == 1 else -1
        for t in range(first + 1, last + 1):
            if frames[t - 1]["label"] == DEAD and frames[t]["label"] in (HOME, AWAY):
                team = frames[t]["label"]
                kind = _restart_kind(config, frames[t - 1]["dead"], balls[t - 1], team, home_dir)
                restarts.append((t, kind, team))
    return restarts


def simulate_tagging(config, rng, frames, balls):
    """Noisy restart tags [{start, end, kind, team}] inside the halves, in window order."""
    blocks = [(first, last) for _, first, last in half_blocks(config)]

    def window(centre):
        first, last = next(block for block in blocks if block[0] <= centre <= block[1])
        length = _frames(rng, config, config.tag_window_s)
        start = int(round(centre + rng.gauss(0.0, config.tag_jitter_s) * config.fps - length / 2))
        start = _clamp(start, first, last)
        return start, min(last + 1, start + length)

    def team_tag(kind, team):
        if kind == THROW_IN or rng.random() < config.tag_na_team_prob:
            return None
        return _other(team) if rng.random() < config.tag_wrong_team_prob else team

    tags = []
    for frame, kind, team in true_restarts(config, frames, balls):
        if rng.random() < config.tag_miss_prob:
            continue
        start, end = window(frame)
        tags.append({"start": start, "end": end, "kind": kind, "team": team_tag(kind, team)})
    per_frame = config.tag_false_per_min / (60.0 * config.fps)
    for first, last in blocks:
        t = first + (rng.expovariate(per_frame) if per_frame else math.inf)
        while t <= last:
            kind = rng.choice(FALSE_TAG_KINDS)
            start, end = window(int(t))
            tags.append({"start": start, "end": end, "kind": kind,
                         "team": team_tag(kind, rng.choice((HOME, AWAY)))})
            t += rng.expovariate(per_frame)
    return sorted(tags, key=lambda tag: (tag["start"], tag["end"], tag["kind"]))


# -- writing -------------------------------------------------------------------------------------

def _group(x, y, speed, config):
    return "{:.5f},{:.5f},{:.2f}".format(x / config.pitch_length, y / config.pitch_width, speed)


def _speed(now, before, fps):
    if before is None:
        return 0.0
    return math.hypot(now[0] - before[0], now[1] - before[1]) * fps


def latent_cause(frame):
    """One CAUSES entry for a latent frame. Draws nothing, so the other files are unchanged."""
    if frame["label"] == UNKNOWN:
        return "ungraded"
    if frame["dead"] == "kickoff":
        return "kickoff"
    if frame["dead"] is not None:
        return "dead_" + frame["dead"]
    if frame["contested"]:
        return "contested"
    if frame["event"]:
        return frame["event"]
    return "flight" if frame["flight"] else "hold"


def file_names(seed):
    stem = "SYN-{}".format(seed)
    return {"atd_raw": stem + "_atd.txt", "atd_meta": stem + "_atd.xml",
            "gt": stem + "_possession_gt.csv", "latent": stem + "_latent.csv",
            "tagging": stem + "_tagging.json", "manifest": stem + "_manifest.json"}


KICKOFF_FIELDS = ("first_half_kickoff", "goal_rate_per_90", "goal_kickoff_s")


def manifest_config(config):
    """asdict(config), without KICKOFF_FIELDS while both kick-off options are off, so default
    manifests keep their earlier bytes."""
    values = asdict(config)
    if not config.first_half_kickoff and config.goal_rate_per_90 == 0:
        for name in KICKOFF_FIELDS:
            del values[name]
    return values


def write_bundle(out_dir, seed, config=SyntheticConfig()):
    """Generate and write one bundle. Returns the manifest dict with absolute file paths."""
    check_paths([out_dir])
    latent_rng = random.Random("{}:latent".format(seed))
    observe_rng = random.Random("{}:observe".format(seed))
    frames = simulate_latent(config, latent_rng, random.Random("{}:events".format(seed)),
                             random.Random("{}:goals".format(seed)))
    players, balls = simulate_kinematics(config, latent_rng, frames)
    observations, slot_team = observe(config, observe_rng, frames, players, balls)
    tags = simulate_tagging(config, random.Random("{}:tagging".format(seed)), frames, balls)

    os.makedirs(out_dir, exist_ok=True)
    names = file_names(seed)
    paths = {key: os.path.join(out_dir, name) for key, name in names.items()}
    missing = "NaN,NaN,NaN"

    with open(paths["atd_raw"], "w", newline="\n") as handle:
        last_slot = [None] * config.n_slots
        last_ball = None
        for t, (slots, ball) in enumerate(observations):
            groups = []
            for s, entry in enumerate(slots):
                if entry is None:
                    groups.append(missing)
                    last_slot[s] = None
                    continue
                groups.append(_group(entry[0], entry[1],
                                     _speed(entry, last_slot[s], config.fps), config))
                last_slot[s] = entry
            ball_text = missing if ball is None else _group(
                ball[0], ball[1], _speed(ball, last_ball, config.fps), config)
            last_ball = ball
            handle.write("{}:{}:{}\n".format(t + 1, ";".join(groups), ball_text))

    with open(paths["atd_meta"], "w", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<main>\n<Metadata>\n<Teams>\n'
                     '<Team id="SYN_H"><Name>Synthetic Home</Name></Team>\n'
                     '<Team id="SYN_A"><Name>Synthetic Away</Name></Team>\n</Teams>\n<Players>\n')
        for s, team in enumerate(slot_team):
            handle.write('<Player id="P{0}" teamId="SYN_{1}"><Name>Track {0}</Name>'
                         '<ShirtNumber>{0}</ShirtNumber></Player>\n'.format(s + 1, team))
        handle.write("</Players>\n</Metadata>\n</main>\n")

    with open(paths["gt"], "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(t, frame["label"]) for t, frame in enumerate(frames))

    with open(paths["latent"], "w", newline="\n") as handle:
        handle.write("frame,cause,ball_missing\n")
        handle.writelines("{},{},{}\n".format(t, latent_cause(frame), int(ball is None))
                          for t, (frame, (_, ball)) in enumerate(zip(frames, observations)))

    with open(paths["tagging"], "w", newline="\n") as handle:
        json.dump(tags, handle, indent=1, sort_keys=True)
        handle.write("\n")

    manifest = {
        "format": FORMAT,
        "seed": seed,
        "config": manifest_config(config),
        "n_frames": config.n_frames,
        "fps": config.fps,
        "pitch": [config.pitch_length, config.pitch_width],
        "blocks": [list(block) for block in half_blocks(config)],
        "files": names,
    }
    with open(paths["manifest"], "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return dict(manifest, paths=paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True, help="directory to write the bundle into")
    parser.add_argument("--config", help="JSON file of SyntheticConfig overrides")
    parser.add_argument("--n-frames", type=int, help="override the configured frame count")
    args = parser.parse_args(argv)
    guard_paths([args.out, args.config])
    overrides = {"n_frames": args.n_frames} if args.n_frames else {}
    manifest = write_bundle(args.out, args.seed, load_config(args.config, **overrides))
    print("synthetic bundle (not match data): seed {} -> {}".format(args.seed, args.out))
    for key in sorted(manifest["files"]):
        print("  {}".format(manifest["files"][key]))


if __name__ == "__main__":
    main()
