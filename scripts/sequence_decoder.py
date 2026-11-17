"""Experimental tracking-only possession sequence decoder, beside the fixed official baseline.

The official baseline (challenge/starter/baseline.py) says D whenever the tracking loses the ball,
so every occlusion becomes an invented dead-ball stretch that splits a possession. This decoder
starts from the same per-frame evidence -- the team of the nearest tracked player to the ball --
and adds possession continuity:

  1. Evidence per frame: H or A (nearest player's team), D only where there is positive dead-ball
     evidence (the ball observed beyond a touch or goal line), and nothing where the ball or every
     player is missing, or where the nearest player is more than `control_radius` metres away
     (a ball in flight is no evidence for whoever it happens to pass closest to).
     Opt-in (`stationary_ball_frames` >= 1): a ball that stays within `stationary_ball_radius`
     of where it stopped for that many frames inside a half is also D evidence, after a
     `stationary_ball_lead_in` that keeps a player briefly standing on the ball in possession.
     `stationary_ball_anchor` "centroid" measures that radius from the run's running mean
     rather than its first noisy observation. Opt-in (`stationary_backfill_frames` >= 1): a run
     at least that long is D from its first frame, without the lead-in.
     Opt-in (`lost_at_line_frames` >= 1): a missing-ball run of that many frames is D when the
     ball was last seen inside the pitch, within `lost_at_line_distance` of a line and moving out
     over it at `lost_at_line_min_speed` or more. A missing ball alone is never D.
     Opt-in (`max_control_speed` >= 0): a ball moving faster than that many m/s is no team
     evidence either (a firm pass skimming past an opponent is not their possession). Speed is
     the displacement from the earliest observation in the previous `control_speed_span` frames
     of the same half over the elapsed time, wide enough that a carried ball's jitter stays slow.
     It runs before the dead-ball rules and leaves D and missing frames alone.
     With `fast_ball_weight` >= 0 (Viterbi only) the fast frames keep their team evidence, but
     contradicting it costs that factor times its usual weight instead of being deleted.
     With `fast_ball_require_observed`, a frame is fast only when the ball was observed on every
     frame of its span window, so a ball reappearing after a gap is not fast from a stale point.
     Opt-in (`ball_interp_max_gap` >= 1): a missing-ball run of at most that many frames inside
     a half, with both neighbouring observations in bounds, is filled with linearly interpolated
     ball positions, and each of those frames still without evidence gets the team of the nearest
     player observed on it, within `control_radius`. It runs last and never adds D.
     With `ball_interp_max_speed` >= 0 a run is filled only when the ball moved at most that
     many m/s at both edges (measured as for `max_control_speed`, over `control_speed_span`
     frames back from the last observation before it and forward from the first after it), so a
     carried ball is bridged and a pass in flight is not.
     `ball_interp_mode` "agree" keeps such a call only when its team equals the observed
     evidence on the frame just before or just after that run, so a fill can extend a spell
     but never start one.
     Opt-in (`kickoff_spot_radius` >= 0): inside a half, a run of at least `kickoff_min_frames`
     observed frames whose ball lies within that radius of the centre spot, with each team's
     tracked players on opposite sides of halfway (at most KICKOFF_EXEMPT_PLAYERS within
     KICKOFF_EXEMPT_RADIUS of the ball excepted), is D evidence: the ball is waiting for a kick
     off. Missing frames inside it are tolerated up to `stationary_ball_max_gap`, or up to
     `kickoff_bridge_gap` (opt-in, >= 1) since the observed ball on both sides of a gap is on
     the spot; the run still ends at its last on-spot observation, so D never reaches the first
     observed off-spot frame. It is independent of the stationary-ball rule and has no lead-in.
  2. Within each half, a centred majority vote over the observed team frames removes flicker.
     Missing frames stay missing, so gaps remain explicit.
  3. Gaps of at most `max_gap` frames with the same label on both sides are bridged with it.
     A short gap between different labels is a hand-over: split at its midpoint, abstain, or
     (`transition_fill` "hold") keep the left label until the right one is first observed, as
     challenge/CHALLENGE.md keeps a loose ball with the last team. D follows the same rule, so a
     team->D gap stays with the team and a D->team gap stays D.
     Longer gaps, and gaps touching a half boundary, are unknown (X) by default. `long_gap_fill`
     "hold" fills them as the scorer's forward-fill would (the left label, or the right label at
     the head of a half; X only in a half with no labels at all), and "dead" writes D in them.
     Opt-in (`min_dead_run` >= 1): after the gaps are filled, a D run shorter than that many
     frames whose nearest H/A labels on both sides (skipping X) are the same team becomes that
     team, so a brief false stoppage does not split one possession into two spells. D runs at a
     half's edge or between different teams are kept.
     Opt-in (`max_dead_bridge` >= 1): before that merge, a non-D run (H, A or X) of at most
     that many frames with D on both sides becomes D, so one stoppage is one D spell rather than
     several split by a stray team call. Runs at a half's edge are kept.
     Opt-in (`handover_search` >= 1): last, each H<->A change inside a half moves to just after
     the last frame within +/- that many frames on which the outgoing team has a player within
     `control_radius` of the observed ball, so a hand-over called while the old team still has
     the ball is delayed until it loses control. The change stays between the two team runs it
     separates (each keeps a frame), so it never crosses a D or X run; with no such frame it
     is kept.
  4. Halves are decoded independently; frames outside both halves are written as X (ungraded).

`--segmenter viterbi` replaces the vote in step 2 with a minimum-cost H/A/D segmentation of each
half: a contradicted frame costs 1, a team label on a D-evidence frame costs DEAD_EVIDENCE_COST,
missing frames cost nothing, and every change of state costs `switch_cost` frames (half that for
a change to or from D). Missing frames are then blanked and step 3 runs unchanged, so long gaps
and half edges still follow `long_gap_fill`. `vote_window` is unused by it.
Opt-in (`contested_switch_factor` >= 0): an H<->A change into a contested frame, where players of
both teams are within `control_radius` of the observed ball, costs that factor times
`switch_cost`, so a brief contested possession by the other team is cheaper to call.

`--evidence-weighting margin` (Viterbi only) makes ambiguous frames cheaper to contradict. A frame
whose nearest player is barely closer to the ball than the nearest opponent is weak evidence, so
contradicting it (or putting D on it) costs min(1, (d_opp - d_own) / margin_scale), floored at
MIN_EVIDENCE_WEIGHT. The default `hard` keeps every contradiction at 1.

Assumptions, all configurable and none tuned on the supplied match or its public ground truth:
  - ATD coordinates are normalised 0-1 over the pitch, origin at a corner, as DATA.md describes;
    a ball more than `out_of_bounds_margin` metres outside that box is out of play.
  - The defaults (1 s vote, 2 s bridge, 1 m margin, 3 m control radius) are a-priori football
    scales, not fitted. A negative margin, control radius or control speed disables that rule.

Standard library only, so it runs and tests without numpy. It mirrors the baseline's raw ATD
parsing rules; tests/test_sequence_decoder.py checks the evidence agrees with the baseline when
numpy is available.

    python scripts/sequence_decoder.py --out outputs/sequence/sequence_decoder.csv

--smart-tagging adds restart evidence from the Smart Tagging (scripts/tagging_evidence.py); the
default stays tracking-only. Prints only parameters, aggregate tagging counts and the output
path, never tracking records, tags or labels.

--config-file PATH takes the defaults from a JSON config: a flat dict of DecoderConfig fields
with an optional "smart_tagging" (true or a dict of TaggingConfig fields), or a
scripts/decoder_experiments.py JSONL log, whose record --record N (default: the last) supplies
its "chosen_config". Flags given on the command line still override it.
"""

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "challenge"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

HOME, AWAY, DEAD, UNKNOWN = "H", "A", "D", "X"
TEAMS = (HOME, AWAY)

# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", groups "x,y,speed", "NaN" when missing.
# Same format rules as the official baseline.
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

SEGMENTERS = ("vote", "viterbi")
TRANSITION_FILLS = ("midpoint", "unknown", "hold")
LONG_GAP_FILLS = ("unknown", "hold", "dead")
EVIDENCE_WEIGHTINGS = ("hard", "margin")
STATIONARY_ANCHORS = ("first", "centroid")
BALL_INTERP_MODES = ("any", "agree")
# Smallest margin weight, so even a tie between the two teams' nearest players costs something.
MIN_EVIDENCE_WEIGHT = 0.05
# Viterbi emission cost of a team state on a D-evidence frame: a ball beyond a line is strong.
DEAD_EVIDENCE_COST = 5.0
# Widest frame gap between the two observations that give a lost ball's last velocity
# (tagging_evidence.TaggingConfig.speed_max_gap's default).
LOST_AT_LINE_SPEED_GAP = 5
# Kick-off formation: up to this many players within this many metres of the ball (the taker and
# a team mate, or an opponent closing in) may stand anywhere; everyone else must be on their
# team's side of halfway.
KICKOFF_EXEMPT_PLAYERS = 2
KICKOFF_EXEMPT_RADIUS = 3.0
# Config key for restart tagging settings, as decoder_experiments.config_record() writes it.
SMART_TAGGING = "smart_tagging"


@dataclass(frozen=True)
class DecoderConfig:
    vote_window: int = 25            # +/- frames of majority vote over observed team evidence
    max_gap: int = 50                # longest run of missing evidence bridged, in frames
    transition_fill: str = "midpoint"  # short gap between labels: "midpoint", "unknown" or "hold"
    out_of_bounds_margin: float = 1.0  # metres beyond a line before the ball counts as out; <0 disables
    control_radius: float = 3.0      # metres to the nearest player beyond which no team; <0 disables
    stationary_ball_frames: int = -1  # shortest still-ball run that means dead; <0 disables
    stationary_ball_radius: float = 0.5  # metres the ball may stray from where it stopped
    stationary_ball_lead_in: int = 25  # first frames of a stationary run left as they are
    stationary_ball_max_gap: int = 5   # missing-ball frames tolerated inside a stationary run
    lost_at_line_frames: int = -1    # shortest missing-ball run after an outward ball; <0 disables
    lost_at_line_distance: float = 2.0  # metres from a line at the last observation
    lost_at_line_min_speed: float = 2.0  # m/s of last observed velocity towards that line
    segmenter: str = "vote"          # within-half smoothing: "vote" or "viterbi"
    switch_cost: float = 25.0        # viterbi: penalty per H<->A change, in frames of contradiction
    evidence_weighting: str = "hard"  # viterbi: "hard" (each contradiction costs 1) or "margin"
    margin_scale: float = 2.0        # metres of own-vs-opponent margin that give full weight
    long_gap_fill: str = "unknown"   # long gap or half edge: "unknown" (X), "hold" or "dead" (D)
    max_control_speed: float = -1.0  # m/s above which the ball is no team evidence; <0 disables
    control_speed_span: int = 10     # frames back to the observation the speed is measured from
    stationary_ball_anchor: str = "first"  # still-ball radius from: "first" point or "centroid"
    ball_interp_max_gap: int = -1    # longest missing-ball run interpolated; <0 disables
    min_dead_run: int = -1           # shorter D runs inside one team's possession join it; <0 disables
    max_dead_bridge: int = -1        # non-D runs this short between two D runs become D; <0 disables
    kickoff_spot_radius: float = -1.0  # metres from the centre spot of a kick-off ball; <0 disables
    kickoff_min_frames: int = 25     # shortest run of observed frames on the spot that means dead
    fast_ball_weight: float = -1.0   # viterbi: fast-ball evidence weight factor, not a gate; <0 disables
    stationary_backfill_frames: int = -1  # still-ball runs this long skip the lead-in; <0 disables
    handover_search: int = -1        # H<->A change to old team's last control +/- N; <0 disables
    ball_interp_mode: str = "any"    # interpolated team calls: "any", or "agree" with a gap side
    ball_interp_max_speed: float = -1.0  # m/s edge ball speed above which no fill; <0 disables
    fast_ball_require_observed: bool = False  # fast only if the whole speed span was observed
    kickoff_bridge_gap: int = -1     # missing frames bridged between on-spot kick-off balls; <0 off
    contested_switch_factor: float = -1.0  # viterbi: H<->A cost factor on contested frames; <0 disables
    control_motion_max_error: float = -1.0  # opt-in co-motion proxy, m/s; <0 preserves incumbent
    control_motion_lag: int = 5      # all observations required across this same-slot span
    control_motion_min_run: int = 3  # consecutive supported frames, offline confirmed

    def __post_init__(self):
        import math
        if not math.isfinite(self.control_motion_max_error) or self.control_motion_max_error == 0:
            raise ValueError('control_motion_max_error must be nonzero and finite')
        if any(type(v) is not int or v < 1 for v in (self.control_motion_lag, self.control_motion_min_run)):
            raise ValueError('motion observation lengths must be positive integers')
        if self.control_motion_max_error > 0 and self.control_radius <= 0:
            raise ValueError('motion evidence requires a positive control radius')
        if self.vote_window < 0 or self.max_gap < 0:
            raise ValueError("vote_window and max_gap must be >= 0")
        if self.transition_fill not in TRANSITION_FILLS:
            raise ValueError("transition_fill must be 'midpoint', 'unknown' or 'hold'")
        if self.long_gap_fill not in LONG_GAP_FILLS:
            raise ValueError("long_gap_fill must be 'unknown', 'hold' or 'dead'")
        if self.control_radius != self.control_radius:
            raise ValueError("control_radius must be a number; a negative value disables it")
        if self.stationary_ball_frames == 0:
            raise ValueError("stationary_ball_frames must be >= 1; a negative value disables it")
        if not self.stationary_ball_radius >= 0:
            raise ValueError("stationary_ball_radius must be a number >= 0")
        if self.stationary_ball_lead_in < 0 or self.stationary_ball_max_gap < 0:
            raise ValueError("stationary_ball_lead_in and stationary_ball_max_gap must be >= 0")
        if self.stationary_ball_anchor not in STATIONARY_ANCHORS:
            raise ValueError("stationary_ball_anchor must be 'first' or 'centroid'")
        if self.lost_at_line_frames == 0:
            raise ValueError("lost_at_line_frames must be >= 1; a negative value disables it")
        if not 0 <= self.lost_at_line_distance < float("inf"):
            raise ValueError("lost_at_line_distance must be a finite number >= 0")
        if not 0 <= self.lost_at_line_min_speed < float("inf"):
            raise ValueError("lost_at_line_min_speed must be a finite number >= 0")
        if self.segmenter not in SEGMENTERS:
            raise ValueError("segmenter must be 'vote' or 'viterbi'")
        if not 0 <= self.switch_cost < float("inf"):
            raise ValueError("switch_cost must be a finite number >= 0")
        if self.evidence_weighting not in EVIDENCE_WEIGHTINGS:
            raise ValueError("evidence_weighting must be 'hard' or 'margin'")
        if self.evidence_weighting == "margin" and self.segmenter != "viterbi":
            raise ValueError("evidence_weighting 'margin' needs the viterbi segmenter")
        if not 0 < self.margin_scale < float("inf"):
            raise ValueError("margin_scale must be a finite number > 0")
        if self.max_control_speed != self.max_control_speed:
            raise ValueError("max_control_speed must be a number; a negative value disables it")
        if self.control_speed_span < 1:
            raise ValueError("control_speed_span must be >= 1")
        if self.ball_interp_max_gap == 0:
            raise ValueError("ball_interp_max_gap must be >= 1; a negative value disables it")
        if self.min_dead_run == 0:
            raise ValueError("min_dead_run must be >= 1; a negative value disables it")
        if self.max_dead_bridge == 0:
            raise ValueError("max_dead_bridge must be >= 1; a negative value disables it")
        if not self.ball_interp_max_speed < float("inf"):
            raise ValueError("ball_interp_max_speed must be a finite number; a negative value "
                             "disables it")
        if self.ball_interp_mode not in BALL_INTERP_MODES:
            raise ValueError("ball_interp_mode must be 'any' or 'agree'")
        if not self.kickoff_spot_radius < float("inf"):
            raise ValueError("kickoff_spot_radius must be a finite number; a negative value "
                             "disables it")
        if self.kickoff_min_frames < 1:
            raise ValueError("kickoff_min_frames must be >= 1")
        if not self.fast_ball_weight < float("inf"):
            raise ValueError("fast_ball_weight must be a finite number; a negative value "
                             "disables it")
        if self.fast_ball_weight >= 0 and self.segmenter != "viterbi":
            raise ValueError("fast_ball_weight needs the viterbi segmenter")
        if self.stationary_backfill_frames == 0:
            raise ValueError("stationary_backfill_frames must be >= 1; a negative value "
                             "disables it")
        if self.handover_search == 0:
            raise ValueError("handover_search must be >= 1; a negative value disables it")
        if self.kickoff_bridge_gap == 0:
            raise ValueError("kickoff_bridge_gap must be >= 1; a negative value disables it")
        if not -float("inf") < self.contested_switch_factor <= 1:
            raise ValueError("contested_switch_factor must be a finite number <= 1; a negative "
                             "value disables it")
        if self.contested_switch_factor >= 0 and self.segmenter != "viterbi":
            raise ValueError("contested_switch_factor needs the viterbi segmenter")


def tagging_config(value):
    """A TaggingConfig from a config's "smart_tagging" value; None when tagging is off."""
    if value is None or value is False:
        return None
    from tagging_evidence import TaggingConfig
    if value is True:
        return TaggingConfig()
    if not isinstance(value, dict):
        raise ValueError("smart_tagging must be true, false or a dict of TaggingConfig fields")
    unknown = sorted(set(value) - {field.name for field in fields(TaggingConfig)})
    if unknown:
        raise ValueError("unknown smart_tagging config keys: {}".format(unknown))
    return TaggingConfig(**value)


def parse_config(entry):
    """(DecoderConfig, TaggingConfig or None) from a flat dict of DecoderConfig fields.

    The dict may add "smart_tagging" (see tagging_config); missing fields take their defaults.
    """
    if not isinstance(entry, dict):
        raise ValueError("a decoder config must be a JSON object of DecoderConfig fields")
    entry = dict(entry)
    tagging = tagging_config(entry.pop(SMART_TAGGING, None))
    unknown = sorted(set(entry) - {field.name for field in fields(DecoderConfig)})
    if unknown:
        raise ValueError("unknown decoder config keys: {}".format(unknown))
    return DecoderConfig(**entry), tagging


def load_config_file(path, record=None):
    """(DecoderConfig, TaggingConfig or None, config_source) from a config file.

    The file is one JSON object of decoder fields, or a decoder_experiments JSONL log; for a log,
    `record` indexes its records (from 0, negative from the end, default the last) and that
    record's "chosen_config" is used. Nothing from the file is printed.
    """
    with open(path) as handle:
        text = handle.read()
    try:
        whole = json.loads(text)
    except ValueError:
        whole = None
    if isinstance(whole, dict) and "chosen_config" not in whole:
        if record is not None:
            raise ValueError("{} is a single config, not an experiment log".format(path))
        return parse_config(whole) + ({"path": path},)
    records = [] if whole is None else [whole]
    if whole is None:
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    raise ValueError("{} line {} is not JSON".format(path, number))
    if not records:
        raise ValueError("{} has no records".format(path))
    index = len(records) - 1 if record is None else record
    if index < 0:
        index += len(records)
    if not 0 <= index < len(records):
        raise ValueError("{} has {} records; record {} is out of range".format(
            path, len(records), record))
    chosen = records[index]
    if not isinstance(chosen, dict) or "chosen_config" not in chosen:
        raise ValueError("{} record {} has no chosen_config".format(path, index))
    return parse_config(chosen["chosen_config"]) + ({"path": path, "record": index},)


def team_of_each_track(meta_path):
    """List over track slots: H or A. First team in the ATD metadata is home, as in the baseline."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return [HOME if team_id == team_ids[0] else AWAY for team_id in slots]


def _xy(group):
    first = group.index(",")
    return float(group[:first]), float(group[first + 1:group.index(",", first + 1)])


def read_evidence(raw_path, team_of_track, n_frames, pitch, margin, radius=-1.0):
    """Per-frame evidence: H/A from the nearest player, D for a ball beyond the lines, else None.

    `margin` and `radius` are in metres. A negative margin disables the dead-ball evidence; a
    negative radius gives the nearest player's team however far away they are.
    The default preserves older evidence-only callers; the decoder passes its configured radius.
    """
    length, width = pitch
    evidence = [None] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            frame = int(parts[0]) - 1  # ATD frame_id counts from 1, submissions from 0
            ball_x, ball_y = _xy(parts[2])
            ball_x, ball_y = ball_x * length, ball_y * width
            if margin >= 0 and not (-margin <= ball_x <= length + margin
                                    and -margin <= ball_y <= width + margin):
                evidence[frame] = DEAD
                continue
            best, best_team = None, None
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = _xy(group)
                distance = (x * length - ball_x) ** 2 + (y * width - ball_y) ** 2
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]
            if radius >= 0 and best is not None and best ** 0.5 > radius:
                continue  # nobody close enough to control the ball: no team evidence
            evidence[frame] = best_team
    return evidence


def read_team_control(raw_path, team_of_track, n_frames, pitch, margin, radius):
    """Per-frame frozenset of the teams with a player within `radius` metres of the ball.

    Empty where the ball is missing or more than `margin` metres beyond a line (a negative margin
    disables that); a negative radius counts every team with a tracked player.
    """
    length, width = pitch
    control = [frozenset()] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            ball_x, ball_y = _xy(parts[2])
            ball_x, ball_y = ball_x * length, ball_y * width
            if margin >= 0 and not (-margin <= ball_x <= length + margin
                                    and -margin <= ball_y <= width + margin):
                continue
            teams = set()
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = _xy(group)
                if radius < 0 or ((x * length - ball_x) ** 2
                                  + (y * width - ball_y) ** 2) ** 0.5 <= radius:
                    teams.add(team_of_track[slot])
            control[int(parts[0]) - 1] = frozenset(teams)
    return control


def read_evidence_weights(raw_path, team_of_track, n_frames, pitch, scale, margin=-1.0):
    """Per-frame confidence in the nearest-player evidence, in [MIN_EVIDENCE_WEIGHT, 1].

    With d_own the nearest player's distance to the ball and d_opp the distance of the other
    team's nearest player, the weight is min(1, (d_opp - d_own) / scale), all in metres. It is 1
    where the ball or every player is missing, the other team has no tracked player, or the ball
    is more than `margin` metres beyond a line (D evidence; a negative margin disables that).
    """
    length, width = pitch
    weights = [1.0] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            ball_x, ball_y = _xy(parts[2])
            ball_x, ball_y = ball_x * length, ball_y * width
            if margin >= 0 and not (-margin <= ball_x <= length + margin
                                    and -margin <= ball_y <= width + margin):
                continue
            nearest = {}
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = _xy(group)
                distance = (x * length - ball_x) ** 2 + (y * width - ball_y) ** 2
                team = team_of_track[slot]
                if distance < nearest.get(team, float("inf")):
                    nearest[team] = distance
            if len(nearest) < 2:
                continue  # nobody tracked, or only one team: nothing to weigh the call against
            own, opp = sorted(distance ** 0.5 for distance in nearest.values())
            weights[int(parts[0]) - 1] = max(MIN_EVIDENCE_WEIGHT, min(1.0, (opp - own) / scale))
    return weights


def align_weights(weights, raw_evidence, evidence):
    """`weights` with 1.0 wherever `evidence` is no longer the tracking evidence they describe.

    Stationary-ball, kick-off and Smart Tagging evidence replace some frames; those labels are not a
    nearest-player call, so contradicting them keeps the hard cost.
    """
    return [weight if new == old else 1.0
            for weight, old, new in zip(weights, raw_evidence, evidence)]


def ball_speed(ball, t, others, fps=GAMES[GAME]["fps"]):
    """m/s between the observed ball at `t` and the first observed frame in `others`, or None.

    `others` lists candidate frames in order of preference, the farthest first so a carried
    ball's jitter averages out. None when the ball at `t` or at every candidate is missing.
    """
    if ball[t] is None:
        return None
    for u in others:
        if ball[u] is not None:
            distance = math.hypot(ball[t][0] - ball[u][0], ball[t][1] - ball[u][1])
            return distance * fps / abs(t - u)
    return None


def fast_ball_frames(ball, blocks, config, fps=GAMES[GAME]["fps"]):
    """Frames inside a half where the ball moves faster than `max_control_speed`; empty if off.

    A frame's speed is its distance from the earliest observation in the previous
    `control_speed_span` frames of the same half, over the elapsed time. Frames with no such
    observation (or no ball) are never fast. Near a half's start, or after a missing-ball run of
    at least the span, the earliest observation is closer, so jitter weighs more there. With
    `fast_ball_require_observed`, a frame with any missing ball in that window is never fast.
    """
    fast = set()
    if config.max_control_speed < 0:
        return fast
    span = config.control_speed_span
    for _, first, last in blocks:
        for t in range(first, last + 1):
            window = range(max(first, t - span), t)
            if config.fast_ball_require_observed and any(ball[u] is None for u in window):
                continue
            speed = ball_speed(ball, t, window, fps)
            if speed is not None and speed > config.max_control_speed:
                fast.add(t)
    return fast


def gate_fast_ball(evidence, fast):
    """Evidence with the H/A calls on `fast` frames removed; D and missing frames are kept."""
    if not fast:
        return evidence
    return [None if t in fast and label in TEAMS else label
            for t, label in enumerate(evidence)]


def soft_fast_ball(config):
    """True when fast-ball frames are down-weighted (fast_ball_weights) rather than gated."""
    return config.fast_ball_weight >= 0 and config.max_control_speed >= 0


def fast_ball_weights(weights, raw_evidence, fast, factor):
    """Per-frame Viterbi weights with the team evidence on `fast` frames scaled by `factor`.

    `weights` are the margin weights, or None for hard evidence (1.0 on every frame). Only H/A
    tracking evidence is scaled; pass the result through align_weights so frames whose evidence
    was later replaced (stationary ball, Smart Tagging) keep the hard cost.
    """
    if weights is None:
        weights = [1.0] * len(raw_evidence)
    return [weight * factor if t in fast and label in TEAMS else weight
            for t, (weight, label) in enumerate(zip(weights, raw_evidence))]


def stationary_runs(ball, first, last, config):
    """(start, end) frame spans in [first, last] where the ball stays put; end is exclusive.

    A run is anchored on its first observed position, or with `stationary_ball_anchor` "centroid"
    on the running mean of its observations so far, which a single noisy first point cannot
    throw off. It continues while every observation is within `stationary_ball_radius` of the
    anchor and at most `stationary_ball_max_gap` missing frames separate consecutive
    observations; it spans first to last observation.
    """
    centroid = config.stationary_ball_anchor == "centroid"
    runs = []
    start = anchor = previous = None
    count = 0
    for t in range(first, last + 2):
        position = ball[t] if t <= last else None
        if position is None and t <= last:
            continue
        if anchor is not None and position is not None and \
                t - previous - 1 <= config.stationary_ball_max_gap and \
                math.hypot(position[0] - anchor[0], position[1] - anchor[1]) <= \
                config.stationary_ball_radius:
            previous = t
            if centroid:
                count += 1
                anchor = (anchor[0] + (position[0] - anchor[0]) / count,
                          anchor[1] + (position[1] - anchor[1]) / count)
            continue
        if anchor is not None and previous + 1 - start >= config.stationary_ball_frames:
            runs.append((start, previous + 1))
        start = anchor = previous = None
        if position is not None:
            start, anchor, previous, count = t, position, t, 1
    return runs


def stationary_dead(evidence, ball, blocks, config):
    """Evidence with D added where the ball sits still inside a half; a no-op when disabled.

    A ball that stays put for `stationary_ball_frames` is waiting for a restart (free kick,
    kick off, a ball placed for a goal kick); a carried ball keeps moving around the carrier's
    feet. The first `stationary_ball_lead_in` frames of each run keep their evidence, so a player
    who briefly stands on the ball before playing on is not cut out of their possession. A run of
    at least `stationary_backfill_frames` (when >= 1) is a stoppage, not a pause on the ball, so
    it is D from its first frame. Only frames that are not already D change; missing frames
    inside a run become D too.
    """
    if config.stationary_ball_frames < 0:
        return evidence
    out = list(evidence)
    for _, first, last in blocks:
        for start, end in stationary_runs(ball, first, last, config):
            backfill = 0 < config.stationary_backfill_frames <= end - start
            lead_end = start if backfill else min(end, start + config.stationary_ball_lead_in)
            out[lead_end:end] = [DEAD] * (end - lead_end)
    return out


def kickoff_formation(raw_path, team_of_track, n_frames, pitch):
    """Per frame: True when the tracked players stand in a kick-off formation, else False.

    That is: once at most KICKOFF_EXEMPT_PLAYERS of the players nearest the ball and within
    KICKOFF_EXEMPT_RADIUS metres of it are set aside, every remaining home player is on one side
    of halfway and every remaining away player on the other. Which side is not checked, so
    neither the ends teams defend nor a change of ends at half time needs to be known. False
    where the ball is missing or either team has no tracked player.
    """
    length, width = pitch
    halfway = length / 2.0
    ok = [False] * n_frames
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3 or parts[2][0] == MISSING:
                continue
            ball_x, ball_y = _xy(parts[2])
            ball_x, ball_y = ball_x * length, ball_y * width
            players = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = _xy(group)
                x, y = x * length, y * width
                players.append((math.hypot(x - ball_x, y - ball_y), x, team_of_track[slot]))
            players.sort()
            exempt = 0
            while exempt < min(KICKOFF_EXEMPT_PLAYERS, len(players)) and \
                    players[exempt][0] <= KICKOFF_EXEMPT_RADIUS:
                exempt += 1
            sides = {HOME: set(), AWAY: set()}
            for _, x, team in players[exempt:]:
                sides[team].add(x < halfway)
            if len(sides[HOME]) == len(sides[AWAY]) == 1 and sides[HOME] != sides[AWAY]:
                ok[int(parts[0]) - 1] = True
    return ok


def kickoff_runs(ball, first, last, config, pitch=GAMES[GAME]["pitch"], formation=None):
    """(start, end) frame spans in [first, last] where the ball waits on the centre spot.

    A run continues while every observed ball is within `kickoff_spot_radius` of the pitch
    centre (on a frame whose `formation` is True, when given) and at most
    `stationary_ball_max_gap` missing frames separate consecutive observations; with
    `kickoff_bridge_gap` >= 1, at most the larger of the two. Both sides of a bridged gap are
    on-spot observations, since an off-spot one ends the run. It spans first to last on-spot
    observation (end exclusive), so it stops before the first observed off-spot frame, and
    counts when it has at least `kickoff_min_frames` observed frames.
    """
    spot = (pitch[0] / 2.0, pitch[1] / 2.0)
    max_gap = max(config.stationary_ball_max_gap, config.kickoff_bridge_gap)
    runs = []
    start = previous = None
    count = 0
    for t in range(first, last + 2):
        position = ball[t] if t <= last else None
        if position is None and t <= last:
            continue
        on_spot = position is not None and \
            math.hypot(position[0] - spot[0], position[1] - spot[1]) <= \
            config.kickoff_spot_radius and (formation is None or formation[t])
        if on_spot and start is not None and \
                t - previous - 1 <= max_gap:
            previous, count = t, count + 1
            continue
        if start is not None and count >= config.kickoff_min_frames:
            runs.append((start, previous + 1))
        start = previous = None
        if on_spot:
            start, previous, count = t, t, 1
    return runs


def kickoff_dead(evidence, ball, blocks, config, pitch=GAMES[GAME]["pitch"], formation=None):
    """Evidence with D where the ball waits on the centre spot for a kick off; a no-op when off.

    Every frame of each kickoff_runs() span becomes D, missing frames inside it included.
    `formation` is kickoff_formation()'s per-frame result, or None to skip that check.
    """
    if config.kickoff_spot_radius < 0:
        return evidence
    out = list(evidence)
    for _, first, last in blocks:
        for start, end in kickoff_runs(ball, first, last, config, pitch, formation):
            out[start:end] = [DEAD] * (end - start)
    return out


def heading_out(before, after, frames, pitch, fps, config):
    """True when a ball moving from `before` to `after` in `frames` frames leaves at a line.

    `after` must be inside the pitch and within `lost_at_line_distance` of a line, with its
    velocity towards that line at least `lost_at_line_min_speed`. Any of the four lines counts,
    so a ball heading over the goal line near a corner is not judged by the nearer touch line.
    """
    length, width = pitch
    x, y = after
    if not (0 <= x <= length and 0 <= y <= width):
        return False
    vx = (x - before[0]) * fps / frames
    vy = (y - before[1]) * fps / frames
    for distance, outward in ((y, -vy), (width - y, vy), (x, -vx), (length - x, vx)):
        if distance <= config.lost_at_line_distance and outward >= config.lost_at_line_min_speed:
            return True
    return False


def lost_ball_dead(evidence, ball, blocks, config, pitch=GAMES[GAME]["pitch"],
                   fps=GAMES[GAME]["fps"]):
    """Evidence with D added where the ball is lost while leaving at a line; a no-op when disabled.

    A missing ball alone is never dead. Inside each half, a run of at least `lost_at_line_frames`
    missing-ball frames becomes D only when the last two observations before it, inside the same
    half and at most LOST_AT_LINE_SPEED_GAP frames apart, show the ball inside the pitch, near a
    line and moving out over it (heading_out). Runs are cut at the half's edges, and only frames
    without evidence change, so D and team evidence are left alone.
    """
    if config.lost_at_line_frames < 0:
        return evidence
    out = list(evidence)
    for _, first, last in blocks:
        before = after = None  # the last two observed frames in this half
        t = first
        while t <= last:
            if ball[t] is not None:
                before, after = after, t
                t += 1
                continue
            start = t
            while t <= last and ball[t] is None:
                t += 1
            if t - start >= config.lost_at_line_frames and before is not None and \
                    after - before <= LOST_AT_LINE_SPEED_GAP and \
                    heading_out(ball[before], ball[after], after - before, pitch, fps, config):
                out[start:t] = [DEAD if label is None else label for label in out[start:t]]
    return out


def interpolate_ball(ball, blocks, config, pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"]):
    """{frame: (x, y)} linearly interpolated over short missing-ball runs; empty when disabled.

    Inside each half, a run of at most `ball_interp_max_gap` missing frames is filled when both
    of its neighbouring observations are in that half and in bounds (no more than
    `out_of_bounds_margin` beyond a line; on the pitch when the margin is negative). Runs touching
    a half's edge are left alone. With `ball_interp_max_speed` >= 0, both edge speeds must also be
    known and at most that: ball_speed over `control_speed_span` frames of the half, back from
    the last observation before the run and forward from the first one after it.
    """
    positions = {}
    if config.ball_interp_max_gap < 0:
        return positions
    length, width = pitch
    margin = max(config.out_of_bounds_margin, 0.0)

    def in_bounds(point):
        return -margin <= point[0] <= length + margin and -margin <= point[1] <= width + margin

    for _, first, last in blocks:
        t = first
        while t <= last:
            if ball[t] is not None:
                t += 1
                continue
            start = t
            while t <= last and ball[t] is None:
                t += 1
            if start == first or t > last or t - start > config.ball_interp_max_gap:
                continue
            (x0, y0), (x1, y1) = ball[start - 1], ball[t]
            if not (in_bounds(ball[start - 1]) and in_bounds(ball[t])):
                continue
            if config.ball_interp_max_speed >= 0:
                reach = config.control_speed_span
                speeds = (ball_speed(ball, start - 1,
                                     range(max(first, start - 1 - reach), start - 1), fps),
                          ball_speed(ball, t, range(min(last, t + reach), t, -1), fps))
                if any(speed is None or speed > config.ball_interp_max_speed
                       for speed in speeds):
                    continue
            span = float(t - start + 1)
            for u in range(start, t):
                share = (u - start + 1) / span
                positions[u] = (x0 + (x1 - x0) * share, y0 + (y1 - y0) * share)
    return positions


def interpolated_evidence(evidence, raw_path, team_of_track, positions, pitch, radius,
                          mode="any"):
    """Evidence with nearest-player team calls on interpolated ball `positions` added.

    Only frames without evidence change, and only to H or A: the team of the nearest player
    observed on that frame, unless a non-negative `radius` (metres) leaves them all too far.
    Never D, and nothing when `positions` is empty. With `mode` "agree", a call is kept only when
    it equals `evidence` on the frame just before or just after its run of consecutive
    `positions` frames (one interpolated gap).
    """
    if not positions:
        return evidence
    length, width = pitch
    out = list(evidence)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame not in positions or out[frame] is not None:
                continue
            ball_x, ball_y = positions[frame]
            best, best_team = None, None
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                x, y = _xy(group)
                distance = (x * length - ball_x) ** 2 + (y * width - ball_y) ** 2
                if best is None or distance < best:
                    best, best_team = distance, team_of_track[slot]
            if best is None or (radius >= 0 and best ** 0.5 > radius):
                continue
            out[frame] = best_team
    if mode == "agree":
        for frame in positions:
            if out[frame] == evidence[frame]:
                continue  # not filled
            start = end = frame
            while start - 1 in positions:
                start -= 1
            while end + 1 in positions:
                end += 1
            sides = (evidence[start - 1] if start > 0 else None,
                     evidence[end + 1] if end + 1 < len(evidence) else None)
            if out[frame] not in sides:
                out[frame] = evidence[frame]
    return out


def vote(evidence, window):
    """Centred majority vote over team frames only; D and missing frames pass through unchanged.

    A tie keeps the frame's own observation rather than favouring either side.
    """
    if window <= 0:
        return list(evidence)
    home, away = [0], [0]
    for label in evidence:
        home.append(home[-1] + (label == HOME))
        away.append(away[-1] + (label == AWAY))
    out = list(evidence)
    n = len(evidence)
    for t, label in enumerate(evidence):
        if label not in TEAMS:
            continue
        low, high = max(0, t - window), min(n, t + window + 1)
        h, a = home[high] - home[low], away[high] - away[low]
        out[t] = HOME if h > a else AWAY if a > h else label
    return out


def viterbi(evidence, switch_cost, weights=None, team_costs=None):
    """Minimum-cost H/A/D path over one half's evidence. Returns a label for every frame.

    Emission: 0 when the state matches the evidence, 1 when it contradicts it, DEAD_EVIDENCE_COST
    for a team state on D evidence, 0 on missing frames. Transition: `switch_cost` for H<->A, half
    of it for a team<->D change. Ties keep the previous state. O(n) time, 3n bytes of backpointers.
    With `weights` (one per frame), contradicting team evidence costs weights[t] instead of 1.
    With `team_costs` (one per frame), an H<->A change into frame t costs team_costs[t] instead
    of `switch_cost`; team<->D changes keep half of `switch_cost`.
    """
    n = len(evidence)
    team, dead = float(switch_cost), switch_cost / 2.0
    back_h, back_a, back_d = bytearray(n), bytearray(n), bytearray(n)
    ch = ca = cd = 0.0
    for t, label in enumerate(evidence):
        if t:
            if team_costs is not None:
                team = team_costs[t]
            # Best predecessor of each state (0 = H, 1 = A, 2 = D); staying wins ties.
            nh, bh = ch, 0
            if ca + team < nh:
                nh, bh = ca + team, 1
            if cd + dead < nh:
                nh, bh = cd + dead, 2
            na, ba = ca, 1
            if ch + team < na:
                na, ba = ch + team, 0
            if cd + dead < na:
                na, ba = cd + dead, 2
            nd, bd = cd, 2
            if ch + dead < nd:
                nd, bd = ch + dead, 0
            if ca + dead < nd:
                nd, bd = ca + dead, 1
            ch, ca, cd = nh, na, nd
            back_h[t], back_a[t], back_d[t] = bh, ba, bd
        if label == HOME:
            cost = 1.0 if weights is None else weights[t]
            ca += cost
            cd += cost
        elif label == AWAY:
            cost = 1.0 if weights is None else weights[t]
            ch += cost
            cd += cost
        elif label == DEAD:
            ch += DEAD_EVIDENCE_COST
            ca += DEAD_EVIDENCE_COST
    state = 0
    if ca < ch:
        state = 1
    if cd < min(ch, ca):
        state = 2
    backs, states = (back_h, back_a, back_d), (HOME, AWAY, DEAD)
    path = [None] * n
    for t in range(n - 1, -1, -1):
        path[t] = states[state]
        state = backs[state][t]
    return path


def contested_team_costs(control, config):
    """Per-frame Viterbi H<->A costs: `switch_cost` times `contested_switch_factor` on frames
    where both teams have a player near the ball (len(control[t]) >= 2), `switch_cost` elsewhere.
    """
    low = config.switch_cost * config.contested_switch_factor
    return [low if len(teams) >= 2 else config.switch_cost for teams in control]


def segment(half, config, weights=None, control=None):
    """One half smoothed by the configured segmenter; missing frames stay None for fill_gaps.

    `weights` are the half's per-frame evidence weights for the Viterbi; None is hard evidence.
    `control` is the half's per-frame team control, used when `contested_switch_factor` is on.
    """
    if config.segmenter == "viterbi":
        team_costs = None
        if config.contested_switch_factor >= 0:
            team_costs = contested_team_costs(control, config)
        path = viterbi(half, config.switch_cost, weights, team_costs)
        return [None if label is None else state for label, state in zip(half, path)]
    return vote(half, config.vote_window)


def fill_gaps(labels, config):
    """Resolve runs of missing evidence inside one half. Returns H/A/D/X for every frame."""
    out = list(labels)
    n, t = len(out), 0
    while t < n:
        if out[t] is not None:
            t += 1
            continue
        start = t
        while t < n and out[t] is None:
            t += 1
        left = out[start - 1] if start > 0 else None
        right = out[t] if t < n else None
        length = t - start
        if left is None or right is None or length > config.max_gap:
            # half edge or long gap: no consistent evidence to bridge
            if config.long_gap_fill == "dead":
                fill = [DEAD] * length
            elif config.long_gap_fill == "hold" and (left or right) is not None:
                # the scorer's forward-fill, back-filled at the head of a half
                fill = [left or right] * length
            else:
                fill = [UNKNOWN] * length
        elif left == right:
            fill = [left] * length
        elif config.transition_fill == "midpoint":
            split = length // 2
            fill = [left] * split + [right] * (length - split)
        elif config.transition_fill == "hold":
            fill = [left] * length  # a loose ball stays with the last label until the next is seen
        else:
            fill = [UNKNOWN] * length
        out[start:t] = fill
    return out


def merge_short_dead(labels, min_run):
    """One filled half with short D runs inside a team's possession given to that team.

    A maximal D run shorter than `min_run` frames becomes H or A when the nearest label on each
    side that is not X is that same team. Runs touching the half's edge, or with D, nothing or
    different teams beyond them, are kept. A negative `min_run` returns `labels` unchanged.
    """
    if min_run < 0:
        return labels
    out = list(labels)
    n, t = len(out), 0
    while t < n:
        if out[t] != DEAD:
            t += 1
            continue
        start = t
        while t < n and out[t] == DEAD:
            t += 1
        if start == 0 or t == n or t - start >= min_run:
            continue
        before, after = start - 1, t
        while before >= 0 and out[before] == UNKNOWN:
            before -= 1
        while after < n and out[after] == UNKNOWN:
            after += 1
        left = out[before] if before >= 0 else None
        right = out[after] if after < n else None
        if left in TEAMS and left == right:
            out[start:t] = [left] * (t - start)
    return out


def bridge_dead_gaps(labels, max_bridge):
    """One filled half with short non-D runs between two D runs made D.

    A maximal run of H, A or X of at most `max_bridge` frames with D immediately on both sides
    becomes D, so a stray team call inside a stoppage does not split it into two D spells. Runs
    touching the half's edge are kept. A negative `max_bridge` returns `labels` unchanged.
    """
    if max_bridge < 0:
        return labels
    out = list(labels)
    n, t = len(out), 0
    while t < n:
        if out[t] == DEAD:
            t += 1
            continue
        start = t
        while t < n and out[t] != DEAD:
            t += 1
        if start == 0 or t == n or t - start > max_bridge:
            continue
        out[start:t] = [DEAD] * (t - start)
    return out


def refine_handovers(labels, control, window):
    """One decoded half with each H<->A change moved to the outgoing team's last control.

    For a change at frame c from team P to team Q, the latest frame f in [c - window,
    c + window - 1] where P is in `control[f]` (read_team_control) makes f + 1 the first Q frame.
    f stays inside the P run and the Q run around c, each keeping at least one frame, so a change
    never crosses a D or X run or another change. With no such f the change is kept. A negative
    `window` returns `labels` unchanged.
    """
    if window < 0:
        return labels
    out = list(labels)
    n, t = len(out), 1
    while t < n:
        old, new = out[t - 1], out[t]
        if old == new or old not in TEAMS or new not in TEAMS:
            t += 1
            continue
        start, end = t - 1, t + 1
        while start > 0 and out[start - 1] == old:
            start -= 1
        while end < n and out[end] == new:
            end += 1
        low, high = max(start, t - window), min(end - 2, t + window - 1)
        found = next((f for f in range(high, low - 1, -1) if old in control[f]), None)
        moved = t if found is None else found + 1
        if moved < t:
            out[moved:t] = [new] * (t - moved)
        elif moved > t:
            out[t:moved] = [old] * (moved - t)
        t = max(t, moved) + 1
    return out


def needs_control(config):
    """True when decoding uses the per-frame team control (read_team_control)."""
    return config.handover_search >= 0 or config.contested_switch_factor >= 0


def decode(evidence, blocks, config=DecoderConfig(), weights=None, control=None):
    """Labels for every frame. Each (half, first, last) block is decoded on its own; X elsewhere.

    `weights`, when given, are per-frame evidence weights aligned with `evidence`. `control`
    (read_team_control, aligned with `evidence`) is needed only when `handover_search` or
    `contested_switch_factor` is on.
    """
    if needs_control(config) and control is None:
        raise ValueError("handover_search and contested_switch_factor need the per-frame team "
                         "control")
    labels = [UNKNOWN] * len(evidence)
    for _, first, last in blocks:
        half = evidence[first:last + 1]
        half_weights = None if weights is None else weights[first:last + 1]
        half_control = None if control is None else control[first:last + 1]
        filled = fill_gaps(segment(half, config, half_weights, half_control), config)
        merged = merge_short_dead(
            bridge_dead_gaps(filled, config.max_dead_bridge), config.min_dead_run)
        if config.handover_search >= 0:
            merged = refine_handovers(merged, half_control, config.handover_search)
        labels[first:last + 1] = merged
    return labels


def write_submission(path, labels):
    """The frame,label CSV the challenge requires."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def run(out, config=DecoderConfig(), game=GAME, data_dir=None, tagging=None,
        config_source=None):
    """Decode and write `out`. Returns (out, aggregate Smart Tagging counts or None).

    `tagging` is a tagging_evidence.TaggingConfig; None keeps the decoder tracking-only.
    `config_source` (the --config-file path and record) is saved with the config when given.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    team_of_track = team_of_each_track(meta_path)
    evidence = read_evidence(raw_path, team_of_track, n_frames,
                             GAMES[game]["pitch"], config.out_of_bounds_margin,
                             config.control_radius)
    raw_evidence = evidence
    blocks = half_video_blocks(game)
    if config.control_motion_max_error > 0:
        from control_motion import MotionConfig, control_support, gate_evidence, tracking_arrays
        players, observed_ball = tracking_arrays(raw_path, team_of_track, n_frames, GAMES[game]['pitch'])
        motion = MotionConfig(radius_m=config.control_radius,
            max_relative_speed_mps=config.control_motion_max_error,
            lag_frames=config.control_motion_lag, minimum_run_frames=config.control_motion_min_run)
        evidence = gate_evidence(evidence, control_support(players, observed_ball, team_of_track,
            blocks, GAMES[game]['fps'], motion))
    if config.max_control_speed >= 0 or config.stationary_ball_frames >= 0 or \
            config.lost_at_line_frames >= 0 or config.ball_interp_max_gap >= 0 or \
            config.kickoff_spot_radius >= 0:
        import tagging_evidence
        ball = tagging_evidence.read_ball(raw_path, n_frames, GAMES[game]["pitch"])
        fast = fast_ball_frames(ball, blocks, config, GAMES[game]["fps"])
        if not soft_fast_ball(config):
            evidence = gate_fast_ball(evidence, fast)
        evidence = stationary_dead(evidence, ball, blocks, config)
        if config.kickoff_spot_radius >= 0:
            evidence = kickoff_dead(
                evidence, ball, blocks, config, GAMES[game]["pitch"],
                kickoff_formation(raw_path, team_of_track, n_frames, GAMES[game]["pitch"]))
        evidence = lost_ball_dead(evidence, ball, blocks, config, GAMES[game]["pitch"],
                                  GAMES[game]["fps"])
        evidence = interpolated_evidence(
            evidence, raw_path, team_of_track,
            interpolate_ball(ball, blocks, config, GAMES[game]["pitch"], GAMES[game]["fps"]),
            GAMES[game]["pitch"], config.control_radius, config.ball_interp_mode)
    stats = None
    saved = dict(asdict(config), game=game)
    if tagging is not None:
        import tagging_evidence
        evidence, stats = tagging_evidence.apply_bundle(
            evidence, meta_path, raw_path, blocks, GAMES[game]["pitch"], GAMES[game]["fps"],
            tagging, game, data_dir)
        saved[SMART_TAGGING] = asdict(tagging)
    if config_source is not None:
        saved["config_source"] = config_source
    weights = None
    if config.evidence_weighting == "margin":
        weights = read_evidence_weights(raw_path, team_of_track, n_frames, GAMES[game]["pitch"],
                                        config.margin_scale, config.out_of_bounds_margin)
    if soft_fast_ball(config):
        weights = fast_ball_weights(weights, raw_evidence, fast, config.fast_ball_weight)
    if weights is not None:
        weights = align_weights(weights, raw_evidence, evidence)
    control = None
    if needs_control(config):
        control = read_team_control(raw_path, team_of_track, n_frames, GAMES[game]["pitch"],
                                    config.out_of_bounds_margin, config.control_radius)
    labels = decode(evidence, blocks, config, weights, control)
    write_submission(out, labels)
    with open(out + ".config.json", "w") as handle:
        json.dump(saved, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return out, stats


def main(argv=None):
    defaults = DecoderConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "sequence",
                                                      "sequence_decoder.csv"))
    parser.add_argument("--data-dir", help="data bundle root (default: MTAH_DATA_DIR or ./data)")
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--vote-window", type=int, default=defaults.vote_window)
    parser.add_argument("--max-gap", type=int, default=defaults.max_gap)
    parser.add_argument("--transition-fill", choices=TRANSITION_FILLS,
                        default=defaults.transition_fill,
                        help="short gap between different labels: split at its midpoint, "
                             "abstain (X), or hold the last label until the next is observed")
    parser.add_argument("--long-gap-fill", choices=LONG_GAP_FILLS,
                        default=defaults.long_gap_fill,
                        help="long gap or half edge: abstain (X), hold the last label (the next "
                             "one at the head of a half), or dead (D)")
    parser.add_argument("--out-of-bounds-margin", type=float,
                        default=defaults.out_of_bounds_margin,
                        help="metres; negative disables out-of-bounds dead-ball evidence")
    parser.add_argument("--control-radius", type=float, default=defaults.control_radius,
                        help="metres; a ball farther than this from every player gives no team "
                             "evidence; negative disables")
    parser.add_argument("--stationary-ball-frames", type=int,
                        default=defaults.stationary_ball_frames,
                        help="frames a ball must stay put inside a half to count as dead; "
                             "negative (default) disables")
    parser.add_argument("--stationary-ball-radius", type=float,
                        default=defaults.stationary_ball_radius,
                        help="metres the ball may stray from where it stopped")
    parser.add_argument("--stationary-ball-lead-in", type=int,
                        default=defaults.stationary_ball_lead_in,
                        help="first frames of a stationary run not marked dead")
    parser.add_argument("--stationary-ball-max-gap", type=int,
                        default=defaults.stationary_ball_max_gap,
                        help="missing-ball frames tolerated inside a stationary run")
    parser.add_argument("--stationary-ball-anchor", choices=STATIONARY_ANCHORS,
                        default=defaults.stationary_ball_anchor,
                        help="measure the stationary radius from the run's first observation or "
                             "from the running centroid of its observations")
    parser.add_argument("--lost-at-line-frames", type=int, default=defaults.lost_at_line_frames,
                        help="missing-ball frames after a ball last seen leaving at a line that "
                             "count as dead; negative (default) disables")
    parser.add_argument("--lost-at-line-distance", type=float,
                        default=defaults.lost_at_line_distance,
                        help="metres from a line of the last observed ball")
    parser.add_argument("--lost-at-line-min-speed", type=float,
                        default=defaults.lost_at_line_min_speed,
                        help="m/s of the last observed velocity towards that line")
    parser.add_argument("--segmenter", choices=SEGMENTERS, default=defaults.segmenter,
                        help="within-half smoothing: majority vote or H/A/D Viterbi")
    parser.add_argument("--switch-cost", type=float, default=defaults.switch_cost,
                        help="viterbi: cost of a possession change, in frames of contradiction")
    parser.add_argument("--evidence-weighting", choices=EVIDENCE_WEIGHTINGS,
                        default=defaults.evidence_weighting,
                        help="viterbi: 'margin' weighs each frame by how much nearer the ball is "
                             "to its team than to the other team")
    parser.add_argument("--margin-scale", type=float, default=defaults.margin_scale,
                        help="metres of own-vs-opponent margin that give full evidence weight")
    parser.add_argument("--max-control-speed", type=float, default=defaults.max_control_speed,
                        help="m/s; a faster ball gives no team evidence; negative (default) "
                             "disables")
    parser.add_argument("--control-speed-span", type=int, default=defaults.control_speed_span,
                        help="frames back to the observation the ball speed is measured from")
    parser.add_argument("--fast-ball-require-observed", action="store_true",
                        default=defaults.fast_ball_require_observed,
                        help="with --max-control-speed: a ball is fast only when it was observed "
                             "on every frame of the speed span")
    parser.add_argument("--ball-interp-max-gap", type=int, default=defaults.ball_interp_max_gap,
                        help="missing-ball runs of at most this many frames inside a half are "
                             "interpolated for nearest-player team evidence; negative (default) "
                             "disables")
    parser.add_argument("--ball-interp-mode", choices=BALL_INTERP_MODES,
                        default=defaults.ball_interp_mode,
                        help="interpolated team calls: keep any, or only those that agree with "
                             "the team observed just before or after the gap")
    parser.add_argument("--ball-interp-max-speed", type=float,
                        default=defaults.ball_interp_max_speed,
                        help="m/s; interpolate a gap only when the ball is no faster at both of "
                             "its edges (over --control-speed-span frames); negative (default) "
                             "disables")
    parser.add_argument("--min-dead-run", type=int, default=defaults.min_dead_run,
                        help="shorter D runs between two labels of the same team join that "
                             "team's possession; negative (default) disables")
    parser.add_argument("--max-dead-bridge", type=int, default=defaults.max_dead_bridge,
                        help="non-D runs of at most this many frames between two D runs become "
                             "D; negative (default) disables")
    parser.add_argument("--kickoff-spot-radius", type=float, default=defaults.kickoff_spot_radius,
                        help="metres from the centre spot; a ball waiting there in a kick-off "
                             "formation counts as dead; negative (default) disables")
    parser.add_argument("--kickoff-min-frames", type=int, default=defaults.kickoff_min_frames,
                        help="observed frames the ball must wait on the centre spot")
    parser.add_argument("--kickoff-bridge-gap", type=int, default=defaults.kickoff_bridge_gap,
                        help="missing frames bridged inside a kick-off run when the observed "
                             "ball on both sides is on the spot; negative (default) keeps "
                             "--stationary-ball-max-gap")
    parser.add_argument("--fast-ball-weight", type=float, default=defaults.fast_ball_weight,
                        help="viterbi, with --max-control-speed: keep fast-ball team evidence at "
                             "this factor of its weight instead of removing it; negative "
                             "(default) keeps the hard gate")
    parser.add_argument("--stationary-backfill-frames", type=int,
                        default=defaults.stationary_backfill_frames,
                        help="stationary runs of at least this many frames are dead from their "
                             "first frame, without the lead-in; negative (default) disables")
    parser.add_argument("--handover-search", type=int, default=defaults.handover_search,
                        help="frames either side of each H<->A change searched for the outgoing "
                             "team's last control of the ball; negative (default) disables")
    parser.add_argument("--contested-switch-factor", type=float,
                        default=defaults.contested_switch_factor,
                        help="viterbi: an H<->A change into a frame where both teams have a "
                             "player within --control-radius of the ball costs this factor (<= 1) "
                             "of --switch-cost; negative (default) disables")
    parser.add_argument("--smart-tagging", action="store_true",
                        help="add restart evidence from Smart Tagging (see tagging_evidence.py)")
    import tagging_evidence
    tag_fields = fields(tagging_evidence.TaggingConfig)
    for field in tag_fields:
        parser.add_argument("--tag-" + field.name.replace("_", "-"), type=type(field.default),
                            default=field.default, help="with --smart-tagging")
    parser.add_argument("--config-file",
                        help="JSON decoder config, or a decoder_experiments JSONL log whose "
                             "chosen_config is used; explicit flags override it")
    parser.add_argument("--record", type=int,
                        help="with a JSONL --config-file: record index from 0, negative from "
                             "the end (default: the last)")
    args = parser.parse_args(argv)
    config_source = None
    if args.record is not None and not args.config_file:
        parser.error("--record needs --config-file")
    if args.config_file:
        from path_guard import check_paths
        try:
            check_paths([args.config_file])
            file_config, file_tagging, config_source = load_config_file(args.config_file,
                                                                        args.record)
        except (OSError, ValueError) as error:
            parser.error("--config-file: {}".format(error))
        values = asdict(file_config)
        if file_tagging is not None:
            values["smart_tagging"] = True
            values.update(("tag_" + name, value) for name, value in asdict(file_tagging).items())
        for action in parser._actions:  # JSON 40 in a float field becomes 40.0, as the flag would
            if action.dest in values and action.type is float:
                values[action.dest] = float(values[action.dest])
        parser.set_defaults(**values)
        args = parser.parse_args(argv)  # flags given explicitly still override the file
    config = DecoderConfig(args.vote_window, args.max_gap, args.transition_fill,
                           args.out_of_bounds_margin, args.control_radius,
                           args.stationary_ball_frames, args.stationary_ball_radius,
                           args.stationary_ball_lead_in, args.stationary_ball_max_gap,
                           args.lost_at_line_frames, args.lost_at_line_distance,
                           args.lost_at_line_min_speed, args.segmenter, args.switch_cost,
                           args.evidence_weighting, args.margin_scale, args.long_gap_fill,
                           args.max_control_speed, args.control_speed_span,
                           args.stationary_ball_anchor, args.ball_interp_max_gap,
                           args.min_dead_run, args.max_dead_bridge, args.kickoff_spot_radius,
                           args.kickoff_min_frames, args.fast_ball_weight,
                           args.stationary_backfill_frames, args.handover_search,
                           args.ball_interp_mode, args.ball_interp_max_speed,
                           args.fast_ball_require_observed, args.kickoff_bridge_gap,
                           args.contested_switch_factor)
    tagging = None
    if args.smart_tagging:
        tagging = tagging_evidence.TaggingConfig(
            **{field.name: getattr(args, "tag_" + field.name) for field in tag_fields})
    print("config: {}".format(json.dumps(asdict(config), sort_keys=True)))
    if tagging is not None:
        print("smart tagging config: {}".format(json.dumps(asdict(tagging), sort_keys=True)))
    out, stats = run(args.out, config, args.game, args.data_dir, tagging, config_source)
    if stats is not None:
        print("smart tagging: {restart_events} restart events, {used} used, {ignored} ignored, "
              "{contradicted} contradicted; team prior overridden {team_prior_overridden}, "
              "team from tracking {team_from_tracking}; D frames added {d_frames_added}; "
              "located by {located}".format(located=json.dumps(stats["located_by"],
                                                               sort_keys=True), **stats))
    print("wrote {}".format(out))


if __name__ == "__main__":
    main()
