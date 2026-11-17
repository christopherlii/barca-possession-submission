"""Synthetic decoder experiments: pick a sequence-decoder config on development seeds, then score it
once on held-out seeds beside the official baseline.

Every number this prints comes from generated matches (scripts/synthetic_match.py). It is a
synthetic benchmark for comparing methods, NOT an estimate of performance on the supplied match.

  1. Generate one bundle per seed with the same generator config.
  2. Score every decoder config in the grid on the development seeds with the official scorer
     (challenge/evaluation/possession_metrics.evaluate). Choose the config with the highest mean
     spell_f1_mean; ties go to the higher mean macro_f1, then to the earlier config in canonical
     order (sorted JSON of its fields). Held-out seeds play no part in the choice.
  3. Score only the chosen config and baseline_smooth50 (the official baseline's nearest-player
     labels with its +/-50 frame vote) on the held-out seeds, with aggregate diagnostics against
     the generator's latent causes (per-cause accuracy and confusion, ball-missing split, spell
     boundary matching). Diagnostics are reported only; they never feed the selection. The
     console shows development diagnostics; held-out ones are logged for audit only.
  4. Append one JSONL record -- git revision, generator config, grid, seeds, chosen config,
     aggregate metrics and diagnostics -- to <out-dir>/decoder_experiments.jsonl.

    python scripts/decoder_experiments.py --dev-seeds 1 2 3 4 --heldout-seeds 101 102 103
    python scripts/decoder_experiments.py --grid grid.json --bootstrap 1000 ...

`--grid` is a JSON list of DecoderConfig field dicts (missing fields take their defaults). An
entry may add "smart_tagging": a dict of TaggingConfig overrides, or true for its defaults; that
entry then adds the bundle's synthetic restart tags (tagging_evidence.apply_restarts) before
decoding. Prints aggregate numbers only. Needs numpy and pandas, like the scorer.

`--scenarios FILE` is a JSON object mapping a scenario name to SyntheticConfig overrides (applied
on top of --generator-config, under --n-frames); top-level keys starting with "_" are comments.
Each scenario gets its own dev and held-out bundles from the same seed lists, under
<work-dir>/<scenario>/. Selection then scores each config by its per-scenario mean dev
spell_f1_mean, aggregated across scenarios by `--select mean` (default) or `--select worst`
(the minimum); macro_f1 is aggregated the same way for ties. The chosen config and
baseline_smooth50 are scored on every scenario's held-out seeds, plus a cross-scenario mean.
Without --scenarios the run, its choice and its log record are unchanged.

    python scripts/decoder_experiments.py --dev-seeds 1 2 3 --heldout-seeds 101 102 \\
        --scenarios scripts/scenarios_default.json --select worst

`--search-steps N` (default 0 = off, run and record unchanged) continues from the grid winner
with a deterministic coordinate search on the same development scores and selection rule. Each
step tries every one-field move in SEARCH_NUMERIC / SEARCH_CATEGORICAL: numeric fields times and
divided by SEARCH_FACTOR (ints rounded, moved at least 1), a disabled negative sentinel only
enabled to its table default, categorical fields switched to each other allowed value; moves
DecoderConfig rejects and fields inactive under the config are skipped, and the tagging config is
kept. The best neighbour (same tie-break as the grid) is accepted only if it strictly beats the
current (spell_f1_mean, macro_f1); the search stops after N moves or when none improves. Scores
are cached by canonical key. Held-out bundles are generated only after the search, and only the
final config and baseline_smooth50 are scored on them. The record adds "search" (steps_requested,
steps_taken, factor, start_config, trail of {config, spell_f1_mean, macro_f1}) and its
chosen_config is the final config, so `sequence_decoder.py --config-file` picks it up.

    python scripts/decoder_experiments.py --dev-seeds 1 2 3 --heldout-seeds 101 102 \\
        --search-steps 5

`--reference PATH` (optional `--reference-record N`) loads a reference config the way
`sequence_decoder.py --config-file` does: a JSON config, or a decoder_experiments JSONL log and
the "chosen_config" of record N (default the last). After selection and any search it is scored on
the same held-out bundles as the chosen config; it never takes part in selection or search. The
record adds "reference": source, config, held-out summary (plus per scenario), and "paired":
per-seed chosen minus reference spell_f1_mean and macro_f1 deltas with their mean,
wins/ties/losses and a seed-level percentile bootstrap interval of the mean delta (fixed RNG
seed). With --scenarios each seed's delta is first averaged over scenarios, as
cross_scenario_mean does, and a per-scenario breakdown is kept. Without it the run is unchanged.

    python scripts/decoder_experiments.py --dev-seeds 1 2 3 --heldout-seeds 101 102 \\
        --reference outputs/experiments/decoder_experiments.jsonl --reference-record -1

`--seed-ledger PATH` (default experiments/seed_ledger.json) is checked before any bundle is
generated: held-out seeds already in "inspected_heldout" are refused, and so is any dev or
held-out seed in the reserved "final_audit" range unless --final-audit is given. A --final-audit
run must take its held-out seeds from that range and is refused once "final_audit_used" is set.
After the record is logged the held-out seeds are added to "inspected_heldout" (or
"final_audit_used" is set) and the dev seeds to "used_dev", by an atomic rewrite. The record's
"seed_ledger" gives the path; `--no-seed-ledger` (unit tests, ad hoc runs) skips it and the
record says "bypassed". run_experiment bypasses the ledger unless given `seed_ledger`.
--diagnose-only reads the ledger only to refuse final-audit dev seeds and never writes it.

After selection, the chosen config (and the reference, when given) is also diagnosed on every
development bundle, pooled over seeds per scenario like the held-out diagnostics; the record adds
"dev_diagnostics" ({"chosen", "reference"} -> diagnostics, or scenario -> that with --scenarios,
plus "dev_diagnostics_pooled" over all scenarios). Development diagnostics also count the raw
nearest-player evidence, ball observations and two-team contests behind each unmatched GT spell
of the no_overlap and merged classes (spell_failures "gt_support", spell_support). The console
prints these development tables; held-out per-cause diagnostics stay in the log for audit only.

`--diagnose-only CONFIG [CONFIG ...]` skips search and held-out generation: it decodes each
config (a JSON config, or the last record of a decoder_experiments log) on the development seeds
and appends their diagnostics and dev metrics to <out-dir>/decoder_diagnostics.jsonl, so a cycle
can study failure modes before forming a hypothesis without touching held-out data.

    python scripts/decoder_experiments.py --dev-seeds 81 82 83 84 --n-frames 6000 \\
        --scenarios scripts/scenarios_default.json --diagnose-only experiments/incumbent.json
"""

import argparse
from dataclasses import asdict, dataclass, replace
import datetime
import json
import os
import random
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "challenge", "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "challenge", "starter"))
from path_guard import check_paths, guard_paths  # noqa: E402
from sequence_decoder import (BALL_INTERP_MODES, EVIDENCE_WEIGHTINGS,  # noqa: E402
                              LONG_GAP_FILLS, SEGMENTERS, SMART_TAGGING, STATIONARY_ANCHORS,
                              TRANSITION_FILLS, DecoderConfig, align_weights, decode, fast_ball_frames, fast_ball_weights,
                              gate_fast_ball, interpolate_ball, interpolated_evidence,
                              kickoff_dead, kickoff_formation, load_config_file, lost_ball_dead,
                              needs_control, parse_config, read_evidence, read_evidence_weights,
                              read_team_control, soft_fast_ball, stationary_dead,
                              team_of_each_track)
import synthetic_match  # noqa: E402
import tagging_evidence  # noqa: E402
from tagging_evidence import Restart  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(ROOT, "outputs", "experiments")
LOG_NAME = "decoder_experiments.jsonl"
DEFAULT_SEED_LEDGER = os.path.join(ROOT, "experiments", "seed_ledger.json")
LEDGER_LISTS = ("inspected_heldout", "used_dev", "final_audit")
DIAGNOSE_LOG_NAME = "decoder_diagnostics.jsonl"
HELDOUT_AUDIT_NOTE = ("held-out per-cause diagnostics are in the log only, for audit; do not "
                      "tune against them")
BASELINE_NAME = "baseline_smooth50"
BASELINE_WINDOW = 50
REPORTED = ("spell_f1_mean", "macro_f1", "accuracy", "spell_count_error", "fragmentation")
BANNER = "SYNTHETIC benchmark -- generated matches, not an estimate of real-match performance"
CODE_LABELS = "HADX"  # possession_metrics codes 0..3
UNKNOWN = "X"
SELECTIONS = ("mean", "worst")
SELECTION_TEXT = "max mean dev spell_f1_mean, then macro_f1, then canonical grid order"
SCENARIO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PAIRED = ("spell_f1_mean", "macro_f1", "accuracy")
PAIRED_DRAWS = 2000
PAIRED_RNG_SEED = 0
PAIRED_LEVEL = 0.95
TIE_TOLERANCE = 1e-12
# Spell failure classes for unmatched spells at the headline IoU, in test order (spell_failures).
SPELL_FAILURES = ("no_overlap", "merged", "split", "misaligned")
SPELL_SPLIT_COVER = 0.2  # share of a spell each of 2+ same-team overlaps must cover for "split"
# Evidence support of unmatched GT spells (spell_support): the classes counted, the per-spell
# counts over graded frames, and the own-team share buckets (0, (0, 0.25], (0.25, 0.5], > 0.5).
SUPPORT_CLASSES = ("no_overlap", "merged")
SUPPORT_COUNTS = ("spells", "frames", "ball_observed", "own", "other", "both_near")
SUPPORT_BUCKETS = ("0", "<=0.25", "<=0.5", ">0.5")
TEAM_LABELS = ("H", "A")

# A small default grid around the decoder's a-priori defaults, plus stationary-ball dead-ball
# evidence after 2 s and 4 s of a still ball (a-priori restart set-up times), plus Viterbi
# switch costs, with hard and distance-margin evidence weights (1-4 m: a-priori stride scales),
# plus synthetic restart tags (Smart Tagging defaults and a shorter stoppage cap), plus a ball
# lost while leaving at a line as dead after 1 s and 2 s missing, alone and with the Viterbi,
# plus "hold" gap fill (a loose ball stays with the last team) for the vote and the Viterbi,
# plus long-gap / half-edge fill "hold" (the scorer's own forward-fill) for the vote and the
# Viterbi, and "dead", plus no team evidence from a ball faster than a firm pass (12 m/s, and
# 8 m/s for the vote), plus a still ball measured from its running centroid that tolerates a 1 s
# dropout, for the vote and the Viterbi.
DEFAULT_GRID = [
    {"vote_window": vote_window, "max_gap": max_gap, "transition_fill": "midpoint"}
    for vote_window in (12, 25, 50) for max_gap in (25, 50, 100)
] + [{"transition_fill": "unknown"}, {"out_of_bounds_margin": -1.0},
     {"stationary_ball_frames": 50}, {"stationary_ball_frames": 100}] + [
    {"segmenter": "viterbi", "switch_cost": switch_cost}
    for switch_cost in (10.0, 25.0, 50.0, 100.0)
] + [
    {"segmenter": "viterbi", "evidence_weighting": "margin", "switch_cost": switch_cost,
     "margin_scale": margin_scale}
    for switch_cost in (10.0, 25.0, 50.0) for margin_scale in (1.0, 2.0, 4.0)
] + [
    {"max_gap": max_gap, SMART_TAGGING: True} for max_gap in (50, 100)
] + [{SMART_TAGGING: {"max_stoppage": 750}}] + [
    dict(segmenter, lost_at_line_frames=frames)
    for segmenter in ({}, {"segmenter": "viterbi"}) for frames in (25, 50)
] + [
    {"transition_fill": "hold"},
    {"segmenter": "viterbi", "switch_cost": 25.0, "transition_fill": "hold"},
    {"segmenter": "viterbi", "evidence_weighting": "margin", "switch_cost": 25.0,
     "margin_scale": 2.0, "transition_fill": "hold"},
    {"long_gap_fill": "hold"},
    {"segmenter": "viterbi", "switch_cost": 25.0, "long_gap_fill": "hold"},
    {"long_gap_fill": "dead"},
    {"max_control_speed": 12.0}, {"max_control_speed": 8.0},
    {"segmenter": "viterbi", "switch_cost": 25.0, "max_control_speed": 12.0},
    {"segmenter": "viterbi", "evidence_weighting": "margin", "switch_cost": 25.0,
     "margin_scale": 2.0, "max_control_speed": 12.0},
] + [
    dict(segmenter, stationary_ball_frames=50, stationary_ball_anchor="centroid",
         stationary_ball_max_gap=25)
    for segmenter in ({}, {"segmenter": "viterbi", "switch_cost": 25.0})
]


SEARCH_FACTOR = 1.5
# Coordinate search neighbourhood: (field, value that enables a disabled negative sentinel or
# None, whether the field matters under a config). Enable values are the decoder defaults or the
# default grid's a-priori settings.
SEARCH_NUMERIC = (
    ("vote_window", None, lambda c: c.segmenter == "vote"),
    ("max_gap", None, lambda c: True),
    ("out_of_bounds_margin", 1.0, lambda c: True),
    ("control_radius", 3.0, lambda c: True),
    ("stationary_ball_frames", 50, lambda c: True),
    ("stationary_ball_radius", None, lambda c: c.stationary_ball_frames > 0),
    ("stationary_ball_lead_in", None, lambda c: c.stationary_ball_frames > 0),
    ("stationary_ball_max_gap", None, lambda c: c.stationary_ball_frames > 0
     or c.kickoff_spot_radius >= 0),
    ("lost_at_line_frames", 25, lambda c: True),
    ("lost_at_line_distance", None, lambda c: c.lost_at_line_frames > 0),
    ("lost_at_line_min_speed", None, lambda c: c.lost_at_line_frames > 0),
    ("switch_cost", None, lambda c: c.segmenter == "viterbi"),
    ("margin_scale", None, lambda c: c.evidence_weighting == "margin"),
    ("max_control_speed", 12.0, lambda c: True),
    ("control_speed_span", None, lambda c: c.max_control_speed > 0),
    ("ball_interp_max_gap", 10, lambda c: True),
    # 4 m/s, an a-priori upper bound for a ball carried at a player's feet.
    ("ball_interp_max_speed", 4.0, lambda c: c.ball_interp_max_gap > 0),
    # 2 s, an a-priori shortest real stoppage; only where some rule can put D on a frame.
    ("min_dead_run", 50, lambda c: c.out_of_bounds_margin >= 0 or c.stationary_ball_frames > 0
     or c.lost_at_line_frames > 0 or c.long_gap_fill == "dead" or c.kickoff_spot_radius >= 0),
    # 2 s, the same a-priori stoppage scale; only where some rule can put D on a frame.
    ("max_dead_bridge", 50, lambda c: c.out_of_bounds_margin >= 0 or c.stationary_ball_frames > 0
     or c.lost_at_line_frames > 0 or c.long_gap_fill == "dead" or c.kickoff_spot_radius >= 0),
    ("kickoff_spot_radius", 1.5, lambda c: True),
    ("kickoff_min_frames", None, lambda c: c.kickoff_spot_radius >= 0),
    # Half the switch cost on contested frames, the cycle-023 hypothesis.
    ("contested_switch_factor", 0.5, lambda c: c.segmenter == "viterbi"),
)
SEARCH_CATEGORICAL = (
    ("transition_fill", TRANSITION_FILLS, lambda c: True),
    ("long_gap_fill", LONG_GAP_FILLS, lambda c: True),
    ("segmenter", SEGMENTERS, lambda c: True),
    ("evidence_weighting", EVIDENCE_WEIGHTINGS, lambda c: True),
    ("stationary_ball_anchor", STATIONARY_ANCHORS, lambda c: c.stationary_ball_frames > 0),
    ("ball_interp_mode", BALL_INTERP_MODES, lambda c: c.ball_interp_max_gap > 0),
)


@dataclass(frozen=True)
class GridConfig:
    """One grid entry: a decoder config, plus restart tagging settings or None (tracking only)."""
    decoder: DecoderConfig
    tagging: object = None


def check_seed_lists(dev_seeds, heldout_seeds):
    """Reject empty, repeated or overlapping development / held-out seed lists."""
    if not dev_seeds or not heldout_seeds:
        raise ValueError("need at least one development and one held-out seed")
    for name, seeds in (("development", dev_seeds), ("held-out", heldout_seeds)):
        if len(set(seeds)) != len(seeds):
            raise ValueError("{} seeds repeat: {}".format(name, list(seeds)))
    overlap = sorted(set(dev_seeds) & set(heldout_seeds))
    if overlap:
        raise ValueError("development and held-out seeds overlap: {}".format(overlap))


def load_seed_ledger(path):
    """The seed ledger at `path`: its JSON object with the seed lists checked to be int lists."""
    check_paths([path])
    with open(path) as handle:
        ledger = json.load(handle)
    if not isinstance(ledger, dict):
        raise ValueError("seed ledger {} is not a JSON object".format(path))
    for key in LEDGER_LISTS:
        seeds = ledger.setdefault(key, [])
        if not isinstance(seeds, list) or not all(
                isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
            raise ValueError("seed ledger {}: {} must be a list of integer seeds".format(path, key))
    if not isinstance(ledger.setdefault("final_audit_used", False), bool):
        raise ValueError("seed ledger {}: final_audit_used must be true or false".format(path))
    return ledger


def check_seed_ledger(ledger, dev_seeds, heldout_seeds, final_audit=False):
    """Reject held-out seeds already inspected and any use of the reserved final-audit seeds,
    which only a --final-audit run may score as held-out, once, with no audit seed in dev."""
    inspected = sorted(set(heldout_seeds) & set(ledger["inspected_heldout"]))
    if inspected:
        raise ValueError("held-out seeds already inspected (seed ledger): {}; pick fresh "
                         "held-out seeds".format(inspected))
    audit = set(ledger["final_audit"])
    if final_audit:
        if ledger["final_audit_used"]:
            raise ValueError("the final audit seeds are already marked used (seed ledger)")
        in_dev = sorted(set(dev_seeds) & audit)
        if in_dev:
            raise ValueError("development seeds in the final audit range: {}".format(in_dev))
        outside = sorted(set(heldout_seeds) - audit)
        if outside:
            raise ValueError("--final-audit held-out seeds must come from the final audit range; "
                             "not in it: {}".format(outside))
        return
    reserved = sorted((set(dev_seeds) | set(heldout_seeds)) & audit)
    if reserved:
        raise ValueError("seeds reserved for the untouched final audit: {}; pass --final-audit "
                         "only for the locked final evaluation".format(reserved))


def update_seed_ledger(path, dev_seeds, heldout_seeds, final_audit=False):
    """Re-read the ledger, then atomically mark the held-out seeds inspected (or the final audit
    used) and add the dev seeds to used_dev."""
    ledger = load_seed_ledger(path)
    if final_audit:
        ledger["final_audit_used"] = True
    else:
        ledger["inspected_heldout"] = sorted(set(ledger["inspected_heldout"]) |
                                             set(heldout_seeds))
    ledger["used_dev"] = sorted(set(ledger["used_dev"]) | set(dev_seeds))
    folder = os.path.dirname(os.path.abspath(path))
    handle, temp = tempfile.mkstemp(dir=folder, prefix=".seed_ledger.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.remove(temp)
        raise


def config_record(config):
    """Flat DecoderConfig fields, plus "smart_tagging" (all TaggingConfig fields) when used."""
    record = asdict(config.decoder)
    if config.tagging is not None:
        record[SMART_TAGGING] = asdict(config.tagging)
    return record


def canonical_key(config):
    return json.dumps(config_record(config), sort_keys=True)


def parse_grid(entries):
    """GridConfigs from a list of field dicts, de-duplicated, in canonical order."""
    configs = {}
    for entry in entries:
        config = GridConfig(*parse_config(entry))
        configs[canonical_key(config)] = config
    if not configs:
        raise ValueError("the config grid is empty")
    return [configs[key] for key in sorted(configs)]


def load_grid(path=None):
    if not path:
        return parse_grid(DEFAULT_GRID)
    check_paths([path])
    with open(path) as handle:
        return parse_grid(json.load(handle))


def load_reference(path, record=None):
    """(GridConfig, source) from a JSON config or a decoder_experiments log record
    (sequence_decoder.load_config_file)."""
    check_paths([path])
    decoder, tagging, source = load_config_file(path, record)
    return GridConfig(decoder, tagging), source


def parse_scenarios(entries, generator_config_path=None, **overrides):
    """{name: SyntheticConfig} from a JSON object of override dicts, in file order.

    Each entry goes through synthetic_match.load_config on top of generator_config_path, with
    `overrides` (e.g. n_frames) last; unknown keys are an error. Names become work subdirectory
    names, so they must be plain path components. Keys starting with "_" are comments.
    """
    if not isinstance(entries, dict):
        raise ValueError("scenarios must be a JSON object mapping names to override dicts")
    scenarios = {}
    for name, values in entries.items():
        if name.startswith("_"):
            continue
        if not SCENARIO_NAME.match(name):
            raise ValueError("scenario name {!r} must be letters, digits, '_', '.' or '-'"
                             .format(name))
        if not isinstance(values, dict):
            raise ValueError("scenario {!r} must be a dict of SyntheticConfig overrides"
                             .format(name))
        try:
            scenarios[name] = synthetic_match.load_config(generator_config_path,
                                                          **dict(values, **overrides))
        except ValueError as error:
            raise ValueError("scenario {!r}: {}".format(name, error))
    if not scenarios:
        raise ValueError("the scenario file defines no scenarios")
    return scenarios


def load_scenarios(path, generator_config_path=None, **overrides):
    check_paths([path, generator_config_path])
    with open(path) as handle:
        return parse_scenarios(json.load(handle), generator_config_path, **overrides)


class Bundle:
    """One generated match: ground truth codes plus cached decoder evidence."""

    def __init__(self, manifest):
        import possession_metrics as pm
        check_paths(manifest["paths"].values())
        self.seed = manifest["seed"]
        self.paths = manifest["paths"]
        self.n_frames = manifest["n_frames"]
        self.blocks = [tuple(block) for block in manifest["blocks"]]
        self.pitch = tuple(manifest["pitch"])
        self.fps = manifest["fps"]
        self.gt_codes = pm.label_codes(pm.read_label_csv(self.paths["gt"]))
        # Latent sidecar: diagnostics only, never decoder input or selection.
        with open(self.paths["latent"]) as handle:
            rows = [line.rstrip("\n").split(",") for line in handle.readlines()[1:]]
        self.causes = [row[1] for row in rows]
        self.ball_missing = [row[2] == "1" for row in rows]
        self._teams = team_of_each_track(self.paths["atd_meta"])
        self._evidence = {}
        self._weights = {}
        self._fast = {}
        self._control = {}
        self._ball = None
        self._formation = None
        self._restarts = None

    def evidence(self, margin, radius, config=None):
        """Cached evidence; `config` adds its fast-ball gate, its stationary-ball, kick-off and
        lost-at-line D evidence and its interpolated-ball team evidence when those are enabled,
        in sequence_decoder.run()'s order."""
        fast = () if config is None or config.max_control_speed < 0 or \
            soft_fast_ball(config) else (config.max_control_speed, config.control_speed_span,
                                         config.fast_ball_require_observed)
        stationary = () if config is None or config.stationary_ball_frames < 0 else (
            config.stationary_ball_frames, config.stationary_ball_radius,
            config.stationary_ball_lead_in, config.stationary_ball_max_gap,
            config.stationary_ball_anchor, config.stationary_backfill_frames)
        lost = () if config is None or config.lost_at_line_frames < 0 else (
            config.lost_at_line_frames, config.lost_at_line_distance,
            config.lost_at_line_min_speed)
        interp = () if config is None or config.ball_interp_max_gap < 0 else (
            config.ball_interp_max_gap, config.ball_interp_mode, config.ball_interp_max_speed,
            config.control_speed_span if config.ball_interp_max_speed >= 0 else None)
        kickoff = () if config is None or config.kickoff_spot_radius < 0 else (
            config.kickoff_spot_radius, config.kickoff_min_frames, config.stationary_ball_max_gap,
            config.kickoff_bridge_gap)
        motion = () if config is None or config.control_motion_max_error < 0 else (
            config.control_motion_max_error, config.control_motion_lag, config.control_motion_min_run)
        key = (margin, radius, stationary, lost, fast, interp, kickoff, motion)
        if key not in self._evidence:
            evidence = read_evidence(self.paths["atd_raw"], self._teams, self.n_frames,
                                     self.pitch, margin, radius)
            if motion:
                from control_motion import MotionConfig, control_support, gate_evidence, tracking_arrays
                players, ball = tracking_arrays(self.paths['atd_raw'], self._teams, self.n_frames, self.pitch)
                supported = control_support(players, ball, self._teams, self.blocks, self.fps,
                    MotionConfig(radius_m=radius, max_relative_speed_mps=motion[0],
                                 lag_frames=motion[1], minimum_run_frames=motion[2]))
                evidence = gate_evidence(evidence, supported)
            if fast:
                evidence = gate_fast_ball(evidence, self.fast_frames(config))
            if stationary:
                evidence = stationary_dead(evidence, self.ball(), self.blocks, config)
            if kickoff:
                evidence = kickoff_dead(evidence, self.ball(), self.blocks, config, self.pitch,
                                        self.formation())
            if lost:
                evidence = lost_ball_dead(evidence, self.ball(), self.blocks, config, self.pitch,
                                          self.fps)
            if interp:
                evidence = interpolated_evidence(
                    evidence, self.paths["atd_raw"], self._teams,
                    interpolate_ball(self.ball(), self.blocks, config, self.pitch, self.fps),
                    self.pitch,
                    radius, config.ball_interp_mode)
            self._evidence[key] = evidence
        return self._evidence[key]

    def weights(self, scale, margin):
        """Cached distance-margin evidence weights (sequence_decoder.read_evidence_weights)."""
        key = (scale, margin)
        if key not in self._weights:
            self._weights[key] = read_evidence_weights(self.paths["atd_raw"], self._teams,
                                                       self.n_frames, self.pitch, scale, margin)
        return self._weights[key]

    def control(self, margin, radius):
        """Cached per-frame teams near the ball (sequence_decoder.read_team_control)."""
        key = (margin, radius)
        if key not in self._control:
            self._control[key] = read_team_control(self.paths["atd_raw"], self._teams,
                                                   self.n_frames, self.pitch, margin, radius)
        return self._control[key]

    def support_inputs(self, config):
        """(raw nearest-player evidence before any stationary-ball, kick-off or other rewrite,
        team control, ball observed) per frame at the config's margin and radius, for
        spell_support. Diagnostics only."""
        margin, radius = config.out_of_bounds_margin, config.control_radius
        return (self.evidence(margin, radius), self.control(margin, radius),
                [position is not None for position in self.ball()])

    def fast_frames(self, config):
        """Cached sequence_decoder.fast_ball_frames for the config's speed settings."""
        key = (config.max_control_speed, config.control_speed_span,
               config.fast_ball_require_observed)
        if key not in self._fast:
            self._fast[key] = fast_ball_frames(self.ball(), self.blocks, config, self.fps)
        return self._fast[key]

    def formation(self):
        """Cached per-frame kick-off formation flags (sequence_decoder.kickoff_formation)."""
        if self._formation is None:
            self._formation = kickoff_formation(self.paths["atd_raw"], self._teams,
                                                self.n_frames, self.pitch)
        return self._formation

    def ball(self):
        if self._ball is None:
            self._ball = tagging_evidence.read_ball(self.paths["atd_raw"], self.n_frames,
                                                    self.pitch)
        return self._ball

    def restarts(self):
        """The bundle's synthetic restart tags as tagging_evidence.Restart, in window order."""
        if self._restarts is None:
            with open(self.paths["tagging"]) as handle:
                tags = json.load(handle)
            self._restarts = sorted((Restart(tag["start"], tag["end"], tag["kind"], tag["team"])
                                     for tag in tags), key=lambda r: (r.start, r.end))
        return self._restarts

    def decoder_labels(self, config, tagging=None):
        """Decoded labels; with a TaggingConfig, restart evidence is added before decoding."""
        evidence = self.evidence(config.out_of_bounds_margin, config.control_radius, config)
        if tagging is not None:
            evidence, _ = tagging_evidence.apply_restarts(evidence, self.ball(), self.blocks,
                                                          self.restarts(), tagging, self.fps)
        weights = None
        if config.evidence_weighting == "margin":
            weights = self.weights(config.margin_scale, config.out_of_bounds_margin)
        raw = self.evidence(config.out_of_bounds_margin, config.control_radius)
        if soft_fast_ball(config):
            weights = fast_ball_weights(weights, raw, self.fast_frames(config),
                                        config.fast_ball_weight)
        if weights is not None:
            weights = align_weights(weights, raw, evidence)
        control = None
        if needs_control(config):
            control = self.control(config.out_of_bounds_margin, config.control_radius)
        return decode(evidence, self.blocks, config, weights, control)

    def baseline_labels(self, window=BASELINE_WINDOW):
        import baseline
        codes = baseline.nearest_player_labels(
            self.paths["atd_raw"], baseline.team_of_each_track(self.paths["atd_meta"]),
            self.n_frames, self.pitch)
        return baseline.smooth(codes, window)


def generate_bundles(seeds, generator_config, work_dir):
    check_paths([work_dir])
    return [Bundle(synthetic_match.write_bundle(os.path.join(work_dir, "seed-{}".format(seed)),
                                                seed, generator_config))
            for seed in seeds]


def score(gt_codes, labels, bootstrap_n=0):
    """The official scorer, called directly."""
    import possession_metrics as pm
    return pm.evaluate(gt_codes, pm.label_codes(list(labels)), bootstrap_n=bootstrap_n)


def summarise(results):
    """Seed-mean of the reported metrics, plus mean absolute spell-count error."""
    n = float(len(results))
    summary = {name: sum(result[name] for result in results) / n for name in REPORTED}
    summary["abs_spell_count_error"] = sum(abs(r["spell_count_error"]) for r in results) / n
    return summary


def dev_summaries(grid, dev_bundles):
    """Seed-mean summary of every grid config, in grid order. Sees development bundles only."""
    return [summarise([score(bundle.gt_codes,
                             bundle.decoder_labels(config.decoder, config.tagging))
                       for bundle in dev_bundles])
            for config in grid]


def aggregate(values, rule):
    """Combine one config's per-scenario scores: their mean, or their minimum for "worst"."""
    if rule not in SELECTIONS:
        raise ValueError("selection must be one of {}".format(SELECTIONS))
    values = list(values)
    return min(values) if rule == "worst" else sum(values) / float(len(values))


def select_across(grid, summaries_by_scenario, rule="mean"):
    """(chosen config, per-config aggregated spell_f1_mean and macro_f1 rows).

    Each config's per-scenario dev summaries are aggregated by `rule`; the highest spell_f1_mean
    wins, ties go to the higher macro_f1, then to the earlier config in canonical order.
    """
    scores = [{name: aggregate([summaries[order][name]
                                for summaries in summaries_by_scenario.values()], rule)
               for name in ("spell_f1_mean", "macro_f1")}
              for order in range(len(grid))]
    best = min(range(len(grid)),
               key=lambda order: (-scores[order]["spell_f1_mean"], -scores[order]["macro_f1"],
                                  order))
    return grid[best], [dict(config=config_record(config), **row)
                        for config, row in zip(grid, scores)]


def _number_neighbours(value, kind, factor):
    """value * factor and value / factor; ints are rounded and moved at least one step."""
    if kind is int:
        return [max(value + 1, int(round(value * factor))),
                min(value - 1, int(round(value / factor)))]
    if value == 0:
        return []
    return [round(value * factor, 6), round(value / factor, 6)]


def neighbours(config, factor=SEARCH_FACTOR):
    """Valid one-field moves from a GridConfig, in SEARCH_NUMERIC then SEARCH_CATEGORICAL order.

    A numeric field is multiplied and divided by `factor`; a disabled field (a negative sentinel)
    is only enabled to its table default. A categorical field takes each other allowed value.
    Fields inactive under the current config are skipped; DecoderConfig rejects invalid moves.
    The tagging config is kept as it is.
    """
    decoder = config.decoder
    moves = []
    for name, enable, active in SEARCH_NUMERIC:
        if not active(decoder):
            continue
        value = getattr(decoder, name)
        if enable is not None and value < 0:
            moves.append((name, enable))
        else:
            kind = type(DecoderConfig.__dataclass_fields__[name].default)
            moves.extend((name, moved) for moved in _number_neighbours(value, kind, factor))
    for name, values, active in SEARCH_CATEGORICAL:
        if not active(decoder):
            continue
        moves.extend((name, value) for value in values if value != getattr(decoder, name))
    found = []
    for name, value in moves:
        try:
            found.append(GridConfig(replace(decoder, **{name: value}), config.tagging))
        except ValueError:
            continue
    return found


def coordinate_search(start, score_of, steps, factor=SEARCH_FACTOR):
    """(final config, trail) of a deterministic coordinate search from `start`.

    score_of(config) gives {"spell_f1_mean", "macro_f1"}. Each step moves to the best neighbour
    (higher spell_f1_mean, then macro_f1, then earlier canonical key) if it strictly beats the
    current config on (spell_f1_mean, macro_f1); the search stops after `steps` moves or when no
    neighbour improves. The trail starts with `start` and adds each accepted config.
    """
    def objective(config):
        row = score_of(config)
        return row["spell_f1_mean"], row["macro_f1"]

    def entry(config):
        spell, macro = objective(config)
        return {"config": config_record(config), "spell_f1_mean": spell, "macro_f1": macro}

    current, trail = start, [entry(start)]
    for _ in range(steps):
        candidates = {canonical_key(config): config for config in neighbours(current, factor)}
        if not candidates:
            break
        best = min(sorted(candidates),
                   key=lambda key: [-value for value in objective(candidates[key])])
        if objective(candidates[best]) <= objective(current):
            break
        current = candidates[best]
        trail.append(entry(current))
    return current, trail


def dev_scorer(dev_bundles_by_scenario, rule, known=()):
    """score_of(config) for coordinate_search: aggregated dev spell_f1_mean and macro_f1, cached
    by canonical_key and seeded with `known` (config, select_across row) pairs. Sees development
    bundles only."""
    names = ("spell_f1_mean", "macro_f1")
    cache = {canonical_key(config): {name: row[name] for name in names} for config, row in known}

    def score_of(config):
        key = canonical_key(config)
        if key not in cache:
            summaries = [dev_summaries([config], bundles)[0]
                         for bundles in dev_bundles_by_scenario.values()]
            cache[key] = {name: aggregate([summary[name] for summary in summaries], rule)
                          for name in names}
        return cache[key]

    score_of.cache = cache
    return score_of


def dev_table(grid, summaries):
    return [dict(config=config_record(config), **summary)
            for config, summary in zip(grid, summaries)]


def select_config(grid, dev_bundles):
    """(chosen config, per-config development summaries). Sees development bundles only."""
    summaries = dev_summaries(grid, dev_bundles)
    chosen, _ = select_across(grid, {None: summaries})
    return chosen, dev_table(grid, summaries)


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return float(ordered[middle]) if len(ordered) % 2 else (ordered[middle - 1]
                                                            + ordered[middle]) / 2.0


def _changes(labels):
    """Frames t where the label differs from t - 1 and neither side is X."""
    return [t for t in range(1, len(labels)) if labels[t] != labels[t - 1]
            and labels[t] != UNKNOWN and labels[t - 1] != UNKNOWN]


def diagnose(gt_codes, labels, causes, ball_missing, fps=25, tolerance_s=2.0, support=None):
    """Aggregate error diagnostics against the synthetic latent causes. Pure; counts only.

    by_cause: graded-frame count, accuracy and gt -> predicted label confusion per latent cause.
    by_ball_missing: graded-frame accuracy when the ball was dropped or observed.
    boundaries: each GT label change between graded labels, grouped by the cause of its first
    frame (and "all"), matched to the nearest predicted change within +/- tolerance_s; the matched
    share and mean / median absolute offset in frames of the matched ones, and their mean / median
    signed offset (predicted minus GT frame, so positive means the prediction lags). An early and
    a late predicted change at equal distance resolve to the early one.
    dead_spells: the number of maximal D runs in the ground truth and in the prediction, so a
    stoppage split into several D spells shows up as fragmentation.
    spell_failures: the class of every unmatched GT and predicted possession spell at the
    headline IoU, with counts by latent cause (spell_failures()); with `support`
    (raw_evidence, control, ball_observed per frame) it adds their evidence support counts.
    """
    gt = [CODE_LABELS[int(code)] for code in gt_codes]
    pred = [str(label).strip().upper() for label in labels]
    if not len(gt) == len(pred) == len(causes) == len(ball_missing):
        raise ValueError("diagnose inputs differ in length")

    by_cause, by_missing = {}, {}
    for t, truth in enumerate(gt):
        if truth == UNKNOWN:
            continue
        hit = int(pred[t] == truth)
        row = by_cause.setdefault(causes[t], {"frames": 0, "correct": 0, "confusion": {}})
        row["frames"] += 1
        row["correct"] += hit
        counts = row["confusion"].setdefault(truth, {})
        counts[pred[t]] = counts.get(pred[t], 0) + 1
        split = by_missing.setdefault("missing" if ball_missing[t] else "observed",
                                      {"frames": 0, "correct": 0})
        split["frames"] += 1
        split["correct"] += hit
    for row in list(by_cause.values()) + list(by_missing.values()):
        row["accuracy"] = row.pop("correct") / float(row["frames"])

    tolerance = int(round(tolerance_s * fps))
    predicted = _changes(pred)
    offsets = {}
    for t in _changes(gt):
        nearest = min((p - t for p in predicted if abs(p - t) <= tolerance),
                      key=lambda signed: (abs(signed), signed), default=None)
        for group in (causes[t], "all"):
            offsets.setdefault(group, []).append(nearest)
    boundaries = {}
    for group, found in offsets.items():
        signed = [offset for offset in found if offset is not None]
        matched = [abs(offset) for offset in signed]
        boundaries[group] = {
            "count": len(found),
            "matched_share": len(matched) / float(len(found)),
            "offset_mean": sum(matched) / float(len(matched)) if matched else None,
            "offset_median": _median(matched) if matched else None,
            "signed_offset_mean": sum(signed) / float(len(signed)) if signed else None,
            "signed_offset_median": _median(signed) if signed else None,
        }
    return {"by_cause": by_cause, "by_ball_missing": by_missing, "boundaries": boundaries,
            "boundary_tolerance_frames": tolerance,
            "dead_spells": {"true": _dead_spells(gt), "predicted": _dead_spells(pred)},
            "spell_failures": spell_failures(gt_codes, pred, causes, support)}


def _classify_unmatched(spells, others, overlaps, matched, causes, names=None):
    """Failure classes of the unmatched `spells` against `others` (see spell_failures).

    `overlaps` are spell_overlaps() triples re-keyed (iou, spell index, other index), in the
    scorer's matching order, so each spell's first entry is its best same-team overlap. A
    `names` dict gets {spell index: class} for every unmatched spell."""
    partners, other_partners = {}, {}
    for _, index, other in overlaps:
        partners.setdefault(index, []).append(other)
        other_partners.setdefault(other, []).append(index)
    classes = {name: {"count": 0, "by_cause": {}} for name in SPELL_FAILURES}
    starts, ends = [], []
    for index, (start, stop, _) in enumerate(spells):
        if matched[index]:
            continue
        found = partners.get(index, [])
        if not found:
            name = "no_overlap"
        elif len(other_partners[found[0]]) >= 2:
            name = "merged"
        elif sum(min(stop, others[other][1]) - max(start, others[other][0])
                 >= SPELL_SPLIT_COVER * (stop - start) for other in found) >= 2:
            name = "split"
        else:
            name = "misaligned"
            starts.append(others[found[0]][0] - start)
            ends.append(others[found[0]][1] - stop)
        if names is not None:
            names[index] = name
        row = classes[name]
        row["count"] += 1
        row["by_cause"][causes[start]] = row["by_cause"].get(causes[start], 0) + 1
    misaligned = classes["misaligned"]
    for key, offsets in (("start", starts), ("end", ends)):
        misaligned[key + "_offset_mean"] = sum(offsets) / float(len(offsets)) if offsets else None
        misaligned[key + "_abs_offset_mean"] = (sum(abs(o) for o in offsets) / float(len(offsets))
                                                if offsets else None)
    return {"count": len(spells), "matched": int(sum(matched)), "classes": classes}


def _share_bucket(own, frames):
    """SUPPORT_BUCKETS entry for `own` of `frames` frames."""
    if own == 0:
        return SUPPORT_BUCKETS[0]
    share = own / float(frames)
    return SUPPORT_BUCKETS[1 if share <= 0.25 else 2 if share <= 0.5 else 3]


def spell_support(spells, names, gt, raw_evidence, control, ball_observed, causes):
    """Evidence behind the unmatched GT spells of the SUPPORT_CLASSES. Pure; counts only.

    `spells` are possession_spells() of the GT codes, `names` the {spell index: failure class} of
    the unmatched ones (_classify_unmatched), `gt` the per-frame GT labels (H/A/D/X). Per frame,
    aligned with them: `raw_evidence` is the nearest-player evidence before any stationary-ball,
    kick-off, tagging or other rewrite (read_evidence), `control` the teams with a player within
    control_radius of the ball (read_team_control) and `ball_observed` whether the tracking has a
    ball. Returns {class: {latent cause of the spell's first frame: row}}; each row counts its
    spells and, over their graded frames, the frames with an observed ball, whose raw evidence
    names the spell's team (own) or the other team (other), and where both teams are near the
    ball (both_near), plus the spells per own-team share bucket (own / graded frames)."""
    if not len(gt) == len(raw_evidence) == len(control) == len(ball_observed) == len(causes):
        raise ValueError("spell_support inputs differ in length")
    support = {name: {} for name in SUPPORT_CLASSES}
    for index, name in sorted(names.items()):
        if name not in support:
            continue
        start, stop, code = spells[index]
        team = CODE_LABELS[int(code)]
        row = support[name].setdefault(causes[start], dict(
            {key: 0 for key in SUPPORT_COUNTS}, own_share={b: 0 for b in SUPPORT_BUCKETS}))
        frames = [t for t in range(start, stop) if gt[t] != UNKNOWN]
        own = sum(raw_evidence[t] == team for t in frames)
        row["spells"] += 1
        row["frames"] += len(frames)
        row["ball_observed"] += sum(bool(ball_observed[t]) for t in frames)
        row["own"] += own
        row["other"] += sum(raw_evidence[t] in TEAM_LABELS and raw_evidence[t] != team
                            for t in frames)
        row["both_near"] += sum(len(control[t]) >= 2 for t in frames)
        row["own_share"][_share_bucket(own, len(frames))] += 1
    return support


def support_summary(support, cause="contested"):
    """(spells with own-team share > 0.25, all spells) over the SUPPORT_CLASSES rows of one
    start cause of a spell_support() result: cycle 032's decision count."""
    rows = [support[name][cause] for name in SUPPORT_CLASSES if cause in support.get(name, {})]
    return (sum(row["own_share"][bucket] for row in rows for bucket in SUPPORT_BUCKETS[2:]),
            sum(row["spells"] for row in rows))


def spell_failures(gt_codes, labels, causes, support=None):
    """Why possession spells go unmatched at the scorer's headline IoU. Pure; counts only.

    Spells are the scorer's (possession_spells on the abstention-filled prediction) and so is the
    greedy one-to-one match (spell_overlaps, matched_gt_flags at SPELL_IOU_HEADLINE). Each
    unmatched GT spell gets exactly one class, tested in SPELL_FAILURES order:
      no_overlap  no same-team predicted spell overlaps it (wrong team throughout, or missed);
      merged      its best same-team predicted spell (highest IoU) overlaps 2+ GT spells;
      split       2+ same-team predicted spells each cover >= SPELL_SPLIT_COVER of it;
      misaligned  anything else; mean signed (and absolute) start and end offsets of the best
                  predicted spell minus the GT spell, in frames.
    Unmatched predicted spells (spurious) get the same rule with GT and prediction swapped, so a
    predicted "merged" is a fragment whose best GT spell overlaps 2+ predictions, and a predicted
    "split" spans 2+ GT spells. Each class also counts spells by the latent cause of the spell's
    first frame. With `support` (raw_evidence, control, ball_observed; see spell_support) the
    result adds "gt_support" for the unmatched GT spells of the SUPPORT_CLASSES."""
    import numpy as np
    import possession_metrics as pm
    gt_codes = np.asarray(gt_codes)
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pm.label_codes(list(labels)), runs)
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    swapped = [(iou, pred, gt) for iou, gt, pred in overlaps]
    # The greedy match is symmetric, so replaying the same order with the roles swapped marks
    # exactly the predicted spells the scorer matched.
    gt_matched = pm.matched_gt_flags(overlaps, len(gt_spells), pm.SPELL_IOU_HEADLINE)
    pred_matched = pm.matched_gt_flags(swapped, len(pred_spells), pm.SPELL_IOU_HEADLINE)
    names = {}
    result = {"iou": pm.SPELL_IOU_HEADLINE, "split_cover": SPELL_SPLIT_COVER,
              "gt": _classify_unmatched(gt_spells, pred_spells, overlaps, gt_matched, causes,
                                        names),
              "predicted": _classify_unmatched(pred_spells, gt_spells, swapped, pred_matched,
                                               causes)}
    if support is not None:
        gt = [CODE_LABELS[int(code)] for code in gt_codes]
        result["gt_support"] = spell_support(gt_spells, names, gt, *support, causes)
    return result


def _dead_spells(labels):
    """Number of maximal runs of D."""
    return sum(label == "D" and (t == 0 or labels[t - 1] != "D")
               for t, label in enumerate(labels))


def pooled_diagnostics(bundles, labels_by_bundle, config=None):
    """diagnose() over the bundles joined into one sequence, as heldout_report pools seeds.

    With a DecoderConfig, the unmatched-spell evidence support (spell_support) is added from each
    bundle's support_inputs(config)."""
    support = None
    if config is not None:
        inputs = [bundle.support_inputs(config) for bundle in bundles]
        support = tuple([value for part in inputs for value in part[k]] for k in range(3))
    return diagnose([code for bundle in bundles for code in bundle.gt_codes],
                    [label for labels in labels_by_bundle for label in labels],
                    [cause for bundle in bundles for cause in bundle.causes],
                    [missing for bundle in bundles for missing in bundle.ball_missing],
                    bundles[0].fps, support=support)


def dev_diagnostics(configs, bundles_by_scenario):
    """({scenario: {name: diagnostics}}, {name: diagnostics pooled over every scenario}) for
    each named GridConfig in `configs` ({name: GridConfig}) on each scenario's development
    bundles, pooled over seeds, with the unmatched GT spells' evidence support. Reported only;
    never feeds selection or search."""
    labels = {scenario: {name: [bundle.decoder_labels(config.decoder, config.tagging)
                                for bundle in bundles]
                         for name, config in configs.items()}
              for scenario, bundles in bundles_by_scenario.items()}
    by_scenario = {scenario: {name: pooled_diagnostics(bundles, labels[scenario][name],
                                                       configs[name].decoder)
                              for name in configs}
                   for scenario, bundles in bundles_by_scenario.items()}
    every = [bundle for bundles in bundles_by_scenario.values() for bundle in bundles]
    pooled = {name: pooled_diagnostics(every, [row for scenario in bundles_by_scenario
                                               for row in labels[scenario][name]],
                                       configs[name].decoder)
              for name in configs}
    return by_scenario, pooled


def heldout_report(labels_of, heldout_bundles, bootstrap_n=0, seed_scores=None):
    """Seed-mean metrics and pooled latent-cause diagnostics; with bootstrap_n, block-bootstrap
    intervals over all held-out seeds. A `seed_scores` list gets one {"seed", spell_f1_mean,
    macro_f1, accuracy} row per bundle; the report itself is the same either way.

    Pooling joins the seeds into one sequence. Each bundle starts and ends on ungraded X frames,
    so the scorer keeps every half a separate run, and no segment or boundary spans two matches.
    """
    import numpy as np
    labels_by_bundle = [list(labels_of(bundle)) for bundle in heldout_bundles]
    results = [score(bundle.gt_codes, labels)
               for bundle, labels in zip(heldout_bundles, labels_by_bundle)]
    if seed_scores is not None:
        seed_scores.extend(dict(seed=bundle.seed, **{name: result[name] for name in PAIRED})
                           for bundle, result in zip(heldout_bundles, results))
    report = summarise(results)
    gt = np.concatenate([bundle.gt_codes for bundle in heldout_bundles])
    labels = [label for bundle_labels in labels_by_bundle for label in bundle_labels]
    report["diagnostics"] = pooled_diagnostics(heldout_bundles, labels_by_bundle)
    if bootstrap_n:
        pooled = score(gt, labels, bootstrap_n)
        report["bootstrap_n"] = bootstrap_n
        report["spell_f1_interval"] = [pooled["spell_bootstrap_low"],
                                       pooled["spell_bootstrap_high"]]
        report["macro_f1_interval"] = [pooled["bootstrap_low"], pooled["bootstrap_high"]]
    return report


def git_revision():
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return revision + ("+dirty" if dirty else "")


def cross_scenario_mean(reports):
    """Unweighted mean over scenarios of the seed-mean held-out metrics (no diagnostics)."""
    reports = list(reports)
    return {name: sum(report[name] for report in reports) / float(len(reports))
            for name in REPORTED + ("abs_spell_count_error",)}


def mean_delta_interval(deltas, draws=PAIRED_DRAWS, rng_seed=PAIRED_RNG_SEED,
                        level=PAIRED_LEVEL):
    """[low, high] percentile bootstrap interval of the mean of `deltas`, resampling seeds."""
    rng = random.Random(rng_seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / float(n) for _ in range(draws))
    tail = (1.0 - level) / 2.0
    return [means[int(tail * (draws - 1))], means[int(round((1.0 - tail) * (draws - 1)))]]


def paired_deltas(chosen_rows, reference_rows):
    """Per-seed chosen minus reference deltas of PAIRED metrics, with their mean, wins/ties/losses
    (|delta| <= TIE_TOLERANCE is a tie) and a seed bootstrap interval of the mean. Rows are
    {"seed", spell_f1_mean, macro_f1, accuracy}, paired by seed. Each metric's interval reuses
    the same rng seed, so adding a metric leaves the others' intervals unchanged."""
    chosen = {row["seed"]: row for row in chosen_rows}
    reference = {row["seed"]: row for row in reference_rows}
    if not chosen or set(chosen) != set(reference):
        raise ValueError("chosen and reference must be scored on the same held-out seeds")
    seeds = [row["seed"] for row in chosen_rows]
    per_seed = [{"seed": seed,
                 "chosen": {name: chosen[seed][name] for name in PAIRED},
                 "reference": {name: reference[seed][name] for name in PAIRED},
                 "delta": {name: chosen[seed][name] - reference[seed][name] for name in PAIRED}}
                for seed in seeds]
    metrics = {}
    for name in PAIRED:
        deltas = [row["delta"][name] for row in per_seed]
        metrics[name] = {
            "mean_delta": sum(deltas) / float(len(deltas)),
            "wins": sum(delta > TIE_TOLERANCE for delta in deltas),
            "ties": sum(abs(delta) <= TIE_TOLERANCE for delta in deltas),
            "losses": sum(delta < -TIE_TOLERANCE for delta in deltas),
            "interval": mean_delta_interval(deltas),
        }
    return {"seeds": len(seeds), "bootstrap_draws": PAIRED_DRAWS, "rng_seed": PAIRED_RNG_SEED,
            "level": PAIRED_LEVEL, "metrics": metrics, "per_seed": per_seed}


def pool_seed_scores(rows_by_scenario):
    """Per-seed rows averaged over scenarios (unweighted, as cross_scenario_mean)."""
    tables = list(rows_by_scenario.values())
    return [dict(seed=row["seed"], **{name: sum(table[order][name] for table in tables)
                                      / float(len(tables)) for name in PAIRED})
            for order, row in enumerate(tables[0])]


def run_experiment(generator_config, grid, dev_seeds, heldout_seeds, out_dir=DEFAULT_OUT_DIR,
                   work_dir=None, bootstrap_n=0, scenarios=None, select="mean", search_steps=0,
                   reference=None, reference_source=None, seed_ledger=None, final_audit=False):
    """Select on development seeds, score on held-out seeds, append and return the record.

    Without `scenarios` the generator config is one implicit scenario and the record has its
    original layout. With `scenarios` ({name: SyntheticConfig}; generator_config is unused), each
    scenario has its own bundles and the record adds per-scenario dev and held-out tables.
    With `search_steps` > 0 a coordinate search on the same dev scores continues from the grid
    winner; the record adds its trail and its final config becomes chosen_config.
    With a `reference` GridConfig (and its load_reference source), the reference is scored on
    the same held-out bundles after the choice is made and the record adds "reference".
    With a `seed_ledger` path the seeds are checked against it before any bundle is generated
    (check_seed_ledger) and, after the record is logged, the held-out seeds are marked inspected
    (or, with `final_audit`, the audit used). None bypasses the ledger; the record says which.
    """
    check_seed_lists(dev_seeds, heldout_seeds)
    if seed_ledger is not None:
        check_seed_ledger(load_seed_ledger(seed_ledger), dev_seeds, heldout_seeds, final_audit)
    elif final_audit:
        raise ValueError("final_audit needs a seed ledger")
    check_paths([out_dir, work_dir])
    aggregate([0.0], select)  # validates the rule before any bundle is generated
    if search_steps < 0:
        raise ValueError("search_steps must be >= 0")
    named = scenarios is not None
    if named:
        if not scenarios:
            raise ValueError("need at least one scenario")
        for name in scenarios:
            if not SCENARIO_NAME.match(name):
                raise ValueError("bad scenario name {!r}".format(name))
    configs = dict(scenarios) if named else {None: generator_config}
    with tempfile.TemporaryDirectory() as temp:
        bundles_dir = work_dir or temp

        def folder(name, part):
            return os.path.join(bundles_dir, name, part) if named else os.path.join(
                bundles_dir, part)

        dev_bundles = {name: generate_bundles(dev_seeds, config, folder(name, "dev"))
                       for name, config in configs.items()}
        summaries = {name: dev_summaries(grid, dev_bundles[name]) for name in configs}
        chosen, scores = select_across(grid, summaries, select)
        search = None
        if search_steps:
            start = chosen
            chosen, trail = coordinate_search(
                start, dev_scorer(dev_bundles, select, zip(grid, scores)), search_steps)
            search = {"steps_requested": search_steps, "steps_taken": len(trail) - 1,
                      "factor": SEARCH_FACTOR, "start_config": config_record(start),
                      "trail": trail}
        diagnosed = {"chosen": chosen}
        if reference is not None:
            diagnosed["reference"] = reference
        diagnostics_by_scenario, diagnostics_pooled = dev_diagnostics(diagnosed, dev_bundles)
        results, reference_results, seed_scores = {}, {}, {}
        for name, config in configs.items():
            heldout = generate_bundles(heldout_seeds, config, folder(name, "heldout"))
            chosen_seeds = [] if reference is not None else None
            results[name] = {
                "chosen": heldout_report(
                    lambda bundle: bundle.decoder_labels(chosen.decoder, chosen.tagging),
                    heldout, bootstrap_n, chosen_seeds),
                BASELINE_NAME: heldout_report(lambda bundle: bundle.baseline_labels(), heldout,
                                              bootstrap_n),
            }
            if reference is not None:
                reference_seeds = []
                reference_results[name] = heldout_report(
                    lambda bundle: bundle.decoder_labels(reference.decoder, reference.tagging),
                    heldout, bootstrap_n, reference_seeds)
                seed_scores[name] = (chosen_seeds, reference_seeds)

    record = {
        "kind": "synthetic",
        "note": BANNER,
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "generator": synthetic_match.FORMAT,
        "grid": [config_record(config) for config in grid],
        "dev_seeds": list(dev_seeds),
        "heldout_seeds": list(heldout_seeds),
        "chosen_config": config_record(chosen),
        "seed_ledger": {"path": seed_ledger, "bypassed": seed_ledger is None,
                        "final_audit": final_audit},
    }
    if search is not None:
        record["search"] = search
    if named:
        record.update({
            "scenarios": {name: asdict(config) for name, config in configs.items()},
            "selection_rule": select,
            "selection": "max {} over scenarios of each scenario's mean dev spell_f1_mean, then "
                         "macro_f1 likewise, then canonical grid order".format(select),
            "dev": scores,
            "dev_by_scenario": {name: dev_table(grid, summaries[name]) for name in configs},
            "dev_diagnostics": diagnostics_by_scenario,
            "dev_diagnostics_pooled": diagnostics_pooled,
            "heldout_by_scenario": results,
            "heldout": {method: cross_scenario_mean(results[name][method] for name in configs)
                        for method in ("chosen", BASELINE_NAME)},
        })
    else:
        record.update({
            "generator_config": asdict(generator_config),
            "selection": SELECTION_TEXT,
            "dev": dev_table(grid, summaries[None]),
            "dev_diagnostics": diagnostics_by_scenario[None],
            "heldout": results[None],
        })
    if reference is not None:
        block = {"source": reference_source, "config": config_record(reference)}
        if named:
            block.update({
                "heldout": cross_scenario_mean(reference_results.values()),
                "heldout_by_scenario": reference_results,
                "paired": paired_deltas(
                    pool_seed_scores({name: pair[0] for name, pair in seed_scores.items()}),
                    pool_seed_scores({name: pair[1] for name, pair in seed_scores.items()})),
                "paired_by_scenario": {name: paired_deltas(*pair)
                                       for name, pair in seed_scores.items()},
            })
        else:
            block.update({"heldout": reference_results[None],
                          "paired": paired_deltas(*seed_scores[None])})
        record["reference"] = block
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, LOG_NAME)
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    if seed_ledger is not None:
        update_seed_ledger(seed_ledger, dev_seeds, heldout_seeds, final_audit)
    record["log_path"] = path
    return record


def config_names(sources):
    """Short unique column names from load_reference sources: file stem, "#record" for a log
    record, "-2", "-3", ... for repeats."""
    names = []
    for source in sources:
        base = os.path.splitext(os.path.basename(source.get("path", "config")))[0]
        if "record" in source:
            base += "#{}".format(source["record"])
        name, copy = base, 1
        while name in names:
            copy += 1
            name = "{}-{}".format(base, copy)
        names.append(name)
    return names


def run_diagnose_only(generator_config, configs, dev_seeds, out_dir=DEFAULT_OUT_DIR,
                      work_dir=None, scenarios=None):
    """Development-only diagnostics of given configs; no search and no held-out bundles.

    `configs` is a list of (GridConfig, load_reference source). Each config is decoded on every
    scenario's development bundles and diagnosed pooled over seeds (and over all scenarios).
    Appends one record to <out_dir>/DIAGNOSE_LOG_NAME, kept apart from the experiment log so
    that `--reference LOG` and `sequence_decoder.py --config-file LOG` are unaffected.
    """
    if not dev_seeds or len(set(dev_seeds)) != len(dev_seeds):
        raise ValueError("need distinct development seeds: {}".format(list(dev_seeds)))
    if not configs:
        raise ValueError("need at least one config to diagnose")
    check_paths([out_dir, work_dir])
    named = scenarios is not None
    if named and not scenarios:
        raise ValueError("need at least one scenario")
    generators = dict(scenarios) if named else {None: generator_config}
    names = config_names([source for _, source in configs])
    by_name = dict(zip(names, (config for config, _ in configs)))
    with tempfile.TemporaryDirectory() as temp:
        bundles_dir = work_dir or temp
        dev_bundles = {name: generate_bundles(
            dev_seeds, config, os.path.join(bundles_dir, name, "dev") if named
            else os.path.join(bundles_dir, "dev")) for name, config in generators.items()}
        summaries = {scenario: dict(zip(names, dev_summaries(list(by_name.values()), bundles)))
                     for scenario, bundles in dev_bundles.items()}
        diagnostics_by_scenario, diagnostics_pooled = dev_diagnostics(by_name, dev_bundles)

    record = {
        "kind": "synthetic_dev_diagnostics",
        "note": BANNER,
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "generator": synthetic_match.FORMAT,
        "dev_seeds": list(dev_seeds),
        "configs": [{"name": name, "source": source, "config": config_record(config)}
                    for name, (config, source) in zip(names, configs)],
    }
    if named:
        record.update({
            "scenarios": {name: asdict(config) for name, config in generators.items()},
            "dev_by_scenario": summaries,
            "dev_diagnostics": diagnostics_by_scenario,
            "dev_diagnostics_pooled": diagnostics_pooled,
        })
    else:
        record.update({
            "generator_config": asdict(generator_config),
            "dev": summaries[None],
            "dev_diagnostics": diagnostics_by_scenario[None],
        })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, DIAGNOSE_LOG_NAME)
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    record["log_path"] = path
    return record


def print_dev_diagnostics(record, names):
    """Development diagnostics of the named configs side by side: the per-cause table (pooled
    over seeds, and over scenarios when there are several) and a per-scenario summary."""
    if "scenarios" in record:
        pooled = record["dev_diagnostics_pooled"]
        print_diagnostics("development by latent cause (all scenarios and dev seeds)",
                          [(name, pooled[name]) for name in names])
        print_scenario_diagnostics(record["dev_diagnostics"], names)
        rows = [("all", name, pooled[name]) for name in names] + [
            (scenario, name, diagnostics[name])
            for scenario, diagnostics in record["dev_diagnostics"].items() for name in names]
    else:
        print_diagnostics("development by latent cause (all dev seeds)",
                          [(name, record["dev_diagnostics"][name]) for name in names])
        rows = [("all", name, record["dev_diagnostics"][name]) for name in names]
    print_spell_failures(rows)
    print_spell_support(rows)


def print_diagnose_only(record):
    print(BANNER)
    print("development diagnostics only (no search, no held-out seeds); dev seeds {}".format(
        record["dev_seeds"]))
    names = [entry["name"] for entry in record["configs"]]
    for entry in record["configs"]:
        print("  {}: {}".format(entry["name"], json.dumps(entry["config"], sort_keys=True)))
    if "scenarios" in record:
        print("scenarios {}".format(list(record["scenarios"])))
        rows = [(scenario, summaries) for scenario, summaries in record["dev_by_scenario"].items()]
    else:
        rows = [("all", record["dev"])]
    print("\ndevelopment (mean over seeds)")
    for scenario, summaries in rows:
        for name in names:
            print("  {:<18} {:<20} spellF1 {:.4f}  macroF1 {:.4f}  acc {:.4f}".format(
                scenario, name, summaries[name]["spell_f1_mean"], summaries[name]["macro_f1"],
                summaries[name]["accuracy"]))
    print_dev_diagnostics(record, names)
    print("\nlogged to {}".format(record["log_path"]))


def print_report(record):
    print(BANNER)
    print("dev seeds {}  held-out seeds {}".format(record["dev_seeds"], record["heldout_seeds"]))
    scenarios = "scenarios" in record
    if scenarios:
        print("scenarios {}  selection {}".format(list(record["scenarios"]),
                                                  record["selection_rule"]))
        print("\ndevelopment ({} over scenarios of seed means)".format(record["selection_rule"]))
    else:
        print("\ndevelopment (mean over seeds)")
    for row in record["dev"]:
        config = {key: row["config"][key] for key in sorted(row["config"])}
        marker = "*" if row["config"] == record["chosen_config"] else " "
        print(" {} spellF1 {:.4f}  macroF1 {:.4f}  {}".format(
            marker, row["spell_f1_mean"], row["macro_f1"], json.dumps(config, sort_keys=True)))
    if "search" in record:
        print_search(record["search"])
    print_dev_diagnostics(record, ["chosen"] + (["reference"] if "reference" in record else []))
    print("\nheld-out (mean over {})".format("scenarios of seed means" if scenarios else "seeds"))
    print("  {:<18} {:>8} {:>8} {:>8} {:>10} {:>8}".format(
        "", "spellF1", "macroF1", "acc", "spellErr", "frag"))
    for name, report in record["heldout"].items():
        print("  {:<18} {:>8.4f} {:>8.4f} {:>8.4f} {:>+10.1f} {:>8.2f}".format(
            "sequence_decoder" if name == "chosen" else name, report["spell_f1_mean"],
            report["macro_f1"], report["accuracy"], report["spell_count_error"],
            report["fragmentation"]))
        if "spell_f1_interval" in report:
            print("  {:<18} spellF1 {:.4f}..{:.4f}  macroF1 {:.4f}..{:.4f}  ({} draws)".format(
                "", *report["spell_f1_interval"], *report["macro_f1_interval"],
                report["bootstrap_n"]))
    if "reference" in record:
        print_reference(record["reference"])
    if scenarios:
        print_scenarios(record["heldout_by_scenario"])
    print("\n" + HELDOUT_AUDIT_NOTE)
    print("logged to {}".format(record["log_path"]))


def print_reference(reference):
    """One line: the chosen config's paired held-out deltas against the reference config."""
    source = reference["source"] or {}
    where = source.get("path", "?") + (" record {}".format(source["record"])
                                       if "record" in source else "")
    paired = reference["paired"]
    parts = []
    for name, label in (("spell_f1_mean", "spellF1"), ("macro_f1", "macroF1"),
                        ("accuracy", "acc")):
        stats = paired["metrics"][name]
        parts.append("{} {:+.4f} [{:+.4f}..{:+.4f}] W/T/L {}/{}/{}".format(
            label, stats["mean_delta"], *stats["interval"], stats["wins"], stats["ties"],
            stats["losses"]))
    print("  vs reference ({}; chosen - reference, {} seeds, {:.0%} seed bootstrap): {}".format(
        where, paired["seeds"], paired["level"], "  ".join(parts)))


def print_search(search):
    """The coordinate search trail: dev scores and the fields each accepted step changed."""
    print("\ncoordinate search from the grid winner ({} of {} steps taken, factor {})".format(
        search["steps_taken"], search["steps_requested"], search["factor"]))
    previous = {}
    for step, row in enumerate(search["trail"]):
        changed = {key: value for key, value in sorted(row["config"].items())
                   if not step or previous.get(key) != value}
        print("  {:>2} spellF1 {:.4f}  macroF1 {:.4f}  {}".format(
            step, row["spell_f1_mean"], row["macro_f1"],
            "start (grid winner)" if not step else json.dumps(changed, sort_keys=True)))
        previous = row["config"]


def print_scenarios(heldout_by_scenario):
    """Per-scenario held-out seed means, decoder beside baseline (diagnostics stay in the log)."""
    print("\nheld-out by scenario (mean over seeds)")
    print("  {:<18} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "", "spl dec", "spl base", "mac dec", "mac base", "acc dec", "acc base"))
    for name, reports in heldout_by_scenario.items():
        pair = (reports["chosen"], reports[BASELINE_NAME])
        print("  {:<18} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            name, *(report[metric] for metric in ("spell_f1_mean", "macro_f1", "accuracy")
                    for report in pair)))


def _number(value, width, digits):
    return "{:>{}}".format("-", width) if value is None else "{:>{}.{}f}".format(
        value, width, digits)


def _overall(diagnostics):
    """{frames, accuracy} over every graded frame of a diagnose() result."""
    rows = list(diagnostics["by_cause"].values())
    frames = sum(row["frames"] for row in rows)
    correct = sum(row["accuracy"] * row["frames"] for row in rows)
    return {"frames": frames, "accuracy": correct / frames if frames else None}


def _lost(row):
    """Graded frames labelled wrongly, or "-"."""
    if row is None or row["accuracy"] is None:
        return "-"
    return int(round(row["frames"] * (1.0 - row["accuracy"])))


def print_diagnostics(title, columns):
    """Per-cause table, one column group per (label, diagnose() result): graded-frame accuracy
    and frames lost, then the GT boundaries matched, their mean absolute offset and their mean /
    median signed offset (predicted minus GT) in frames."""
    labels = [label for label, _ in columns]
    diagnostics = [d for _, d in columns]
    width = max([8] + [len(label) + 5 for label in labels])
    print("\n{} (pooled; acc / lost = graded frames, bnd = GT changes, hit = share matched "
          "within +/-{} frames, off = mean |offset|, sgn / smed = mean / median signed offset, "
          "predicted - GT)".format(title, diagnostics[0]["boundary_tolerance_frames"]))
    print("  {:<18} {:>7} {} {:>5} {}".format(
        "", "frames",
        " ".join("{:>{w}} {:>{w}}".format("acc " + label, "lost " + label, w=width)
                 for label in labels),
        "bnd",
        " ".join("{:>{w}} {:>{w}} {:>{w}} {:>{w}}".format(
            "hit " + label, "off " + label, "sgn " + label, "smed " + label, w=width)
            for label in labels)))

    def line(name, frames, bounds):
        known = next((row for row in frames if row), None)
        bound = next((row for row in bounds if row), None)
        print("  {:<18} {:>7} {} {:>5} {}".format(
            name, known["frames"] if known else "-",
            " ".join("{} {:>{}}".format(_number(row and row["accuracy"], width, 4), _lost(row),
                                        width) for row in frames),
            bound["count"] if bound else "",
            " ".join("{} {} {} {}".format(
                _number(row and row["matched_share"], width, 3),
                _number(row and row["offset_mean"], width, 1),
                _number(row and row.get("signed_offset_mean"), width, 1),
                _number(row and row.get("signed_offset_median"), width, 1))
                for row in bounds) if bound else "").rstrip())

    rows = [cause for cause in synthetic_match.CAUSES
            if any(cause in d["by_cause"] or cause in d["boundaries"] for d in diagnostics)]
    for cause in rows:
        line(cause, [d["by_cause"].get(cause) for d in diagnostics],
             [d["boundaries"].get(cause) for d in diagnostics])
    line("all", [_overall(d) for d in diagnostics],
         [d["boundaries"].get("all") for d in diagnostics])
    for split in ("observed", "missing"):
        line("ball " + split, [d["by_ball_missing"].get(split) for d in diagnostics], [])


def print_spell_failures(rows):
    """One line per (scope, config name, diagnose() result): GT and predicted possession spells,
    how many were matched at the headline IoU, and the unmatched ones per failure class, with the
    mean signed start / end offsets of misaligned GT spells."""
    rows = [row for row in rows if "spell_failures" in row[2]]
    if not rows:
        return
    short = {"no_overlap": "none", "merged": "merg", "split": "split", "misaligned": "misal"}
    print("\nspell failures at IoU {} (pooled over dev seeds; GT / pred = unmatched GT / predicted "
          "spells per class, none = no same-team overlap, start / end = mean offset of misaligned "
          "GT spells in frames)".format(rows[0][2]["spell_failures"]["iou"]))
    header = " ".join("{:>5}".format(short[name]) for name in SPELL_FAILURES)
    print("  {:<18} {:<20} {:>11} {} {:>6} {:>6}  {:>11} {}".format(
        "", "", "GT hit/all", header, "start", "end", "pred hit/all", header))
    for scope, name, diagnostics in rows:
        failures = diagnostics["spell_failures"]
        gt, pred = failures["gt"], failures["predicted"]
        misaligned = gt["classes"]["misaligned"]
        print("  {:<18} {:<20} {:>11} {} {} {}  {:>11} {}".format(
            scope, name, "{}/{}".format(gt["matched"], gt["count"]),
            " ".join("{:>5}".format(gt["classes"][c]["count"]) for c in SPELL_FAILURES),
            _number(misaligned["start_offset_mean"], 6, 1),
            _number(misaligned["end_offset_mean"], 6, 1),
            "{}/{}".format(pred["matched"], pred["count"]),
            " ".join("{:>5}".format(pred["classes"][c]["count"]) for c in SPELL_FAILURES)))


def _support_total(rows):
    """spell_support rows summed."""
    total = dict({key: 0 for key in SUPPORT_COUNTS}, own_share={b: 0 for b in SUPPORT_BUCKETS})
    for row in rows:
        for key in SUPPORT_COUNTS:
            total[key] += row[key]
        for bucket in SUPPORT_BUCKETS:
            total["own_share"][bucket] += row["own_share"][bucket]
    return total


def print_spell_support(rows):
    """Evidence support of the unmatched GT none / merged spells, one line per (scope, config
    name, start cause contested or all, class), then the contested count with own share > 0.25.
    """
    rows = [row for row in rows if "gt_support" in row[2].get("spell_failures", {})]
    if not rows:
        return
    short = {"no_overlap": "none", "merged": "merg"}
    print("\nevidence support of unmatched GT none / merged spells (pooled; frames = graded, "
          "obs = ball observed, own / other = raw nearest-player evidence names the spell's / "
          "the other team, both = both teams within control_radius, share = own / frames)")
    print("  {:<18} {:<20} {:<9} {:<5} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {}".format(
        "", "", "start", "class", "spells", "frames", "obs", "own", "other", "both",
        " ".join("{:>6}".format(bucket) for bucket in SUPPORT_BUCKETS)))
    for scope, name, diagnostics in rows:
        support = diagnostics["spell_failures"]["gt_support"]
        for start in ("contested", "all"):
            for spell_class in SUPPORT_CLASSES:
                by_cause = support[spell_class]
                row = _support_total(by_cause.values() if start == "all"
                                     else [by_cause[start]] if start in by_cause else [])
                print("  {:<18} {:<20} {:<9} {:<5} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7} {}".format(
                    scope, name, start, short[spell_class],
                    *[row[key] for key in SUPPORT_COUNTS],
                    " ".join("{:>6}".format(row["own_share"][b]) for b in SUPPORT_BUCKETS)))
        above, count = support_summary(support)
        print("  {:<18} {:<20} contested none+merged with own share > 0.25: {}/{}".format(
            scope, name, above, count))


def print_scenario_diagnostics(by_scenario, names):
    """Per-scenario development summary for each named config: graded-frame accuracy overall,
    with the ball observed and with it missing, and the share of GT boundaries matched."""
    print("\ndevelopment by scenario (pooled over dev seeds; obs / miss = ball observed / "
          "missing, bnd = GT changes matched, D spells = predicted / true)")
    for scenario, diagnostics in by_scenario.items():
        for name in names:
            d = diagnostics[name]
            spells = d.get("dead_spells", {})
            print("  {:<18} {:<20} acc {} obs {} miss {} bnd {} D spells {}/{}".format(
                scenario, name, _number(_overall(d)["accuracy"], 6, 4),
                _number(d["by_ball_missing"].get("observed", {}).get("accuracy"), 6, 4),
                _number(d["by_ball_missing"].get("missing", {}).get("accuracy"), 6, 4),
                _number(d["boundaries"].get("all", {}).get("matched_share"), 5, 3),
                spells.get("predicted", "-"), spells.get("true", "-")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--heldout-seeds", type=int, nargs="+",
                        help="required unless --diagnose-only")
    parser.add_argument("--diagnose-only", nargs="+", metavar="CONFIG",
                        help="JSON configs or decoder_experiments logs (last record) to diagnose "
                             "on the dev seeds only: no search, no held-out seeds")
    parser.add_argument("--grid", help="JSON list of decoder config dicts (default: built-in)")
    parser.add_argument("--generator-config", help="JSON file of SyntheticConfig overrides")
    parser.add_argument("--n-frames", type=int, help="frames per generated match")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="block-bootstrap draws for held-out intervals (0 = off)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--work-dir", help="keep generated bundles here (default: temporary)")
    parser.add_argument("--scenarios",
                        help="JSON object of scenario name -> SyntheticConfig overrides")
    parser.add_argument("--select", choices=SELECTIONS, default="mean",
                        help="aggregate dev scores across scenarios by mean or worst case")
    parser.add_argument("--search-steps", type=int, default=0,
                        help="coordinate search steps from the grid winner on dev (0 = off)")
    parser.add_argument("--reference",
                        help="JSON config or decoder_experiments log to compare on held-out "
                             "seeds (never used for selection)")
    parser.add_argument("--reference-record", type=int,
                        help="record index in a --reference log (from 0, negative from the end; "
                             "default the last)")
    parser.add_argument("--seed-ledger", default=DEFAULT_SEED_LEDGER,
                        help="JSON ledger of inspected held-out and reserved final-audit seeds "
                             "(default experiments/seed_ledger.json); updated after the run")
    parser.add_argument("--no-seed-ledger", action="store_true",
                        help="skip the seed ledger (unit tests, ad hoc runs); the record says so")
    parser.add_argument("--final-audit", action="store_true",
                        help="score the reserved final-audit seeds as held-out, once")
    args = parser.parse_args(argv)

    seed_ledger = None if args.no_seed_ledger else args.seed_ledger
    guard_paths([args.grid, args.generator_config, args.scenarios, args.out_dir, args.work_dir,
                 args.reference, seed_ledger] + list(args.diagnose_only or []))
    if args.final_audit and seed_ledger is None:
        parser.error("--final-audit needs the seed ledger")
    if args.diagnose_only:
        return diagnose_only_main(parser, args, seed_ledger)
    if not args.heldout_seeds:
        parser.error("--heldout-seeds is required unless --diagnose-only")
    if args.search_steps < 0:
        parser.error("--search-steps must be >= 0")
    if args.reference_record is not None and not args.reference:
        parser.error("--reference-record needs --reference")
    reference, reference_source = None, None
    if args.reference:
        try:
            reference, reference_source = load_reference(args.reference, args.reference_record)
        except (OSError, ValueError) as error:
            parser.error("--reference: {}".format(error))
    try:
        check_seed_lists(args.dev_seeds, args.heldout_seeds)
        if seed_ledger is not None:
            check_seed_ledger(load_seed_ledger(seed_ledger), args.dev_seeds, args.heldout_seeds,
                              args.final_audit)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    scenarios, generator_config = load_generators(parser, args)
    record = run_experiment(generator_config, load_grid(args.grid), args.dev_seeds,
                            args.heldout_seeds, args.out_dir, args.work_dir, args.bootstrap,
                            scenarios, args.select, args.search_steps, reference,
                            reference_source, seed_ledger, args.final_audit)
    print_report(record)


def load_generators(parser, args):
    """(scenarios or None, generator config or None) from the command line."""
    overrides = {"n_frames": args.n_frames} if args.n_frames else {}
    if args.scenarios:
        try:
            return load_scenarios(args.scenarios, args.generator_config, **overrides), None
        except ValueError as error:
            parser.error(str(error))
    return None, synthetic_match.load_config(args.generator_config, **overrides)


def diagnose_only_main(parser, args, seed_ledger=None):
    """--diagnose-only: development diagnostics of the given configs; rejects the held-out,
    selection and comparison options it would ignore, and final-audit dev seeds when
    `seed_ledger` is given (it never writes the ledger)."""
    for flag, given in (("--heldout-seeds", args.heldout_seeds), ("--grid", args.grid),
                        ("--reference", args.reference),
                        ("--reference-record", args.reference_record is not None),
                        ("--search-steps", args.search_steps),
                        ("--bootstrap", args.bootstrap),
                        ("--final-audit", args.final_audit)):
        if given:
            parser.error("{} does not apply to --diagnose-only".format(flag))
    if len(set(args.dev_seeds)) != len(args.dev_seeds):
        parser.error("development seeds repeat: {}".format(args.dev_seeds))
    if seed_ledger is not None:
        try:
            check_seed_ledger(load_seed_ledger(seed_ledger), args.dev_seeds, [])
        except (OSError, ValueError) as error:
            parser.error(str(error))
    configs = []
    for path in args.diagnose_only:
        try:
            configs.append(load_reference(path))
        except (OSError, ValueError) as error:
            parser.error("--diagnose-only: {}".format(error))
    scenarios, generator_config = load_generators(parser, args)
    record = run_diagnose_only(generator_config, configs, args.dev_seeds, args.out_dir,
                               args.work_dir, scenarios)
    print_diagnose_only(record)


if __name__ == "__main__":
    main()
