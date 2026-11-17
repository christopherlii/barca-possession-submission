# Remontada: tactical-tracking possession inference

## Approach and submitted system

The submission predicts home possession (H), away possession (A), or dead ball
(D) for every tactical-video frame. We use the existing automatic tactical
tracking as the full-match backbone. A frozen temporal convolutional network
(TCN) and histogram gradient-boosted tree classifier each produce H/A/D evidence.
Their probabilities are averaged equally after home/away symmetry averaging.
A frozen sequence decoder uses learned switch evidence, team scale 1 and dead
transition cost 16. No training occurs during submission generation.

The TCN has 24,292 parameters and uses 56 global features plus 20 features per
player track. Its receptive radius is 541 frames at 25 fps. The tree receives
440 contextualized tracking/motion features and has 250 boosting iterations.
The implementation preserves the original float64 base-coordinate, float32
motion-feature and float16 player-feature conventions. Missing coordinates remain
missing observations; numerical feature defaults are paired with availability
features rather than invented ball measurements.

The full-match submission uses the frozen tracking backbone. We also package a
separate experimental full-match CV-enhanced output and the existing five-minute
LightGBM prediction. Those artifacts do not replace the submitted default. The
default is reproducible from the original licensed tracking bundle and the
publicly included model parameters; the CV variant depends on an additional
private, hash-verified upstream CV cache.

## Assets and alignment

The input is the automatic tactical ATD text plus its metadata. Metadata maps
track slots to home/away; a slot is not assumed to be a permanent player identity.
Pitch coordinates are converted to metres on the 105.3 × 68 m pitch. ATD frame
IDs are one-based and are converted once to zero-based video IDs. The output is
exactly 199,319 rows, frames 0–199318, at 25 fps. Temporal processing resets at
the supplied half boundaries. Frames outside those supplied active-period
bounds receive D; this rule does not depend on whether CV sees a ball.

The completed CV-supported variant admitted one additional ball observation and changed one H/A prediction. No supplied-match answers were used to judge that change.

The primary model does not use broadcast video, audio, human match event feeds,
public possession answers, or Smart Tagging. Broadcast is not synchronized to
tactical video and is excluded. The frozen LightGBM alternative uses automatic
tactical Smart Tagging as soft restart evidence, not as possession ground truth.
It has 41 features, 120 trees, 15 leaves and its original transition cost 8.

The completed exhaustive CV experiment covers frames 11185–18684 only. Its
adapter has 6,276 observed ball frames and 1,224 missing frames. It accepts direct
detections and provenance-verified SAM recovery, rejects explicit motion/penalty
failures, and excludes interpolated or display-estimated positions. These are
coverage measurements, not detection-accuracy estimates.

## Confidence and airborne uncertainty experiment

A separate adapter preserves image visibility while deciding whether a point is
usable as a ground coordinate. Available tactical observations are retained. A
CV fill requires confidence at least .25 and localization uncertainty at most
2 m. Known airborne risk above .25 blocks a fill. Unknown airborne state requires
agreement within 2 m of observed endpoints bracketing at most 50 missing frames,
inside the same temporal segment. The endpoint interpolation is a consistency
check only; its coordinates are never fed to the model as a measurement.

Detector scores and a .5 verified-mask fallback are explicitly uncalibrated.
A declared 2-pixel localization scale is propagated through the inverse
homography. It does not estimate height. Ground-plane projection of a genuinely
airborne image observation can still be wrong. Existing tactical coordinates can
also be wrong. We therefore keep the experimental adapter separate from the
reproducible tracking-only submission and do not claim it is a validated
monocular airborne detector.

## Training data and supervision

The bundled models were trained before this submission run using five permitted
other-match Sportec/DFL games: J03WMX, J03WOH, J03WOY, J03WQQ and J03WR9. Their
converted timelines contain 714,877 frames in total, including the inserted
ungraded half separators. J03WN1 is the repeatedly inspected development match,
not an untouched test. The reserved real final-audit match J03WPY is not used.

The source is [Bassek et al., *An integrated dataset of spatiotemporal and event
data in elite soccer*](https://www.nature.com/articles/s41597-025-04505-y), with
[the public dataset](https://api.figshare.com/v2/articles/28196177) released under
CC BY 4.0. Sportec/DFL supplies the data; position tracking uses TRACAB, and the
paper documents trained operators for event annotation. We use the provider's
frame-level `BallStatus` and `BallPossession` fields as targets: inactive → D,
active home → H, active away → A, unresolved → X. These fields are excluded from
model observations. They are provider supervision, not our manual annotations,
and their semantics need not exactly match the challenge's human possession rule.

Training also used 40 algorithmically generated sequences: seeds 230001–230008
under five fixed scenarios, at 24,000 frames per sequence. Synthetic labels come
from generator state, not a person annotating the supplied match. Development
seeds are recorded with the checkpoint provenance. No supplied-match human
labels, corrected tracking, or event annotations enter training or model choice.

## Possession rules and limitations

The sequence model learns possession continuity through loose balls, passes and
occlusions from external/synthetic targets. It does not give possession to every
player a flight path passes near. The decoder discourages implausible state
flicker while permitting team changes from its learned switch evidence. Dead
ball is inferred from multiple motion/availability/context features; an uncovered
CV frame is not directly assigned D. The alternative LightGBM adds automatic
restart features but does not copy phase tags into possession labels.

This is offline inference with future context, not live causal prediction.
Tracking identity errors, long ball gaps, airborne projections, ambiguous
challenges and differences between provider and challenge labels remain risks.
The existing CV cache contains nondeterministic upstream SAM recovery; exact
regeneration of that cache from a fresh video run is not established. It is not
needed to reproduce the default submitted file.

## Evaluation and reproduction

Measured development and fresh synthetic-audit results are reported separately
in `EVALUATION_REPORT.md`. Synthetic scores are not supplied-match accuracy.
The frozen ensemble achieved 0.984322 spell F1, 0.995149 macro F1 and 0.996121
frame accuracy on the ten fresh synthetic cases. Its clean J03WN1 development
replay was 0.793206 / 0.952313 / 0.965407 respectively. That real match was used
repeatedly for development, so it is not an independent accuracy estimate.
No public supplied-match score was consulted for this submission's model or
threshold selection. Format validation checks only the output file.

Install the pinned dependencies with Python 3.12, then run:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python run_submission.py --data-dir /path/to/licensed/data \
  --model ensemble --out outputs/submission
.venv/bin/python challenge/evaluation/validate_submission.py outputs/submission/remontada.csv
```

The CSV, probability arrays, availability masks and receipt remain under ignored
`outputs/`. The receipt pins code, model and input checksums, records runtime,
and states the inactive-period rule. See `README.md` for the independent
LightGBM, baseline, private CV and audit commands. The public repository contains
code and frozen model parameters, not licensed match assets or predictions.

Submission CSV SHA-256:
`bea82d693e2ee603da66305523252056f53a8e20c8e28fc9f935e1ef18141b20`.

Public code: [barca-possession-submission](https://github.com/christopherlii/barca-possession-submission)
