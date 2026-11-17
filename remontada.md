# Remontada: CV-supported possession inference

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

The updated full-match submission uses the tactical tracking backbone plus
confidence-qualified CV observations in its missing-ball gaps. The model and
decoder are the same frozen ensemble. The earlier tracking-only submission is
preserved as a fallback. Exact reproduction of v2 requires the original licensed
tracking bundle and the existing private CV cache; a fresh clone contains neither.
The repository does not claim exact regeneration of that cache from raw video.

## Assets and alignment

The input is the automatic tactical ATD text plus its metadata. Metadata maps
track slots to home/away; a slot is not assumed to be a permanent player identity.
Pitch coordinates are converted to metres on the 105.3 × 68 m pitch. ATD frame
IDs are one-based and are converted once to zero-based video IDs. The output is
exactly 199,319 rows, frames 0–199318, at 25 fps. Temporal processing resets at
the supplied half boundaries. Frames outside those supplied active-period
bounds receive D; this rule does not depend on whether CV sees a ball.

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

## Confidence and airborne uncertainty handling

The adapter preserves image visibility while deciding whether a projected point
can enter the existing possession features. Available tactical observations are
retained. A CV fill requires confidence at least .25 and localization uncertainty
at most 2 m. Known airborne risk above .25 blocks a fill. Unknown height remains
unknown, with an explicit uncertainty mask, but no longer requires nearby tracking
anchors. That anchor requirement prevented almost all recovery in long gaps.
The unchanged upstream detector/mask provenance and motion/penalty rejection
checks still apply. No display interpolation enters the model as an observation.

Detector scores and a .5 verified-mask fallback are explicitly uncalibrated. A
2-pixel localization scale is propagated through the inverse homography; it does
not measure height. Unknown-height points can still have projection error. V2
accepts that remaining uncertainty rather than treating unavailable height as
proof that an observation is invalid. This is an input selection policy, not a
newly trained height-aware model or a calibrated airborne detector.

The old policy remains available as `require_anchors`. V2 explicitly selects
`allow_uncertain`, tested with frozen thresholds/model parameters on external
tracking stress cases and fresh synthetic matches before supplied-match inference.

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
regeneration of that cache from a fresh video run is not established. The cache
is required to reproduce the updated CV-supported submission, and must be supplied
privately under the applicable data terms. The tracking-only fallback does not
need it.

## Evaluation and reproduction

Measured results and limitations are in `EVALUATION_REPORT.md`. The controlled
external J03WN1 development tests improved spell F1 from .527273 to .690471
with noisy 10-second recovery and .543345 to .695464 with noisy 30-second
recovery. These use simulated observations on real tracking, not external video.
The fixed policy then passed ten fresh synthetic held-out cases: spell F1
.906930 → .975637, macro F1 .949185 → .993392, accuracy .952541 → .993758.
Synthetic results do not establish real-match accuracy. No thresholds or models
were tuned after this evaluation or against supplied-match labels.

The final adapter adds 265 CV observations to 6,615 tactical observations in the
five-minute interval. There are 6,880 usable observations and 620 missing frames.
All 265 fills retain an unknown-height flag. No supplied-match accuracy is claimed.
Exact CSV replay from the private cache is checked separately from upstream CV
regeneration: raw-video-only reproduction remains unverified, an unresolved
limitation against the competition's bundle-only reproduction requirement.

Install the pinned dependencies with Python 3.12. Place the licensed tactical
bundle under `data/`, then run:

```sh
.venv/bin/python scripts/cv_confidence_adapter.py \
  --manifest /private/exhaustive-part2-input-v1/manifest.json \
  --features /private/cv-fresh-updated-20261117-v2/02-fused.npz \
  --cache /private/cv-fresh-updated-20261117-v2/12-final \
  --unknown-airborne-policy allow_uncertain --out outputs/qualified-cv-v2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python run_submission.py --data-dir data --model ensemble \
  --cv-manifest outputs/qualified-cv-v2/manifest.json --out outputs/submission-v2
.venv/bin/python challenge/evaluation/validate_submission.py outputs/submission-v2/remontada.csv
```

The CSV, probabilities, availability/uncertainty masks and receipt remain under
ignored `outputs/`. The receipt pins code, model and input checksums and records
runtime. The command requires CV explicitly; `--backbone-only` selects the old
fallback. The public repository contains code and model parameters, not licensed
match data or predictions. See `README.md` for all asset and replay requirements.

Public code: [barca-possession-submission](https://github.com/christopherlii/barca-possession-submission)

Submitted CSV identity: SHA-256 `8f18e58701252edf3253ba8b977d83f16fc111b2a7f61faac3912a5a3c214f17`.
The official format validator passed: 199,319 ordered rows, H=74,522,
A=57,505, D=67,292. Full inference took 85.79s.

A clean-checkout replay using the portable private CV bundle reproduced this
CSV byte for byte, with identical probability and availability arrays. This
verifies cache-based replay, not raw-video-only regeneration.
