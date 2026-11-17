# Remontada: tactical-tracking possession inference

## Submitted approach

We predict home possession (H), away possession (A), or dead ball (D) for every
tactical-video frame using the provided automatic tactical tracking. A frozen
player temporal convolutional network (TCN) and histogram gradient-boosted tree
produce H/A/D probabilities. We average their probabilities equally after
home/away symmetry averaging. A frozen sequence decoder uses learned switch
evidence, team scale 1 and dead transition cost 16. No fitting occurs in inference.

The TCN has 24,292 parameters, 56 global features and 20 features per player
track. Its receptive radius is 541 frames at 25 fps. The tree receives 440
contextualized tracking/motion features and has 250 boosting iterations. The
NumPy forward pass preserves float64 base coordinates, float32 motion features
and float16 player features. Missing coordinates retain availability features;
they are not treated as measured zero positions.

## Inputs and alignment

Inference requires only the provided tactical ATD text and metadata XML.
Metadata maps track slots to teams; slots are not assumed to represent permanent
player identities. Coordinates use the 105.3 × 68 m pitch. One-based tracking
IDs are converted once to zero-based video IDs. The output covers all 199,319
frames at 25 fps. Temporal processing resets at the supplied half boundaries.
Frames outside those boundaries receive D and are ungraded.

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

The sequence model learns continuity through loose balls, passes and occlusions
from external and synthetic targets. It does not assign possession to every
player a pass travels near. The decoder discourages flicker while permitting
team changes through learned switch evidence. Dead-ball evidence comes from
motion, availability and context features; a missing ball is not automatically D.

Inference is offline and uses future context. Tracking errors, long observation
gaps, ambiguous challenges and differences between provider labels and the
challenge definition remain limitations. This is a frozen inference package;
training provenance and data sources are documented above and in the model receipt.

## Evaluation

On ten locked synthetic audit cases, spell F1 was 0.984322, macro F1 0.995149
and frame accuracy 0.996121. Synthetic scores do not establish real-match accuracy.
The repeatedly inspected external development match J03WN1 yielded spell F1
0.793206, macro F1 0.952313 and accuracy 0.965407; it is not an untouched test.
No full-match supplied-game accuracy is claimed.

## Reproduction

Use Python 3.12 and the public repository's default branch. The frozen model
files are included. From the repository root:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python run_submission.py --data-dir /path/to/licensed/data \
  --out outputs/submission
.venv/bin/python challenge/evaluation/validate_submission.py outputs/submission/remontada.csv
```

Use a new output directory. The data directory contains `tactical/` with
`BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_atd.xml`. The CSV, probabilities,
availability masks and hash receipt stay in ignored `outputs/`. The repository
contains source and frozen model parameters; licensed data and predictions are
not published. See `SOURCE_MANIFEST.json` and `SUBMISSION_CHECKS.json` for checks.

The submitted CSV has 199,319 ordered frames and labels H=74,465, A=57,596,
D=67,258. Its SHA-256 is
`bea82d693e2ee603da66305523252056f53a8e20c8e28fc9f935e1ef18141b20`.

Public code: [barca-possession-submission](https://github.com/christopherlii/barca-possession-submission)
