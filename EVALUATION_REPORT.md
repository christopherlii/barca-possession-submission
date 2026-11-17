# CV-supported submission v2: evaluation and validation

The numeric thresholds, frozen TCN/tree weights, feature schema and decoder were fixed before this cycle. Only unknown-height admission changed: confidence-qualified CV can fill missing tactical observations without nearby tracking anchors. No supplied-match answer key was used.

## External-data development test

Sportec J03WN1 is reused development data, not an untouched test match. The test uses real player/ball tracking and provider labels, with simulated camera projection, 40% tracking gaps, localization noise and erroneous detections. There is no synchronized external CV video in this experiment. Known provider height creates projection errors but is withheld from the policy. Confidence/error relationships are simulated, not calibrated to our detector.

| Condition | v1 spell F1 | v2 spell F1 | v2 macro F1 | v2 accuracy |
|---|---:|---:|---:|---:|
| accurate_gap10 | 0.527273 | 0.761765 | 0.941970 | 0.956188 |
| projection_gap10 | 0.527273 | 0.690471 | 0.928252 | 0.946708 |
| projection_gap30 | 0.543345 | 0.695464 | 0.932227 | 0.949428 |

All preset checks passed. Runtime 597.92s. All 36 saved score rows, including four official baselines on each policy input, were recomputed by the author and matched. Detailed aggregates and the naive-fill comparison are in `experiments/uncertain-cv-v2-results.json`.

## Fresh synthetic held-out test

Ten cases: seeds 450201/450202, five fixed scenarios, 12,000 frames per case. Seeds were checked and consumed in the ledger. The same fixed policy was evaluated once, with no subsequent tuning. Reserved final-audit data remain untouched.

| Policy | Spell F1 | Macro F1 | Frame accuracy |
|---|---:|---:|---:|
| incumbent | 0.906930 | 0.949185 | 0.952541 |
| candidate | 0.975637 | 0.993392 | 0.993758 |

| Scenario | Mean spell-F1 change |
|---|---:|
| confused_handover | +0.069734 |
| default | +0.069734 |
| hidden_dead_ball | +0.069670 |
| interceptions | +0.087129 |
| sparse_ball | +0.047266 |

All preset held-out checks passed. Runtime 111.64s; all 20 saved score rows recomputed and matched. Two seed clusters and repeated scenario structure limit uncertainty estimates. These synthetic scores do not establish supplied-match accuracy or actual CV detection quality.

## Why the old filter admitted one point

Of 302 CV observations in tactical-tracking gaps, 265 passed the unchanged confidence/localization checks. Of those 265, 261 had no short tracking bracket, three disagreed with the bracket, and one passed. All had unknown height. The v1 unknown-height anchor requirement therefore vetoed 264 otherwise eligible observations.

V2 keeps the original tactical observations and admits all 265 confidence-qualified gap observations. It still rejects low/unknown confidence, excessive localization uncertainty, known airborne-risk flags, and upstream invalid/interpolated evidence. Height remains unknown in the saved sidecar. The frozen model is not newly trained to estimate height.

## Supplied-match coverage only

- The completed CV clip remains 7,500 frames with 6,276 observed CV positions.
- Tactical ball observations in the clip: 6,615. V2 adds 265, yielding 6,880 observations and 620 missing frames (8.27%).
- Full-match observed ball positions: 113,488 tactical plus 265 CV fills = 113,753. CV is only available in the existing five-minute interval.
- No supplied-match F1 or accuracy is measured. Coverage and changes in predicted labels are not accuracy gains.
- 57 focused original tests and six standalone attachment tests passed. No independent coordinator review is claimed.

## Reproduction limitation

The generated CSV can be reproduced from the frozen model, licensed tracking and the exact private CV adapter artifacts. Exact regeneration of the upstream CV cache from raw video has not been established. Consequently, bundle-only reproduction required by the competition remains an unresolved limitation; the cache must not be posted publicly. The previous tracking-only submission is preserved at tag `submission-v1` and in the original private upload folder. Historical results are in `experiments/submission-v1-evaluation.md`.

## Generated full-match output

The official validator accepted all 199,319 ordered unique frames, 0–199318,
with H=74,522, A=57,505, D=67,292.
There are 574 label transitions and 474 H/A possession spells.
Inference took 85.79s. Probabilities are finite, normalized N×3
with recorded H/A/D order and aligned IDs. Availability and uncertainty masks
match the 265 CV fills. Original CV and frozen model hashes are unchanged.

Compared with the earlier tracking-only submission, 241 labels change.
This is prediction disagreement, not evidence of an accuracy gain or loss.
The five-minute integrated output is a slice of full-match TCN/tree inference,
so it retains full-match temporal context. The earlier five-minute GBDT output
is preserved separately and unchanged.

CSV SHA-256: `8f18e58701252edf3253ba8b977d83f16fc111b2a7f61faac3912a5a3c214f17`.
