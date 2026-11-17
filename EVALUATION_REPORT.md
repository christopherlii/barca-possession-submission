# Submitted model: evaluation and validation

The submitted TCN/tree and decoder were frozen before submission inference.
No supplied-match answers were used for training, tuning, model selection or
this reproduction check.

## Measured evaluation

| Dataset | Spell F1 | Macro F1 | Frame accuracy |
|---|---:|---:|---:|
| Fresh synthetic audit | 0.984322 | 0.995149 | 0.996121 |
| External J03WN1 development match | 0.793206 | 0.952313 | 0.965407 |

The synthetic audit used seeds 450101 and 450102 across five fixed scenarios,
12,000 frames each. It was evaluated once after locking the model; reserved
final-audit data remained untouched. Two seed clusters and short scenarios limit
its interpretation. Synthetic scores do not establish real-match accuracy.
J03WN1 was repeatedly inspected during development and is not an untouched test.
Historical experiment records remain in the original development project.

## Submission validation

The CSV contains 199,319 ordered unique frames, 0–199318, with no missing labels:
H=74,465, A=57,596, D=67,258. It has 572 label transitions and 473 possession
spells. The official format validator passes. Probability rows are finite and
normalized with recorded H/A/D class order and matching frame IDs.

A fresh checkout and fresh Python environment reproduced the exact CSV from
the provided tactical tracking files in 98.51 seconds before source cleanup.
`SUBMISSION_CHECKS.json` records the additional reproduction check after cleanup.
Runtime measurements share machine resources and are not controlled benchmarks.

CSV SHA-256:
`bea82d693e2ee603da66305523252056f53a8e20c8e28fc9f935e1ef18141b20`

The provided tracking has 113,488 observed ball frames (56.94%). Observation
coverage, format validation and exact replay are not prediction-accuracy scores.
No full-match accuracy is claimed.
