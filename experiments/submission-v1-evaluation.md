# Submission evaluation and validation

Primary model and decoder were fixed before the fresh synthetic audit. No supplied-match possession answers were read for submission inference, selection, or this evaluation.

## Fresh synthetic audit

10 cases: two previously unused seeds (450101, 450102), five fixed scenarios, 12,000 frames each. Seeds were checked and consumed in the ledger; reserved final-audit seeds and real match J03WPY remain untouched. No tuning followed this audit. These results do not establish real-match accuracy.

| Frozen system | Spell F1 | Macro F1 | Frame accuracy |
|---|---:|---:|---:|
| ensemble | 0.984322 | 0.995149 | 0.996121 |
| gbdt | 0.797352 | 0.940373 | 0.945065 |
| baseline | 0.583393 | 0.555574 | 0.772010 |

Runtime: 142.69 s. All 30 saved model/case scores were independently recomputed by the author and matched. Two seed clusters and short scenarios limit uncertainty estimates; this is not a cross-match confidence interval.

## External development data

The clean J03WN1 replay exactly reproduced the original ensemble: spell F1 0.793206, macro F1 0.952313, accuracy 0.965407. J03WN1 has been repeatedly inspected for development and is not an untouched evaluation match.

Confidence/ground-uncertainty testing used real external tracking and provider height, with a fixed simulated camera, gaps and sensor errors. There is no synchronized external CV video in this test. Parameters were fixed before scores.

The stress test hid 40% of observations in one-second gaps. It does not validate
long-gap recovery or calibrate the actual detector. In the supplied clip, most
otherwise eligible gap observations lack the required short endpoint support;
only one is admitted by the unchanged conservative policy.

| Observation condition | Tracking only | Naive CV fill | Confidence only | Full quality policy |
|---|---:|---:|---:|---:|
| clean | 0.793206 | 0.793206 | 0.793206 | 0.793206 |
| accurate_recovery | 0.588800 | 0.753165 | 0.753165 | 0.753165 |
| projection_known_height | 0.588800 | 0.412903 | 0.667102 | 0.654751 |
| projection_unknown_height | 0.588800 | 0.412903 | 0.667102 | 0.663677 |

Preset gate: passed. Runtime: 1060.11 s. All 48 saved scores, including four official baselines per selected observation input, were recomputed and matched.

Confidence filtering accounts for most of the gain in the projection-error test. The full airborne/ground-uncertainty policy is 0.01235 spell-F1 below confidence-only when height is known, and 0.00343 below it when height is unknown. This does **not** demonstrate an incremental F1 gain from airborne gating or a validated real-CV confidence estimator. No thresholds were loosened after seeing results.

## Supplied-match coverage and format only

- Full automatic tracking: 199,319 frames; 113,488 have a ball observation (56.94%).
- Completed CV clip: frames 11185–18684, 6,276 observed and 1,224 missing (16.32%).
- Conservative CV adapter: 6,615 baseline observations retained in the clip; one qualified CV gap fill; other unsupported points remain missing.
- The primary CSV has 74,465 H, 57,596 A and 67,258 D labels, 572 transitions and 473 possession spells.
- All 199,319 frame IDs are unique and ordered 0–199318. No blank cells, NaNs or unsupported label values. The official format validator passes.
- Original frozen active-period predictions reproduced exactly; only the declared inactive-period X-to-D rule changes the submission encoding.
- A fresh Python environment installed every pinned dependency and reproduced the primary CSV byte for byte.
- Full primary inference: 102.06 s; clean-environment reproduction: 135.63 s. Runs shared CPU resources, so these are execution receipts rather than controlled speed benchmarks.
- Packaged LightGBM reproduces its original predictions under the same inactive-period rule (50.11 s in the clean environment).
- The verified five-minute GBDT run took 1.68 s inference / 14.05 s including provenance checks.
- 55 focused original tests and three standalone repository tests passed. No independent coordinator review is claimed.

Primary CSV SHA-256: `bea82d693e2ee603da66305523252056f53a8e20c8e28fc9f935e1ef18141b20`.

No supplied-match accuracy is reported. Prediction agreement, ball coverage and passing format validation are not accuracy measurements. Aggregate detail is in `experiments/submission-results.json`; all frame-level artifacts remain private.

## Completed optional CV-supported run

The full-match CV variant admitted 1 additional point and changed 1 label compared with the backbone. It retains all 199,319 frames, with H/A/D counts {'H': 74464, 'A': 57597, 'D': 67258}. Runtime: 62.90s. Its SHA-256 is `c307a84f6352ec375ddacda54e7b9360421defaf730264bb08d1f6a9d14354a4`. This is prediction disagreement, not a measured accuracy gain or loss. The original CV and frozen model hashes remain unchanged.
