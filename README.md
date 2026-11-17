# Remontada — CV-supported possession submission v2

Full-match H/A/D inference using the frozen player TCN/tree ensemble, automatic
tactical tracking as the backbone, and confidence-qualified CV gap recovery.
No model retraining, supplied-match answer key, cloud GPU or full-video CV rerun
is required for Part 2 inference.

## Required private inputs

The CV-supported submission uses the original licensed tactical bundle. The
**verified replay route also requires our completed private CV artifacts**.
[CV_REBUILD.md](CV_REBUILD.md) now supplies the upstream source, checkpoint pins
and raw-video rebuild command. Fresh raw-video byte equality has not yet been
verified. Do not publish the cache or predictions to GitHub. The tracking-only
fallback remains available explicitly.

Place the licensed bundle under `data/tactical/`, including the original ATD text,
metadata, homography and automatic Smart Tagging. Provide these existing files:

- `exhaustive-part2-input-v1/manifest.json` and its observation/tagging sidecars
- `cv-fresh-updated-20261117-v2/02-fused.npz`
- `cv-fresh-updated-20261117-v2/12-final/detections.jsonl` and `config.json`

## Regenerate the CV-supported submission

Use Python 3.12. The frozen model files are included under `models/`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/cv_confidence_adapter.py \
  --manifest /private/exhaustive-part2-input-v1/manifest.json \
  --features /private/cv-fresh-updated-20261117-v2/02-fused.npz \
  --cache /private/cv-fresh-updated-20261117-v2/12-final \
  --data-dir data --unknown-airborne-policy allow_uncertain \
  --out outputs/qualified-cv-v2
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python run_submission.py --data-dir data --model ensemble \
  --cv-manifest outputs/qualified-cv-v2/manifest.json --out outputs/submission-v2
.venv/bin/python challenge/evaluation/validate_submission.py outputs/submission-v2/remontada.csv
```

If the qualified private adapter bundle is already provided, replay directly
with `--cv-manifest /private/qualified-cv-v2/manifest.json`; keep its observation,
quality and restart sidecars together. No upstream CV rerun is needed for this
cache-based replay. The raw-video-only reproducibility limitation still applies.

Use new output directories; existing artifacts are never overwritten. On macOS,
LightGBM may need the system OpenMP runtime (`brew install libomp`). The output
has 199,319 ordered rows, frames 0–199318, using H/A/D only. Receipt hashes,
probabilities and observation/uncertainty masks are saved alongside the CSV.

## What changed in v2

The old policy required short tracking anchors whenever ball height was unknown.
That defeated recovery in long tracking gaps. V2 keeps unknown height explicitly
unknown, but allows an accepted CV observation to fill a gap if its confidence
and localization uncertainty pass the unchanged checks. Known airborne warnings,
low/missing confidence, invalid projection and upstream rejections still block a
fill. Display interpolation is never an observation. Existing tactical positions
remain unchanged; uncovered CV frames continue to use tactical tracking.

The submission command refuses to silently fall back to tracking-only and checks
the exact v2 policy configuration. See [the technical document](remontada.md) and
[measured evaluation results](EVALUATION_REPORT.md). Real CV confidence is still
uncalibrated; external stress tests do not establish supplied-match accuracy.

## Preserved alternatives

```sh
# Explicit tracking-only fallback; reproduces the v1 primary labels.
.venv/bin/python run_submission.py --data-dir data --model ensemble \
  --backbone-only --out outputs/backbone
# Frozen LightGBM alternative using automatic tactical tagging.
.venv/bin/python run_submission.py --data-dir data --model gbdt --out outputs/gbdt
# Unmodified official nearest-player baseline, no scoring.
MTAH_DATA_DIR=data .venv/bin/python challenge/starter/baseline.py \
  --window 50 --out outputs/official-baseline
# Original five-minute LightGBM prediction from the existing CV adapter.
.venv/bin/python scripts/possession_gbdt.py predict \
  --manifest /private/exhaustive-part2-input-v1/manifest.json \
  --model models/gbdt --out outputs/clip-gbdt
```

The original v1 repository is preserved at tag `submission-v1`. The old upload
file is not overwritten by this version.

## Validation and evaluation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

`evaluate_cv_policy.py` compares the locked v1/v2 policies on fresh synthetic
matches after a passing external-development result. It checks and consumes the
seed ledger. Do not reuse recorded held-out seeds as fresh tests or retune on
those results. Reserved final-audit data remain untouched. See the committed
experiment plans and aggregate reports; all frame-level artifacts stay private.

## Form fields

Upload the newly generated `remontada.csv` and [remontada.md](remontada.md), and
enter this public repository URL. The public repository contains code, model
parameters and aggregate diagnostics—not licensed match data or predictions.
Upstream challenge code retains its notice in [challenge/LICENSE](challenge/LICENSE).


The upstream CV source snapshot and guarded rebuild/replay commands are now provided in [CV_REBUILD.md](CV_REBUILD.md). This closes the missing-code packaging gap; a full raw-video byte-equality run remains unverified.
