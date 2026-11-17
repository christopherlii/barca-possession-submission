# Remontada — frame-by-frame possession

Reproducible offline H/A/D possession inference for More Than A Hack. The default
uses full-match automatic tactical tracking and the frozen player TCN/tree
ensemble. Licensed match assets and prediction files are deliberately excluded.

## Regenerate the submission

Use Python 3.12 and the original licensed tactical data bundle. No video decoding,
GPU service, training, network download, or answer key is needed for inference.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python run_submission.py --data-dir /path/to/licensed/data \
  --model ensemble --out outputs/submission
.venv/bin/python challenge/evaluation/validate_submission.py outputs/submission/remontada.csv
```

`/path/to/licensed/data/tactical/` must contain the original
`BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_atd.xml`. The LightGBM alternative
also uses `BAR-ATH_tactical_smart_tagging.xml`. All model parameters are included
under `models/`; their checksums are verified before prediction. On macOS,
LightGBM may additionally require the system OpenMP runtime (`brew install libomp`).

The output has exactly 199,319 ordered rows, frames 0–199318, with only H/A/D.
Inactive frames outside the supplied half bounds receive D. Missing tracking or
uncovered CV **inside** a half is not itself a dead-ball decision. The receipt
records input/model/source hashes, runtime and coverage. Probabilities and
availability masks are saved separately under the same ignored output directory.

Run the fixed LightGBM and reference baseline with separate new destinations:

```sh
.venv/bin/python run_submission.py --data-dir /path/to/licensed/data --model gbdt --out outputs/gbdt
.venv/bin/python run_submission.py --data-dir /path/to/licensed/data --model baseline --out outputs/baseline
```

The packaged baseline uses the official ±50-frame nearest-player majority vote,
plus the same explicit inactive-period rule used by this entry point.
To reproduce the unmodified official baseline as well:

```sh
MTAH_DATA_DIR=/path/to/licensed/data .venv/bin/python challenge/starter/baseline.py \
  --window 50 --out outputs/official-baseline
```

## Optional private CV input

The default submission is self-contained given the licensed tracking bundle.
The experimental CV-enhanced variant additionally requires the existing private
exhaustive CV cache and prepared adapter artifacts; these are **not** in a fresh
clone. This repository does not claim that the nondeterministic upstream SAM
recovery can regenerate the identical private cache from scratch.

To prepare confidence-qualified observations, place the original tactical bundle
under `data/` and provide the verified private files:

```sh
.venv/bin/python scripts/cv_confidence_adapter.py \
  --manifest /private/exhaustive-part2-input-v1/manifest.json \
  --features /private/cv-fresh-updated-20261117-v2/02-fused.npz \
  --cache /private/cv-fresh-updated-20261117-v2/12-final \
  --out outputs/qualified-cv
.venv/bin/python run_submission.py --data-dir data --model ensemble \
  --cv-manifest outputs/qualified-cv/manifest.json --out outputs/with-cv
```

This keeps every available tactical ball observation and admits only qualified
CV observations into gaps. It preserves original image evidence, source,
visibility, confidence, localization uncertainty and unknown airborne state.
Interpolation/display estimates are never treated as measurements. CV is
available only over its recorded 7,500-frame interval; the rest of the match
continues on automatic tactical tracking. See [the technical document](remontada.md).

For the original five-minute LightGBM adapter output:

```sh
.venv/bin/python scripts/possession_gbdt.py predict \
  --manifest /private/exhaustive-part2-input-v1/manifest.json \
  --model models/gbdt --out outputs/clip-gbdt
```

## Checks and evaluation

```sh
.venv/bin/python -m unittest discover -s tests -v
```

The included synthetic audit entry point checks a seed ledger before generating
data and consumes fresh held-out seeds after a successful run. It never changes
model parameters. Recorded audit seeds cannot be rerun as untouched evaluation;
choose genuinely fresh seeds for a new audit. Preserve reserved final-audit data.

```sh
.venv/bin/python evaluate_synthetic.py --ledger experiments/seed_ledger.json \
  --seeds FRESH_SEED_1 FRESH_SEED_2 --out outputs/new-synthetic-audit
```

See `EVALUATION_REPORT.md` for measured scores and their scope. Public supplied-
match answers were not used for this submission's training, tuning or selection.
The official format validator reads the prediction CSV alone.

## Form deliverables

- Upload the locally generated `remontada.csv`.
- Enter this public repository URL.
- Upload [remontada.md](remontada.md) as the short technical document.

Do not upload licensed data or frame-level predictions to GitHub. The original
challenge code retains its notice in [challenge/LICENSE](challenge/LICENSE).
