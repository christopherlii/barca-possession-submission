# Remontada — possession submission

Frame-by-frame H/A/D predictions from the provided automatic tactical tracking.
The submitted system is a frozen player temporal convolutional network (TCN), a
histogram gradient-boosted tree, and a sequence decoder. Inference runs on CPU
with NumPy and scikit-learn. The model weights are included in this repository.

## Reproduce the submitted CSV

Use Python 3.12. Place these provided files in `data/tactical/`:

- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_atd.xml`

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python run_submission.py --data-dir data --out outputs/submission
.venv/bin/python challenge/evaluation/validate_submission.py outputs/submission/remontada.csv
```

Use a new output directory for each run. `--data-dir` may point to another
folder containing `tactical/`. The output contains `remontada.csv`, probabilities,
observation-availability masks, and a receipt with source/model/input hashes.

Expected CSV SHA-256:
`bea82d693e2ee603da66305523252056f53a8e20c8e28fc9f935e1ef18141b20`

The CSV has exactly 199,319 rows, frames 0–199318, and H/A/D labels. Tactical
one-based frame IDs are converted once to zero-based video IDs. Missing ball
observations retain availability masks. Frames outside the supplied half
boundaries receive D; those frames are ungraded.

## Implementation

`run_submission.py` loads tracking, verifies the frozen model, runs inference,
and validates the output. `scripts/` contains only the feature extraction,
NumPy TCN forward pass, tree context, team symmetry, sequence decoder, tracking
loader and artifact checks required by this entry point. `models/ensemble/`
contains the frozen weights and training provenance. The official format
validator and its license are retained under `challenge/`.

See [remontada.md](remontada.md) for the technical document,
[EVALUATION_REPORT.md](EVALUATION_REPORT.md) for measured results and limitations,
and [SOURCE.md](SOURCE.md) for provenance. Licensed match files and generated
predictions remain in ignored folders and must not be committed.
