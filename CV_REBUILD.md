# Rebuilding and checking the v2 CV input

`rebuild_v2_cv.py` packages the previously missing upstream route. It includes
the 23 recorded CV commands and a frozen source snapshot with local Python
dependencies. It never reads possession answer keys or trains a model.

**Status:** source/checkpoint preflight is tested. Historical-cache replay is
tested separately. A complete fresh raw-video rebuild has not been verified;
the original 7,500-frame run took 68.59 minutes on Apple MPS. Therefore adding
this route does not, by itself, establish the competition's exact-reproduction
requirement. The script refuses to report success if its final CSV differs
from v2's recorded hash. It preserves the mismatching output for diagnosis.

The source snapshot comes from the preserved exhaustive-CV worktree. Several
stage implementations were changed between launch and execution in the
historical run: the launch manifest alone was not a complete source snapshot.
The recovered stage versions are now frozen in `cv_rebuild/recipe.json`.
Package versions below describe the observed local runtime; historical
wheel-level identity and cross-device equality were not recorded or proven.
No claim of proven SAM nondeterminism is necessary: exact raw-video replay is
simply unverified. Do not infer it from a cache-only replay.

## Setup

Use Python 3.12 and the normal repository `.venv`. The extra requirements are
isolated from the possession-only installation:

```sh
.venv/bin/python -m pip install -r requirements-cv-rebuild.txt
mkdir -p outputs/cv-assets/two-stage-cv-models outputs/cv-assets/sam2-models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt -o outputs/cv-assets/two-stage-cv-models/yolo11x.pt
curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o outputs/cv-assets/sam2-models/sam2.1_hiera_tiny.pt
git clone https://github.com/facebookresearch/sam2 outputs/cv-assets/sam2-source
git -C outputs/cv-assets/sam2-source checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

The script checks both model SHA-256 values and the SAM source revision before
running. Public pretrained model/source licenses apply separately. Licensed
match data, caches, and all prediction files remain under ignored paths.

## Check and rebuild from licensed data

```sh
.venv/bin/python rebuild_v2_cv.py --check --data-dir /path/to/licensed/data --out outputs/cv-preflight
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python rebuild_v2_cv.py --rebuild --data-dir /path/to/licensed/data --device mps --out outputs/v2-from-video
```

The new private output contains an isolated engine workspace, per-stage logs,
raw observed-only adapter inputs, the confidence-qualified adapter, and
`prediction/remontada.csv`. The engine links the licensed data and model files;
it does not modify them or an existing CV job. No viewer is launched. Each run
requires a new output directory. CPU/CUDA are accepted for portability, but
their equality with the historical MPS result is not established.

An existing asset layout may be supplied with `--assets-dir /private/outputs`.
For the historical local split dependency installation, also pass
`--runtime-dir /private/outputs`; this expects `two-stage-cv-runtime/` and
`sam2-runtime/`. Normally install the pinned requirements instead.

## Replay an existing original CV cache

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python rebuild_v2_cv.py --replay-cache /private/cv-fresh-updated-20261117-v2 \
  --data-dir /path/to/licensed/data --out outputs/v2-cache-check
```

This verifies the original fused features, detections and configuration hashes,
regenerates both adapters, and reruns the frozen TCN/tree. It requires the raw
CV cache, not a supplied possession prediction or pre-qualified adapter.
`rebuild-receipt.json` distinguishes cache replay from raw-video verification.
The expected v2 CSV hash is
`8f18e58701252edf3253ba8b977d83f16fc111b2a7f61faac3912a5a3c214f17`.

## Actual coverage

v2 predicts all 199,319 tactical video frames, but its additional CV covers
only frames 11185–18684 (7,500 frames). Tactical tracking is the full-match
backbone. The CV adapter supplies 265 accepted gap observations and never
replaces an available tactical ball observation. Uncovered CV frames are not
classified as dead ball merely because CV is unavailable.

The separate full-game CV job is not part of this frozen submission. Its
intermediate evidence sweep must not be presented as completed SAM recovery
or silently added to v2. Any later integration needs its own provenance,
source/visibility checks and prediction receipt.

## Verification receipt

The settled-code cache → adapter → confidence filter → frozen ensemble run
reproduced the v2 CSV byte for byte in 107.88 seconds (87.02 seconds for
full-match possession inference). Official format validation passed for all
199,319 frames; probability/class-order/alignment checks passed; 265 CV fills
were used and the three source CV hashes were unchanged. Nine unit tests passed.
All 21 available effective-stage hashes match the recovered source snapshot.
See [validation.json](cv_rebuild/validation.json) for aggregate checks. These
checks do not verify a fresh raw-video rebuild or measure prediction accuracy.
