# Source provenance

This repository packages the inference implementation from the local `barca-hack` project, source revision `9728339b8c5311f9767f577c28d41c04720dcc65`, and a submission entry point with explicit input/output checks. Original project: https://github.com/christopherlii/barca-hack . The included files and this repository's own commit are authoritative; no checkout of another repository is needed for default inference.

The frozen TCN/tree and LightGBM parameters were trained on permitted synthetic and external matches, not the supplied match. `models/ensemble/receipt.json` records the original training revision and configuration. `models/gbdt/model.json` contains its frozen training/schema receipt. `SOURCE_MANIFEST.json` pins packaged code and models.

Upstream challenge code and documentation retain the MIT notice in `challenge/LICENSE`. Licensed match assets, private CV caches, targets and predictions are excluded. The experimental CV adapter uses uncalibrated observation scores and does not claim to estimate ball height.
