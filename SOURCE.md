# Source provenance

The submitted tracking model originates in the [barca-hack project](https://github.com/christopherlii/barca-hack).
Its packaged inference source was taken from revision
`cd029e3bdf2b97b8f824389d4968d4726761571f`. This repository retains the functions,
constants and imports required by the frozen forward pass; unrelated entry
points and experiment code are removed. Numerical inference and model weights
are preserved, and the cleaned entry point is checked against the submitted CSV.

`models/ensemble/receipt.json` records the training revision, external games,
seeds, architecture and frozen model hashes. `SOURCE_MANIFEST.json` records the
current packaged source and model hashes. No checkout of another repository is
needed for inference.

The official challenge files retain their MIT notice in `challenge/LICENSE`.
Licensed match data and generated frame-level outputs are not distributed.
