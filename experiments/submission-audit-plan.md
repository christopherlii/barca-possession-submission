# Locked submission audit

The submission uses the existing cycle-018 TCN/cycle-005 tree ensemble with frozen feature extraction and decoder. This choice is fixed before this audit. No training or model selection uses the audit results.

Evaluate exactly seeds 450101 and 450102 across the five existing default scenario configurations, 12,000 frames each. Compare the frozen ensemble, frozen 41-feature LightGBM and official smooth50 baseline. Report spell F1 first, macro F1 and frame accuracy. These seeds were checked against the current shared seed ledger and original model training/development seeds before use. The submission repository carries its own copied ledger; successful consumption will be merged back into the original ledger only after the running observation-quality experiment finishes its immutable-input checks.

Do not use or modify the reserved final real match or synthetic final-audit seeds. Do not rerun model search on these held-out results. This is synthetic evaluation, not supplied-match accuracy.
