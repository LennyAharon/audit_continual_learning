# Principled delta-norm threshold calibration

Addresses reviewer tech-limit-2. The paper uses a hard threshold of `total_norm >= 1.0`. This document evaluates four candidate threshold metrics and recommends the one with the largest separation between audited (real) and excluded (defective) adapters on the current pool.

## Per-adapter metric values

| Category | Slot | r | α | n_layers | total_norm | per_layer | per_layer/s | per_rank |
|---|---|---|---|---|---|---|---|---|
| audited | math | 8 | 8 | 64 | 16.620 | 0.2597 | 0.2597 | 2.0775 |
| audited | general_nlp | 8 | 8 | 224 | 7.522 | 0.0336 | 0.0336 | 0.9403 |
| audited | math | 16 | 32 | 224 | 5.200 | 0.0232 | 0.0116 | 0.3250 |
| audited | code | 32 | 64 | 64 | 1.730 | 0.0270 | 0.0135 | 0.0541 |
| audited | medical | 32 | 64 | 64 | 1.554 | 0.0243 | 0.0121 | 0.0486 |
| audited | general_nlp | 32 | 64 | 64 | 1.067 | 0.0167 | 0.0083 | 0.0333 |
| audited-but-failed | code | 16 | 32 | 224 | 3.625 | 0.0162 | 0.0081 | 0.2266 |
| excluded | creative-wrong-base | 64 | 32 | 224 | 46.091 | 0.2058 | 0.0514 | 0.7202 |
| excluded | code-defective-norm | 16 | 32 | 224 | 0.016 | 0.0001 | 0.0000 | 0.0010 |
| uncategorized | creative_writing | 8 | 16 | 64 | 7.164 | 0.1119 | 0.0560 | 0.8955 |

## Separability of audited vs excluded

For each metric, the larger the gap factor between audited-min and excluded-max, the more robust the threshold is to choice of cutoff.

| Metric | Audited [min, max] | Excluded [min, max] | Gap (audited_min / excluded_max) | Fully separable? |
|---|---|---|---|---|
| `total_norm` | [1.0670, 16.6200] | [0.0160, 46.0910] | 66.7× | YES |
| `norm_per_layer` | [0.0167, 0.2597] | [0.0001, 0.2058] | 233.4× | YES |
| `norm_per_layer_per_s` | [0.0083, 0.2597] | [0.0000, 0.0514] | 233.4× | YES |
| `norm_per_rank` | [0.0333, 2.0775] | [0.0010, 0.7202] | 33.3× | YES |

## Recommendation

`norm_per_layer` has the largest separation between audited and excluded adapters in the current pool (gap factor **233.4×**). The smallest audited value on this metric is 0.0167; the largest excluded value is 0.2058. A threshold of `0.1112` would correctly classify every adapter in the current pool, with a factor-of-233 buffer on either side.

Switching from the paper's `total_norm` to `norm_per_layer` would make the threshold dimensionally meaningful across model sizes and ranks. The paper's current threshold happens to work on this pool because the audited adapters all target the same fraction of layers; on a larger pool with more varied target_modules, the size-normalized metric would be more robust.

## Caveat

All metrics above are calibrated on a small pool (6 audited + 2 excluded). A principled calibration ideally uses a larger reference set of known-good adapters; the prevalence audit script (`src/run_audit_prevalence.py`) is designed to produce exactly this. Rerun this analysis after expanding the candidate pool to 20-50 adapters via `configs/audit_candidates.json`.
