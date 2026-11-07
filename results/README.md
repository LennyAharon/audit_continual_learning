# Results index

Pre-computed outputs from the paper run. Table, figure and appendix numbers follow the paper. Commands that regenerate each item are in the [top-level README](../README.md).

| File / folder | Paper element |
|---|---|
| `merge_main_baseline.json`, `merge_main_naive.json`, `merge_main_task_arithmetic_lambda0.5.json`, `merge_main_ties_d0.5.json`, `merge_main_dare_d0.5.json`, `merge_main_magmax.json` | Table 1, Table 8 (main rows), Fig. 2 |
| `merge_main_task_arithmetic_lambda0.3.json`, `merge_main_ties_d0.3.json`, `merge_main_dare_d0.3.json` | Table 8 (hyperparameter sweep) |
| `baseline_llama31-8b.json` | Base-model characterisation; contains the base TruthfulQA-MC2 accuracy (0.531) used as the reference in Table 6 |
| `single_adapter_refs/` | Fig. 1, Table 6, App. B (one file per adapter) |
| `audit_pool.json` | Tables 2–3, App. A (candidate-pool audit: declared base, delta norm, verdict) |
| `audit_prevalence.json` | Curated candidate-pool audit summary per family |
| `mistral_audit.json` | Table 4 (Mistral cross-family audit) |
| `mistral_merging/` | Table 5 (audited 2-adapter Mistral merge and single-adapter references) |
| `negation_n200.json`, `negation_n200_history.json` | Fig. 3, Table 7 (negation coordinate search, n=200 probe) |
| `leakfree_block_probe.json`, `leakfree_importance_layers.json` | Fig. 4, App. C and Alg. 1 (per-block importance data) |
| `block_importance.csv`, `BLOCK_IMPORTANCE_ANALYSIS.md` | Derived from `leakfree_block_probe.json` by `analysis/analyze_block_importance.py` |
| `norm_threshold_calibration.json`, `NORM_THRESHOLD_CALIBRATION.md` | App. G, Table 10 (derived from `audit_pool.json` by `analysis/analyze_norm_threshold.py`) |
| `excluded_ablation/` | App. H, Table 11 |
| `audit_random_sample.json` | App. I, Table 12 (random-sample prevalence, N=40 per family) |
| `audit_scales.json` | App. I, Table 13 (cross-scale prevalence) |
| `audit_augmented/` | App. J, Table 14 (`llama_augmented.json`, `mistral_augmented.json`) |
| `seed_robustness/` | App. K (3 seeds, greedy) |
| `sampling_robustness/` | App. K, Table 15 (GSM8K SC@8, HumanEval pass@k) |
| `perstep_audited/` | App. L, Table 16 |
| `orthogonal_merger/` | App. M, Table 17 |

Folders with several runs include a `SUMMARY.json` that aggregates them.

Not committed in this release: `negation.json` (output of `src/experiments/negation_fulltest.py`, behind the "Negation finding" row of Table 8).
