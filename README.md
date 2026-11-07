# Audit Before You Merge

Code for **"Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition"** (Continual Adaptation at Scale: Towards Sustainable AI Workshop at ICML 2026).

> Post-hoc merging of public LoRA adapters is bottlenecked less on new merging methods than on the measurement infrastructure needed to evaluate them. This repo contains the audit, leakage-free probing, and merge-method comparisons from the paper, together with the pre-computed result files behind its tables and figures.

📊 [Adapter pool](configs/adapter_pool.json) · 🗂️ [Which result file feeds which table](results/README.md)

---

## Install

```bash
git clone https://github.com/LennyAharon/audit_continual_learning.git
cd audit_continual_learning
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
pip install -r requirements.txt
```

Single NVIDIA A6000 (49 GB) is enough for every experiment; bf16, no quantisation.

---

## Reproducing the paper

Run all commands from the repo root. Most scripts write a JSON to `results/` and resume from cached cells if interrupted. `analysis/generate_figures.py` reads from `results/` and writes to `figures/`. Table, figure and appendix numbers follow the paper.

| Paper element | Command | Output |
|---|---|---|
| **Table 1, Table 8, Fig. 2** — 4-adapter merge (and the λ=0.3 / d=0.3 sweep) | `python -m src.experiments.merge_main --method {baseline,naive,magmax}`<br>`... --method task_arithmetic --lambda-val 0.5`<br>`... --method ties --density 0.5`<br>`... --method dare --density 0.5` | `results/merge_main_*.json` |
| **Fig. 1, Table 6, App. B** — single-adapter references | `python -m src.experiments.single_adapter_refs` | `results/single_adapter_<task>.json` (committed under `results/single_adapter_refs/`) |
| **Tables 2–3, App. A** — audit of the candidate pool | `python -m src.experiments.audit_pool` | `results/audit_pool.json` |
| **Table 4, App. A** — Mistral audit | `python -m src.experiments.mistral_audit` | prints the audit table; the committed `results/mistral_audit.json` is the saved paper-run output |
| **Table 5, App. A** — Mistral merging | `python -m src.experiments.mistral_merging` | `results/mistral_merging/*.json` |
| **Fig. 3, Table 7, App. C** — negation coordinate search | `python -m src.experiments.negation_search` | `results/negation_n200.json`, `results/negation_n200_history.json` |
| **App. G, Table 10** — audit threshold ablation (CPU only) | `python analysis/analyze_norm_threshold.py` | `results/norm_threshold_calibration.json`, `results/NORM_THRESHOLD_CALIBRATION.md` |
| **App. H, Table 11** — excluded-adapter ablation | `python -m src.experiments.excluded_ablation` | `results/excluded_ablation/*.json` |
| **App. I, Table 12** — random-sample prevalence (Llama-3.1, Mistral) | `python -m src.experiments.prevalence_random` | `results/audit_random_sample.json` |
| **App. I, Table 13** — cross-scale prevalence | `python -m src.experiments.prevalence_scales` | `results/audit_scales.json` |
| **App. J, Table 14** — augmented audit signals | `python -m src.experiments.audit_augmented` | `results/audit_augmented/{llama,mistral}_augmented.json` |
| **App. K** — seed robustness | `python -m src.experiments.seed_robustness` | `results/seed_robustness/*.json` |
| **App. K, Table 15** — sampling robustness (SC@8, HumanEval pass@k) | `python -m src.experiments.sampling_robustness` | `results/sampling_robustness/*.json` |
| **App. L, Table 16** — per-step forgetting curves | `python -m src.experiments.perstep_audited` | `results/perstep_audited/*.json` |
| **App. M, Table 17** — orthogonal-projection merger | `python -m src.experiments.orthogonal_merger` | `results/orthogonal_merger/*.json` |
| **Fig. 4, App. C, Alg. 1** — block-importance data | pre-computed (see notes) | `results/leakfree_block_probe.json`, `results/leakfree_importance_layers.json`; `python analysis/analyze_block_importance.py` writes `results/block_importance.csv` and `results/BLOCK_IMPORTANCE_ANALYSIS.md` |
| **Figures 1–4** | `python analysis/generate_figures.py` | `figures/fig1_upper_bounds` (Fig. 1), `fig1_methods` (Fig. 2), `fig3_negation` (Fig. 3), `fig2_block_importance` (Fig. 4) |

`analysis/generate_figures.py` also writes `figures/fig3_adapter_audit`, which is not used in the paper.

Pre-computed `results/` JSONs from the paper run are committed, so figures regenerate without re-running the experiments.

### Notes on this release

A few items differ from a fully self-contained pipeline:

- **Block-importance probe.** `results/leakfree_block_probe.json` and `results/leakfree_importance_layers.json` are committed as data. The script that produced them is not part of this release; `analysis/` and `generate_figures.py` only read them.
- **Negation full-test row (Table 8).** `src/experiments/negation_fulltest.py` writes `results/negation.json`. That output file is not committed here.
- **Random-sample prevalence.** `prevalence_random.py` reads `configs/audit_candidates.json` to build its skip list. That file is not included; `results/audit_random_sample.json` is the saved paper-run output.
- **Earlier-draft scripts, not used for any table or figure in the paper.** `src/experiments/humaneval_passk.py` (imports a `gf_ewc` module that is not included) and `src/experiments/leakfree_probe.py`. The paper's HumanEval pass@k results come from `sampling_robustness.py`, and its per-step results (Table 16) from `perstep_audited.py`.

---

## Adapter pool

`configs/adapter_pool.json` lists every HuggingFace identifier we tested, with a `paper-audited` / `paper-excluded` / `extra` label per slot. The six in the paper:

| Slot | HF identifier | Verdict |
|---|---|---|
| math | `kai-xu/Llama-3.1-8B-Instruct-MATH-Finetuned-LoRA` | audited |
| code | `yangao381/FlowerTune-Code-Llama-3.1-8B-Instruct-PEFT` | audited |
| general_nlp | `zjudai/flowertune-general-nlp-lora-llama-3.1-8b-instruct` | audited |
| medical | `yangao381/FlowerTune-Medical-Llama-3.1-8B-Instruct-PEFT` | audited |
| no-op code | `TianJun1/llama3.1-8b-code-reflector-lora` | excluded (norm 0.02) |
| wrong-base | `Blackroot/Llama-3-8B-Abomination-LORA` | excluded (Llama-3, not 3.1) |

SHA256 prefixes pinning the exact checkpoints we audited are in Table 3 of the paper.

---

## Project layout

```
src/
  audit_checks.py            # base-model match + reconstructed delta norm
  merge.py                   # naive, TA, TIES, DARE, MagMax
  merge_orthogonal.py        # orthogonal-projection merger (App. M)
  evaluate.py                # lm-eval-harness wrapper
  utils.py                   # model loading, LoRA delta extraction
  experiments/               # one script per paper section
configs/
  adapter_pool.json
  lm_eval_tasks/humaneval_10.yaml   # pass@k task variant
analysis/
  generate_figures.py
  analyze_block_importance.py
  analyze_norm_threshold.py
results/                     # pre-computed paper data (index in results/README.md)
figures/                     # paper figures
```

---

## Citation

```bibtex
@inproceedings{aharonaudit,
  title={Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition},
  author={Aharon, Lenny and Glazer, Neta and Aharon, Lior},
  booktitle={Continual Adaptation at Scale: Towards Sustainable AI Workshop},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
