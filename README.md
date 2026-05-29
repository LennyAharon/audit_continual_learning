# Audit Before You Merge

Code for **"Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition"** (ICML 2026 Workshop).

> Post-hoc merging of public LoRA adapters is bottlenecked less on new merging methods than on the measurement infrastructure needed to evaluate them. This repo contains the audit, leakage-free probing, and merge-method comparisons that produce every number and figure in the paper.

📄 [Paper PDF (OpenReview)](#) · 📊 [Adapter pool](configs/adapter_pool.json)

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

Every script writes a JSON to `results/`. Run them in any order; `analysis/generate_figures.py` reads from `results/` and writes to `figures/`.

| Paper element | Command | Output |
|---|---|---|
| **Table 1, Fig. 1 (methods)** — 4-adapter merge | `python -m src.experiments.merge_main` | `results/merge_main_*.json` |
| **Fig. 1 (upper bounds), Table 5** — single-adapter refs | `python -m src.experiments.single_adapter_refs` | `results/single_adapter_refs.json` |
| **Table 2** — audit on the 6-adapter pool | `python -m src.experiments.audit_pool` | `results/audit_pool.json` |
| **Table 3, App. F** — random-sample prevalence | `python -m src.experiments.prevalence_random` | `results/audit_random_sample.json` |
| **Table 4, App. F** — cross-scale prevalence | `python -m src.experiments.prevalence_scales` | `results/audit_scales.json` |
| **Fig. 2, App. C** — leakage-free block importance | `python -m src.experiments.leakfree_probe` | `results/leakfree_importance_layers.json` |
| **Fig. 3, Table 7** — negation coordinate search | `python -m src.experiments.negation_search --n 200` | `results/negation_n200.json` + `_history.json` |
| **App. C** — negation full-test eval | `python -m src.experiments.negation_fulltest` | `results/negation_fulltest.json` |
| **App. G** — audit threshold ablation | `python analysis/analyze_norm_threshold.py` | `results/norm_threshold_calibration.json` |
| **App. H** — excluded-adapter ablation | `python -m src.experiments.excluded_ablation` | `results/excluded_ablation/*.json` |
| **App. I** — augmented audit signals | `python -m src.experiments.audit_augmented` | `results/audit_augmented.json` |
| **App. J** — seed robustness (3 seeds) | `python -m src.experiments.seed_robustness` | `results/robustness/seed_*.json` |
| **App. J** — sampling robustness (SC@8) | `python -m src.experiments.sampling_robustness` | `results/robustness/sampling_*.json` |
| **App. J** — HumanEval pass@k | `python -m src.experiments.humaneval_passk` | `results/robustness/passk_*.json` |
| **App. K** — per-step forgetting curves | `python -m src.experiments.perstep_audited` | `results/perstep_audited/*.json` |
| **App. L** — orthogonal-projection merger | `python -m src.experiments.orthogonal_merger` | `results/orthogonal_merger/*.json` |
| **Cross-family** — Mistral audit | `python -m src.experiments.mistral_audit` | `results/mistral_audit.json` |
| **Cross-family** — Mistral merging | `python -m src.experiments.mistral_merging` | `results/mistral_merging/*.json` |
| **Figures** | `python analysis/generate_figures.py` | `figures/fig*.pdf` |

Pre-computed `results/` JSONs from the paper run are committed, so figures regenerate without re-running the experiments.

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
  merge_orthogonal.py        # orthogonal-projection merger (App. L)
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
results/                     # pre-computed paper data
figures/                     # paper figures
```

---

## Citation

```bibtex
@inproceedings{aharon2026audit,
  title     = {Audit Before You Merge: Provenance, Probing, and Continual {LoRA} Composition},
  author    = {Aharon, Lenny and Glazer, Neta and Aharon, Lior},
  booktitle = {ICML 2026 Workshop},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
