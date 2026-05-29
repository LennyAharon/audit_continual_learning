"""Per-step full-test GSM8K trajectories with leakage-free importance.
Generates data for paper figure 2 (forgetting curves, leakage-free version).

Runs sequential merging over ordering [code, math, general_nlp, medical]
and evaluates full GSM8K test after each merge step for:
  - naive (alpha=0)
  - GF-EWC alpha=1
  - GF-EWC alpha=6 (paper headline)

Each configuration is cached individually.
"""
import gc, os, sys, json
import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
sys.path.insert(0, 'src')

from utils import load_config, load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from gf_ewc import gf_ewc_merge
from merge import naive_merge

RESULTS_DIR = "results"

import logging
logger = logging.getLogger(__name__)

sys.modules.pop('evaluate', None)
sys.path = [p for p in sys.path if 'audit_continual_learning/src' not in p and p != 'src']
import lm_eval
from lm_eval.models.huggingface import HFLM


def eval_full_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    return res["results"]["gsm8k"]["exact_match,strict-match"]


def run_trajectory(config, importance, label, alpha, use_gfewc):
    """Merge adapters sequentially, eval full GSM8K after each step. Per-step cache."""
    model_name = config["base_models"][0]["name"]
    short_name = config["base_models"][0]["short_name"]
    ordering = config["merge_orderings"][0]
    adapters = config["lora_adapters"][short_name]
    adapter_by_task = {a["task"]: a for a in adapters}

    # Check which steps already cached
    per_step_path = os.path.join(RESULTS_DIR, f"perstep_leakfree_{label}.json")
    existing = {}
    if os.path.exists(per_step_path):
        existing = load_results(per_step_path)
        if len(existing.get("steps", [])) == len(ordering):
            logger.info(f"{label}: all steps cached")
            return existing

    logger.info(f"\n{'='*60}\n{label} (α={alpha}, gfewc={use_gfewc})\n{'='*60}")
    model, tokenizer = load_base_model(model_name)

    steps = existing.get("steps", [])
    start_idx = len(steps)

    # Replay cached merges without evaluating
    for i in range(start_idx):
        task_name = ordering[i]
        logger.info(f"  replaying cached step {i}: +{task_name}")
        delta = extract_lora_delta(model, adapter_by_task[task_name]["id"])
        if use_gfewc:
            gf_ewc_merge(model, delta, importance, alpha=alpha, scaling_fn="linear")
        else:
            naive_merge(model, delta)
        del delta; gc.collect()

    for i in range(start_idx, len(ordering)):
        task_name = ordering[i]
        logger.info(f"\n  step {i}: +{task_name}")
        delta = extract_lora_delta(model, adapter_by_task[task_name]["id"])
        if use_gfewc:
            gf_ewc_merge(model, delta, importance, alpha=alpha, scaling_fn="linear")
        else:
            naive_merge(model, delta)
        del delta; gc.collect()

        gsm = eval_full_gsm8k(model, tokenizer)
        logger.info(f"    GSM8K full: {gsm:.4f}")
        steps.append({"step": i, "adapter": task_name, "gsm8k_full": gsm})

        save_results({"label": label, "alpha": alpha, "use_gfewc": use_gfewc,
                      "ordering": ordering, "steps": steps}, per_step_path)

    del model; torch.cuda.empty_cache(); gc.collect()
    return load_results(per_step_path)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("perstep_leakfree", RESULTS_DIR)

    config = load_config("configs/experiment_config.yaml")
    importance = load_results("results/leakfree_importance_layers.json")

    configs = [
        ("naive", 0.0, False),
        ("gfewc_alpha1", 1.0, True),
        ("gfewc_alpha6", 6.0, True),
    ]

    for label, alpha, use_gfewc in configs:
        run_trajectory(config, importance, label, alpha, use_gfewc)

    logger.info("\n\n=== SUMMARY ===")
    for label, alpha, _ in configs:
        path = os.path.join(RESULTS_DIR, f"perstep_leakfree_{label}.json")
        if os.path.exists(path):
            d = load_results(path)
            vals = [f"{s['gsm8k_full']:.3f}" for s in d["steps"]]
            logger.info(f"  {label}: {vals}")


if __name__ == "__main__":
    main()
