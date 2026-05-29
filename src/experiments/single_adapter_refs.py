"""Phase 2: Single-adapter upper bounds.

For each of the 4 adapters in the final config, apply ONLY that adapter
(no merging, no others) and eval on its target benchmarks. The numbers
produced here establish "what is the most this adapter can do" — i.e. the
ceiling that any merge strategy is working against.

Benchmarks per adapter:
  math         -> MATH-500 + GSM8K
  code         -> HumanEval (single-sample pass@1)
  general_nlp  -> TruthfulQA MC2 + GSM8K
  medical      -> GSM8K (damage check)

Each result is saved to results/single_adapter_{task}.json and skipped if
already present, so this script is resumable.
"""
import gc
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_config, load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge

RESULTS_DIR = "results"

sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if "audit_continual_learning/src" not in p and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


def eval_math500(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    res = lm_eval.simple_evaluate(model=lm, tasks=["hendrycks_math500"], num_fewshot=4)
    r = res["results"]["hendrycks_math500"]
    score = r.get("exact_match,none", r.get("acc,none", r.get("exact_match", r.get("acc"))))
    return score, r


def eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    return res["results"]["gsm8k"]["exact_match,strict-match"]


def eval_humaneval(model, tokenizer):
    """HumanEval pass@1, single sample at temp=0. Fast upper-bound check."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=["humaneval"],
        confirm_run_unsafe_code=True,
    )
    return res["results"]["humaneval"]


def eval_truthfulqa(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["truthfulqa_mc2"])
    r = res["results"]["truthfulqa_mc2"]
    return r.get("acc,none", r.get("acc")), r


BENCHMARKS_PER_TASK = {
    "math": ["math500", "gsm8k"],
    "code": ["humaneval"],
    "general_nlp": ["truthfulqa_mc2", "gsm8k"],
    "medical": ["gsm8k"],
}


def run_for_adapter(config, task_name):
    out_path = os.path.join(RESULTS_DIR, f"single_adapter_{task_name}.json")
    if os.path.exists(out_path):
        existing = load_results(out_path)
        needed = set(BENCHMARKS_PER_TASK[task_name])
        have = set(existing.get("scores", {}).keys())
        if needed.issubset(have):
            logger.info(f"[{task_name}] all benchmarks cached — skipping")
            return existing

    model_name = config["base_models"][0]["name"]
    short_name = config["base_models"][0]["short_name"]
    adapters = config["lora_adapters"][short_name]
    adapter_by_task = {a["task"]: a for a in adapters}
    adapter_id = adapter_by_task[task_name]["id"]

    logger.info(f"\n{'=' * 70}\n[{task_name}]  adapter: {adapter_id}\n{'=' * 70}")

    model, tokenizer = load_base_model(model_name)
    delta = extract_lora_delta(None, adapter_id)
    naive_merge(model, delta)
    del delta; gc.collect()

    scores = {}
    full = {}
    for bench in BENCHMARKS_PER_TASK[task_name]:
        logger.info(f"  -> evaluating {bench} ...")
        if bench == "math500":
            score, fr = eval_math500(model, tokenizer)
            scores["math500"] = score
            full["math500"] = fr
        elif bench == "gsm8k":
            scores["gsm8k"] = eval_gsm8k(model, tokenizer)
        elif bench == "humaneval":
            full["humaneval"] = eval_humaneval(model, tokenizer)
            he = full["humaneval"]
            scores["humaneval_pass1"] = he.get("pass@1,create_test", he.get("pass@1", he.get("pass_at_1")))
        elif bench == "truthfulqa_mc2":
            score, fr = eval_truthfulqa(model, tokenizer)
            scores["truthfulqa_mc2"] = score
            full["truthfulqa_mc2"] = fr
        logger.info(f"     {bench}: {scores.get(bench, scores.get(bench + '_pass1'))}")

    del model; torch.cuda.empty_cache(); gc.collect()

    out = {
        "task": task_name,
        "adapter_id": adapter_id,
        "scores": scores,
        "full": full,
    }
    save_results(out, out_path)
    return out


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("single_adapter_upper_bounds", RESULTS_DIR)
    config = load_config("configs/experiment_config.yaml")

    # Order: cheapest first so we fail fast if the new adapters are broken.
    order = ["medical", "general_nlp", "code", "math"]
    all_results = {}
    for task in order:
        all_results[task] = run_for_adapter(config, task)

    logger.info("\n" + "=" * 70)
    logger.info("SINGLE-ADAPTER UPPER BOUNDS (all 4 adapters, alone)")
    logger.info("=" * 70)
    for task, r in all_results.items():
        logger.info(f"[{task}]  {r['adapter_id']}")
        for k, v in r["scores"].items():
            if v is None:
                logger.info(f"    {k}: (missing)")
            else:
                logger.info(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
