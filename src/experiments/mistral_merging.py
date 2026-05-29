"""Cross-family merging experiments on Mistral-7B-Instruct-v0.3.

Uses only the 2 adapters that pass our audit (math, general_nlp).
The other 2 candidate adapters (code, creative_writing) declared
Mistral-7B-Instruct-v0.1 / v0.2 respectively and were excluded by the
base-model match check (see results/mistral_audit.json).

Configurations:
  baseline        : unmerged Mistral-7B-Instruct-v0.3
  single_math     : math adapter alone
  single_genmlp   : general_nlp adapter alone
  naive           : math + general_nlp summed
  ta_0.5          : task arithmetic, lambda=0.5
  ties_d0.5       : TIES, density=0.5
  dare_d0.5       : DARE, density=0.5

Benchmarks: GSM8K (full, strict-match) and HumanEval (full, pass@1).
Each config caches to results/mistral_merging/<label>.json so the
script is fully resumable.
"""
import gc
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge, task_arithmetic_merge, ties_merge, dare_merge

RESULTS_DIR = "results"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# Only the two adapters that passed the audit.
AUDITED_ADAPTERS = {
    "math":        "xummer/mistral-7b-gsm8k-lora-en",
    "general_nlp": "mkenfenheuer/Mistral-7B-Instruct-v0.3-ha-function-calling-lora",
}
ORDERING = ["math", "general_nlp"]

sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if "audit_continual_learning/src" not in p and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


def eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    return res["results"]["gsm8k"]["exact_match,strict-match"], res["results"]["gsm8k"]


def eval_humaneval(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    return r.get("pass@1,create_test", r.get("pass@1")), r


def run_config(label, apply_fn, benchmarks=("gsm8k", "humaneval")):
    """apply_fn(model) applies whatever merge / no-op the config requires.
    Returns existing dict (cached) or runs evals + saves.
    """
    out_path = os.path.join(RESULTS_DIR, f"mistral_merging/{label}.json")
    existing = load_results(out_path) if os.path.exists(out_path) else {}
    scores = existing.get("scores", {})
    fulls = existing.get("full", {})
    needed = [b for b in benchmarks if b not in scores]
    if not needed:
        logger.info(f"[{label}] all cached.")
        return existing

    logger.info(f"\n{'=' * 70}\n[{label}]  benchmarks needed: {needed}\n{'=' * 70}")
    model, tokenizer = load_base_model(MODEL_NAME)
    apply_fn(model)

    def _save():
        save_results({"method": label, "scores": scores, "full": fulls,
                      "model": MODEL_NAME, "audit_filter": "version_match"},
                     out_path)

    try:
        if "gsm8k" in needed:
            logger.info("  -> GSM8K (full) ...")
            s, r = eval_gsm8k(model, tokenizer)
            scores["gsm8k"] = s; fulls["gsm8k"] = r
            logger.info(f"     GSM8K: {s:.4f}")
            _save()
        if "humaneval" in needed:
            logger.info("  -> HumanEval (full) ...")
            s, r = eval_humaneval(model, tokenizer)
            scores["humaneval"] = s; fulls["humaneval"] = r
            logger.info(f"     HumanEval: {s:.4f}")
            _save()
    except Exception as e:
        logger.error(f"[{label}] eval failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()

    return load_results(out_path)


def apply_baseline(model):
    return  # no-op


def apply_single(task):
    def _apply(model):
        d = extract_lora_delta(None, AUDITED_ADAPTERS[task])
        naive_merge(model, d)
        del d; gc.collect()
    return _apply


def apply_naive(model):
    for t in ORDERING:
        d = extract_lora_delta(None, AUDITED_ADAPTERS[t])
        naive_merge(model, d)
        del d; gc.collect()


def apply_ta(lam):
    def _apply(model):
        for t in ORDERING:
            d = extract_lora_delta(None, AUDITED_ADAPTERS[t])
            task_arithmetic_merge(model, d, lambda_val=lam)
            del d; gc.collect()
    return _apply


def apply_ties(density):
    def _apply(model):
        for t in ORDERING:
            d = extract_lora_delta(None, AUDITED_ADAPTERS[t])
            ties_merge(model, d, density=density)
            del d; gc.collect()
    return _apply


def apply_dare(density):
    def _apply(model):
        for t in ORDERING:
            d = extract_lora_delta(None, AUDITED_ADAPTERS[t])
            dare_merge(model, d, density=density)
            del d; gc.collect()
    return _apply


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("mistral_merging/merging", RESULTS_DIR)
    logger.info(f"Mistral 2-adapter merging on {MODEL_NAME}")
    logger.info(f"Audited adapters: {AUDITED_ADAPTERS}")

    # GSM8K-first ordering: do all configs on GSM8K before any HumanEval,
    # so partial completion still gives a coherent table.
    configs = [
        ("baseline",     apply_baseline),
        ("single_math", apply_single("math")),
        ("single_genmlp", apply_single("general_nlp")),
        ("naive",        apply_naive),
        ("ta_0.5",       apply_ta(0.5)),
        ("ties_d0.5",    apply_ties(0.5)),
        ("dare_d0.5",    apply_dare(0.5)),
    ]

    # Pass 1: GSM8K only.
    for label, fn in configs:
        run_config(label, fn, benchmarks=("gsm8k",))

    # Pass 2: HumanEval (resume-friendly).
    for label, fn in configs:
        run_config(label, fn, benchmarks=("gsm8k", "humaneval"))

    logger.info("\n=== SUMMARY ===")
    for label, _ in configs:
        p = os.path.join(RESULTS_DIR, f"mistral_merging/{label}.json")
        if os.path.exists(p):
            d = load_results(p)
            sc = d.get("scores", {})
            line = f"  {label:18s}"
            for b in ("gsm8k", "humaneval"):
                if b in sc:
                    line += f"  {b}={sc[b]:.4f}"
                else:
                    line += f"  {b}=---"
            logger.info(line)


if __name__ == "__main__":
    main()
