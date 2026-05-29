"""Full-test evaluation of the negation-finding configuration.

Addresses reviewer concern: "Can you evaluate the probe-optimal negation
configuration on the full GSM8K and HumanEval sets?"

Configuration (chosen by per-adapter coordinate search w/ negation on the
GSM8K probe):
    code = +1.0,  math = -0.5,  general_nlp = +1.0,  medical = +1.0

We already have MATH-500 = 0.140 for this config from the original
neg_per_adapter run. This script fills in GSM8K and HumanEval on the full
test sets and writes them to results/negation.json
(merging into the existing file).
"""
import gc
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_config, load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge, task_arithmetic_merge

RESULTS_DIR = "results"

sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if "audit_continual_learning/src" not in p and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


CHOSEN_LAMBDAS = {
    "code":        1.0,
    "math":       -0.5,   # the negation
    "general_nlp": 1.0,
    "medical":     1.0,
}


def eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    return res["results"]["gsm8k"]["exact_match,strict-match"], res["results"]["gsm8k"]


def eval_humaneval(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    return r.get("pass@1,create_test", r.get("pass@1")), r


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("negation_fulltest", RESULTS_DIR)
    config = load_config("configs/experiment_config.yaml")

    model_name = config["base_models"][0]["name"]
    short_name = config["base_models"][0]["short_name"]
    ordering = config["merge_orderings"][0]
    adapters = config["lora_adapters"][short_name]
    adapter_by_task = {a["task"]: a for a in adapters}

    out_path = os.path.join(RESULTS_DIR, "negation.json")
    existing = load_results(out_path) if os.path.exists(out_path) else {
        "method": "neg_per_adapter", "scores": {}, "full": {}
    }
    scores = existing.get("scores", {})
    fulls = existing.get("full", {})
    needed = [b for b in ["gsm8k", "humaneval"] if b not in scores]
    if not needed:
        logger.info("All benchmarks cached for negation full-test; nothing to do.")
        return

    logger.info(f"\n{'=' * 70}\nNegation full-test  λ={CHOSEN_LAMBDAS}  benchmarks={needed}\n{'=' * 70}")

    try:
        model, tokenizer = load_base_model(model_name)
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM at model load: {e}")
        logger.error("GPU is too contended to run this experiment now.")
        return

    # Apply each adapter at its chosen λ.
    for t in ordering:
        d = extract_lora_delta(None, adapter_by_task[t]["id"])
        lam = CHOSEN_LAMBDAS[t]
        logger.info(f"  Applying {t} at λ={lam:+.1f}")
        if lam == 0.0:
            pass
        elif lam == 1.0:
            naive_merge(model, d)
        else:
            task_arithmetic_merge(model, d, lambda_val=lam)
        del d; gc.collect()

    def _save():
        existing["scores"] = scores
        existing["full"] = fulls
        existing["chosen_lambdas"] = CHOSEN_LAMBDAS
        save_results(existing, out_path)

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
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"OOM during evaluation: {e}")
    except Exception as e:
        logger.error(f"Eval failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
