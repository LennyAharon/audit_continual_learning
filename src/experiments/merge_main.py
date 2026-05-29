"""Main 4-adapter merge experiment (Table 1, Figure 1 methods panel).

Reads the four ``paper-audited`` Llama-3.1 slots from configs/adapter_pool.json,
applies a chosen merge method sequentially in the canonical
``code -> math -> general_nlp -> medical`` order, and evaluates on
GSM8K / HumanEval / MATH-500.

Per-(method, benchmark) results are checkpointed to results/merge_main_<method>.json
so interrupted runs resume from the last completed benchmark.

Example:
    python -m src.experiments.merge_main --method ties --density 0.5
    python -m src.experiments.merge_main --method task_arithmetic --lambda-val 0.5
    python -m src.experiments.merge_main --method magmax
"""
import argparse
import gc
import json
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

# Repo-relative imports.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge, task_arithmetic_merge, ties_merge, dare_merge, magmax_merge

# lm-eval imports last (it has an `evaluate` shim that collides with our src/evaluate.py).
sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if not p.endswith("/src") and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(ROOT, "results")
ADAPTER_POOL = os.path.join(ROOT, "configs", "adapter_pool.json")
DEFAULT_BASE = "NousResearch/Meta-Llama-3.1-8B-Instruct"
DEFAULT_ORDER = ["code", "math", "general_nlp", "medical"]


def _load_audited_slots(pool_path: str, family: str = "llama") -> dict:
    """Return {slot_short_name: hf_id} for the four paper-audited adapters."""
    with open(pool_path) as fh:
        pool = json.load(fh)
    candidates = pool[family]["candidates"]
    audited = {}
    for c in candidates:
        label = c.get("slot_label", "")
        if "paper-audited" not in label:
            continue
        slot = label.split()[0]  # "math (paper-audited)" -> "math"
        audited[slot] = c["id"]
    missing = [s for s in DEFAULT_ORDER if s not in audited]
    if missing:
        raise RuntimeError(f"Missing paper-audited slots in {pool_path}: {missing}")
    return audited


def _apply_method(model, delta, method, lambda_val, density):
    if method == "naive":
        naive_merge(model, delta)
    elif method == "task_arithmetic":
        task_arithmetic_merge(model, delta, lambda_val=lambda_val)
    elif method == "ties":
        ties_merge(model, delta, density=density)
    elif method == "dare":
        dare_merge(model, delta, density=density)
    elif method == "magmax":
        magmax_merge(model, delta)
    elif method == "baseline":
        pass
    else:
        raise ValueError(method)


def _eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    r = res["results"]["gsm8k"]
    return {"score": r["exact_match,strict-match"], "full": r}


def _eval_humaneval(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    score = r.get("pass@1,create_test", r.get("pass@1", r.get("pass_at_1")))
    return {"score": score, "full": r}


def _eval_math500(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1)
    res = lm_eval.simple_evaluate(model=lm, tasks=["hendrycks_math500"], num_fewshot=4)
    r = res["results"]["hendrycks_math500"]
    score = r.get("exact_match,none", r.get("acc,none", r.get("exact_match", r.get("acc"))))
    return {"score": score, "full": r}


BENCHMARKS = [
    ("gsm8k", _eval_gsm8k),
    ("humaneval", _eval_humaneval),
    ("math500", _eval_math500),
]


def run(method, lambda_val, density, base_model, family, order, tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging(f"merge_main_{method}{tag}", RESULTS_DIR)

    suffix = ""
    if method == "task_arithmetic":
        suffix = f"_lambda{lambda_val}"
    elif method in ("ties", "dare"):
        suffix = f"_d{density}"
    out_path = os.path.join(RESULTS_DIR, f"merge_main_{method}{suffix}.json")

    existing = load_results(out_path) if os.path.exists(out_path) else {}
    scores = existing.get("scores", {})
    fulls = existing.get("full", {})
    needed = [b for b, _ in BENCHMARKS if b not in scores]
    if not needed:
        logger.info(f"[{method}] all benchmarks cached at {out_path}, skipping")
        return existing

    logger.info(f"\n{'=' * 70}\n[{method}]  pending benchmarks: {needed}\n{'=' * 70}")

    slots = _load_audited_slots(ADAPTER_POOL, family=family)
    model, tokenizer = load_base_model(base_model)
    if method != "baseline":
        for task_name in order:
            delta = extract_lora_delta(None, slots[task_name])
            _apply_method(model, delta, method, lambda_val, density)
            del delta
            gc.collect()

    try:
        for bench_name, eval_fn in BENCHMARKS:
            if bench_name in scores:
                continue
            logger.info(f"  -> {bench_name} ...")
            try:
                r = eval_fn(model, tokenizer)
                scores[bench_name] = r["score"]
                fulls[bench_name] = r["full"]
                logger.info(f"     {bench_name}: {r['score']}")
            except Exception as exc:
                logger.error(f"     {bench_name} FAILED: {exc}")
                scores[bench_name] = None
                fulls[bench_name] = {"error": str(exc)}
            save_results({
                "method": method,
                "params": {"lambda_val": lambda_val, "density": density},
                "base_model": base_model,
                "order": order,
                "scores": scores,
                "full": fulls,
            }, out_path)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return load_results(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="naive",
                   choices=["baseline", "naive", "task_arithmetic", "ties", "dare", "magmax"])
    p.add_argument("--lambda-val", type=float, default=0.5,
                   help="Task Arithmetic coefficient")
    p.add_argument("--density", type=float, default=0.5,
                   help="TIES / DARE density")
    p.add_argument("--base-model", default=DEFAULT_BASE)
    p.add_argument("--family", default="llama", choices=["llama", "mistral"])
    p.add_argument("--order", nargs="+", default=DEFAULT_ORDER,
                   help="Sequential merge order over adapter slots")
    p.add_argument("--tag", default="", help="Optional suffix on log filename")
    args = p.parse_args()
    run(args.method, args.lambda_val, args.density, args.base_model,
        args.family, args.order, args.tag)


if __name__ == "__main__":
    main()
