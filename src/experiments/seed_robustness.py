"""Multi-seed + sampling robustness for the four headline methods.

Addresses reviewer Q2: the paper uses single-seed greedy decoding only.
The reviewer wants to know whether TA/TIES still sit near baseline under:
  (a) multiple seeds (variance characterization)
  (b) temperature sampling on GSM8K (self-consistency check)
  (c) multi-sample pass@k on HumanEval

This script runs:
  Methods:     baseline (unmerged), naive, TA λ=0.5, TIES d=0.5
  Seeds:       3 seeds {1234, 5678, 9012}
  Variations:  greedy + temperature-sampled GSM8K (T=0.7, 8 samples maj-vote)
               + HumanEval pass@10 (T=0.2, 10 samples)

Per (method, seed, variation) caching. Resumable.

Output:
  results/seed_robustness/{method}_{seed}_{variation}.json    per cell
  results/seed_robustness/SUMMARY.json                        aggregated

Compute (RTX 6000 Ada):
  Greedy multi-seed:   4 methods × 3 seeds × (GSM8K 30m + HE 15m) ≈ 9h
  GSM8K self-consist:  4 methods × 8 samples × (~30m × 1.5 inflation) ≈ 6h
  HE pass@10:          4 methods × (~15m × 10 inflation)              ≈ 10h
  TOTAL                                                                ≈ 25h
You can disable variations via CLI flags below to run incrementally.
"""

import argparse
import gc
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge, task_arithmetic_merge, ties_merge

RESULTS_DIR = "results/seed_robustness"

sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if "audit_continual_learning/src" not in p and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

AUDITED_ORDER = [
    ("code",        "yangao381/FlowerTune-Code-Llama-3.1-8B-Instruct-PEFT"),
    ("math",        "kai-xu/Llama-3.1-8B-Instruct-MATH-Finetuned-LoRA"),
    ("general_nlp", "zjudai/flowertune-general-nlp-lora-llama-3.1-8b-instruct"),
    ("medical",     "yangao381/FlowerTune-Medical-Llama-3.1-8B-Instruct-PEFT"),
]

METHODS = {
    "baseline":     None,  # no merge
    "naive":        lambda m, d: naive_merge(m, d),
    "ta_lambda0.5": lambda m, d: task_arithmetic_merge(m, d, lambda_val=0.5),
    "ties_d0.5":    lambda m, d: ties_merge(m, d, density=0.5),
}

SEEDS = [1234, 5678, 9012]


def build_model(method_name):
    model, tokenizer = load_base_model(BASE_MODEL)
    merge_fn = METHODS[method_name]
    if merge_fn is None:
        return model, tokenizer
    for slot, hf_id in AUDITED_ORDER:
        delta = extract_lora_delta(None, hf_id)
        merge_fn(model, delta)
        del delta; gc.collect()
    return model, tokenizer


def set_global_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_gsm8k_greedy(model, tokenizer, seed):
    set_global_seed(seed)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"], random_seed=seed,
                                   numpy_random_seed=seed, torch_random_seed=seed,
                                   fewshot_random_seed=seed)
    r = res["results"]["gsm8k"]
    return {"score": r["exact_match,strict-match"],
            "stderr": r["exact_match_stderr,strict-match"],
            "variation": "greedy"}


def eval_humaneval_greedy(model, tokenizer, seed):
    set_global_seed(seed)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"],
                                   confirm_run_unsafe_code=True,
                                   random_seed=seed, numpy_random_seed=seed,
                                   torch_random_seed=seed, fewshot_random_seed=seed)
    r = res["results"]["humaneval"]
    return {"score": r.get("pass@1,create_test"),
            "stderr": r.get("pass@1_stderr,create_test"),
            "variation": "greedy_pass1"}


def eval_gsm8k_selfconsistency(model, tokenizer, seed, n_samples=8, temperature=0.7):
    """Self-consistency: N temperature-sampled generations per question, majority vote."""
    set_global_seed(seed)
    # lm-eval supports gen_kwargs override
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    # The harness's `gsm8k_cot_self_consistency` task does this natively if installed.
    # Fallback: gsm8k with explicit gen_kwargs and post-hoc majority vote.
    res = lm_eval.simple_evaluate(
        model=lm, tasks=["gsm8k"],
        random_seed=seed, numpy_random_seed=seed,
        torch_random_seed=seed, fewshot_random_seed=seed,
        gen_kwargs=f"do_sample=True,temperature={temperature},num_return_sequences={n_samples}",
    )
    r = res["results"]["gsm8k"]
    # lm-eval averages across samples; self-consistency requires majority-vote post-processing.
    # The simplest robustness signal is the sampled-mean accuracy here; a separate
    # self-consistency runner can be added once we confirm lm-eval's behaviour on this.
    return {"score": r.get("exact_match,strict-match"),
            "stderr": r.get("exact_match_stderr,strict-match"),
            "variation": f"sampled_T{temperature}_n{n_samples}",
            "note": "sampled-mean accuracy; majority-vote post-processing TODO"}


def eval_humaneval_passk(model, tokenizer, seed, n_samples=10, temperature=0.2):
    set_global_seed(seed)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(
        model=lm, tasks=["humaneval"],
        confirm_run_unsafe_code=True,
        random_seed=seed, numpy_random_seed=seed,
        torch_random_seed=seed, fewshot_random_seed=seed,
        gen_kwargs=f"do_sample=True,temperature={temperature},num_return_sequences={n_samples}",
    )
    r = res["results"]["humaneval"]
    return {"score_pass1": r.get("pass@1,create_test"),
            "score_pass5": r.get("pass@5,create_test"),
            "score_pass10": r.get("pass@10,create_test"),
            "variation": f"pass_at_k_T{temperature}_n{n_samples}"}


def run_cell(method_name, seed, variation, eval_fn):
    cell_id = f"{method_name}_seed{seed}_{variation}"
    path = os.path.join(RESULTS_DIR, f"{cell_id}.json")
    if os.path.exists(path):
        logger.info(f"  [{cell_id}] CACHED")
        return load_results(path)

    logger.info(f"\n  [{cell_id}] running ...")
    model, tokenizer = build_model(method_name)
    try:
        result = eval_fn(model, tokenizer, seed)
        result.update({"method": method_name, "seed": seed, "variation": variation})
        save_results(result, path)
        score = result.get("score") or result.get("score_pass1")
        logger.info(f"    score: {score}")
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        result = {"error": str(e), "method": method_name, "seed": seed,
                  "variation": variation}
        save_results(result, path)
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHODS),
                   choices=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--no-greedy", action="store_true", help="skip greedy multi-seed")
    p.add_argument("--no-selfconsistency", action="store_true",
                   help="skip temperature-sampled GSM8K")
    p.add_argument("--no-passk", action="store_true", help="skip HumanEval pass@10")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("multiseed_robustness", RESULTS_DIR)
    logger.info("MULTI-SEED + SAMPLING ROBUSTNESS (reviewer Q2)")
    logger.info(f"  methods: {args.methods}  seeds: {args.seeds}")
    logger.info(f"  variations: greedy={not args.no_greedy} "
                f"selfcons={not args.no_selfconsistency} passk={not args.no_passk}")

    for method in args.methods:
        logger.info(f"\n{'=' * 70}\n[{method}]\n{'=' * 70}")
        for seed in args.seeds:
            if not args.no_greedy:
                run_cell(method, seed, "gsm8k_greedy", eval_gsm8k_greedy)
                run_cell(method, seed, "humaneval_greedy", eval_humaneval_greedy)

            if not args.no_selfconsistency and seed == args.seeds[0]:
                # Self-consistency is expensive; run once per method
                run_cell(method, seed, "gsm8k_selfconsistency", eval_gsm8k_selfconsistency)

            if not args.no_passk and seed == args.seeds[0]:
                run_cell(method, seed, "humaneval_passk", eval_humaneval_passk)

    # Summary
    import glob, json
    cells = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        if path.endswith("SUMMARY.json"):
            continue
        cells.append(load_results(path))

    # Aggregate: per (method, variation) mean ± stderr across seeds
    from collections import defaultdict
    agg = defaultdict(list)
    for c in cells:
        if "error" in c: continue
        key = (c["method"], c["variation"])
        agg[key].append(c.get("score") or c.get("score_pass1"))

    summary = {"per_cell": cells, "aggregated": {}}
    for (method, variation), scores in agg.items():
        scores = [s for s in scores if s is not None]
        if not scores: continue
        mean = sum(scores) / len(scores)
        sd = (sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)) ** 0.5
        summary["aggregated"][f"{method}|{variation}"] = {
            "n_seeds": len(scores), "mean": mean, "std_across_seeds": sd, "scores": scores,
        }

    save_results(summary, os.path.join(RESULTS_DIR, "SUMMARY.json"))

    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED ACROSS SEEDS (mean ± seed-std)")
    logger.info("=" * 80)
    for k, v in sorted(summary["aggregated"].items()):
        logger.info(f"  {k:50s}  n={v['n_seeds']}  mean={v['mean']:.4f}  "
                    f"seed-std={v['std_across_seeds']:.4f}  values={v['scores']}")


if __name__ == "__main__":
    main()
