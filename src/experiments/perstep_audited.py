"""Per-step forgetting curves on the AUDITED 4-adapter Llama set.

Addresses reviewer Q4: produce per-step trajectories for each method
on the audited adapter pool, enabling backward-transfer analysis
typical of continual-learning evaluations.

Distinct from src/run_perstep_leakfree.py, which uses the PRE-AUDIT
ordering (includes wrong-base creative_writing adapter).

Per method, per step: evaluate GSM8K (protected task) and HumanEval
(the code adapter's strength — backward-transfer indicator).
MATH-500 is omitted to keep wall-clock manageable; can be enabled by
extending BENCHMARKS.

Output:
  results/perstep_audited/{method}.json     per-step trajectory
  results/perstep_audited/SUMMARY.json      aggregated table

Compute (RTX 6000 Ada): 3 methods × 4 steps × (GSM8K ~30min + HE ~15min)
≈ 9 hours total. Per-step writes give resumability.
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

RESULTS_DIR = "results/perstep_audited"

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


def magmax_merge(model, lora_delta):
    """MagMax: element-wise keep whichever has larger magnitude."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in lora_delta:
                d = lora_delta[key].to(param.device, dtype=param.dtype)
                merged = param.data + d
                mask = merged.abs() >= param.data.abs()
                param.data = torch.where(mask, merged, param.data)


METHODS = {
    "naive":          lambda m, d: naive_merge(m, d),
    "ta_lambda0.5":   lambda m, d: task_arithmetic_merge(m, d, lambda_val=0.5),
    "ties_d0.5":      lambda m, d: ties_merge(m, d, density=0.5),
    "dare_d0.5":      lambda m, d: dare_merge(m, d, density=0.5),
    "magmax":         lambda m, d: magmax_merge(m, d),
}


def eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    r = res["results"]["gsm8k"]
    return {"score": r["exact_match,strict-match"], "stderr": r["exact_match_stderr,strict-match"]}


def eval_humaneval(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    score = r.get("pass@1,create_test")
    stderr = r.get("pass@1_stderr,create_test")
    return {"score": score, "stderr": stderr}


BENCHMARKS = [("gsm8k", eval_gsm8k), ("humaneval", eval_humaneval)]


def run_method_per_step(method_name, merge_fn, ordering):
    """Sequentially merge each adapter; eval after each step. Resumable per step."""
    path = os.path.join(RESULTS_DIR, f"{method_name}.json")
    existing = load_results(path) if os.path.exists(path) else {}
    steps = existing.get("steps", [])
    completed = {(s["step"], b) for s in steps for b in s.get("metrics", {})}
    if len(steps) == len(ordering) and all(
        all(b in s.get("metrics", {}) for b, _ in BENCHMARKS) for s in steps
    ):
        logger.info(f"[{method_name}] all steps×benchmarks cached, skipping")
        return existing

    logger.info(f"\n{'=' * 70}\n[{method_name}] per-step on {[s for s, _ in ordering]}\n{'=' * 70}")

    model, tokenizer = load_base_model(BASE_MODEL)

    # Replay any cached steps without evaluating (just rebuild model state)
    start_idx = len(steps)
    for i in range(start_idx):
        slot, hf_id = ordering[i]
        logger.info(f"  replaying cached step {i}: +{slot}")
        delta = extract_lora_delta(None, hf_id)
        merge_fn(model, delta)
        del delta; gc.collect()

    for i in range(start_idx, len(ordering)):
        slot, hf_id = ordering[i]
        logger.info(f"\n  step {i}: +{slot}  ← {hf_id}")
        delta = extract_lora_delta(None, hf_id)
        merge_fn(model, delta)
        del delta; gc.collect()

        metrics = {}
        for bench_name, eval_fn in BENCHMARKS:
            logger.info(f"    -> {bench_name} ...")
            try:
                r = eval_fn(model, tokenizer)
                metrics[bench_name] = r
                logger.info(f"       {bench_name}: {r['score']:.4f} ± {r['stderr']:.4f}")
            except Exception as e:
                logger.error(f"       {bench_name} FAILED: {e}")
                import traceback; traceback.print_exc()
                metrics[bench_name] = {"score": None, "stderr": None, "error": str(e)}

        steps.append({"step": i, "adapter": slot, "metrics": metrics})

        # Checkpoint after each step
        save_results({
            "method": method_name,
            "adapter_pool": "audited",
            "ordering": [s for s, _ in ordering],
            "base_model": BASE_MODEL,
            "steps": steps,
        }, path)

    del model; torch.cuda.empty_cache(); gc.collect()
    return load_results(path)


def summarize():
    """Aggregate per-step trajectories into one comparison file."""
    summary = {"methods": {}, "ordering": [s for s, _ in AUDITED_ORDER]}

    # Reference numbers
    base_path = os.path.join("results", "merge_main_baseline.json")
    if os.path.exists(base_path):
        b = load_results(base_path)
        summary["unmerged_baseline"] = b["scores"]

    for method_name in METHODS:
        path = os.path.join(RESULTS_DIR, f"{method_name}.json")
        if not os.path.exists(path):
            summary["methods"][method_name] = {"status": "not yet run"}
            continue
        d = load_results(path)
        per_step = {}
        for s in d.get("steps", []):
            per_step[s["step"]] = {
                "adapter": s["adapter"],
                "gsm8k":     s["metrics"].get("gsm8k", {}).get("score"),
                "humaneval": s["metrics"].get("humaneval", {}).get("score"),
            }
        summary["methods"][method_name] = per_step

    out_path = os.path.join(RESULTS_DIR, "SUMMARY.json")
    save_results(summary, out_path)

    logger.info("\n" + "=" * 80)
    logger.info("PER-STEP FORGETTING CURVES (AUDITED 4-ADAPTER SET)")
    logger.info("=" * 80)
    if "unmerged_baseline" in summary:
        b = summary["unmerged_baseline"]
        logger.info(f"Unmerged baseline: GSM8K={b['gsm8k']:.3f}  HE={b['humaneval']:.3f}")
    logger.info("-" * 80)
    logger.info(f"{'method':18s}  {'step':5s}  {'adapter':15s}  {'GSM8K':>8s}  {'HE':>8s}")
    for method_name in METHODS:
        info = summary["methods"][method_name]
        if "status" in info:
            logger.info(f"{method_name:18s}  NOT YET RUN")
            continue
        for step_idx in sorted(info, key=int):
            s = info[step_idx]
            gs = f"{s['gsm8k']:.3f}"     if s.get("gsm8k")     is not None else "  —  "
            he = f"{s['humaneval']:.3f}" if s.get("humaneval") is not None else "  —  "
            logger.info(f"{method_name:18s}  {step_idx:5}  {s['adapter']:15s}  "
                        f"{gs:>8s}  {he:>8s}")
    logger.info("=" * 80)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("perstep_audited", RESULTS_DIR)
    logger.info("PER-STEP FORGETTING CURVES (reviewer Q4)")
    logger.info(f"  Output: {RESULTS_DIR}")
    logger.info(f"  Methods: {list(METHODS)}")
    logger.info(f"  Ordering: {[s for s, _ in AUDITED_ORDER]}")

    for name, fn in METHODS.items():
        try:
            run_method_per_step(name, fn, AUDITED_ORDER)
        except Exception as e:
            logger.error(f"[{name}] per-step FAILED: {e}")
            import traceback; traceback.print_exc()

    summarize()


if __name__ == "__main__":
    main()
