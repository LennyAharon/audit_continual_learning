"""Controlled ablation isolating each excluded adapter's effect.

Addresses reviewer Q1: substantiate the ~10pp inflation claim by running
each failing adapter individually on top of the audited 4-adapter set,
and (bonus) recover the original pre-audit 4-adapter configuration on
the FULL test set to resolve the apples-to-oranges concern in App. A.

All runs are naive merges (matching the paper's headline interference
claim). Each (config, benchmark) pair writes its own JSON; existing files
are skipped on rerun.

Output:
  results/excluded_ablation/{config}.json   per config
  results/excluded_ablation/SUMMARY.json    aggregated table

Compute (RTX 6000 Ada): 4 configs × {GSM8K full ~30min, HumanEval ~15min}
≈ 3 hours total. Each config independently cached for resumability.
"""
import gc
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge

RESULTS_DIR = "results/excluded_ablation"

sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if "audit_continual_learning/src" not in p and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


BASE_MODEL = "NousResearch/Meta-Llama-3.1-8B-Instruct"

# IDs lifted from results/audit_pool.json
AUDITED = {
    "code":        "yangao381/FlowerTune-Code-Llama-3.1-8B-Instruct-PEFT",
    "math":        "kai-xu/Llama-3.1-8B-Instruct-MATH-Finetuned-LoRA",
    "general_nlp": "zjudai/flowertune-general-nlp-lora-llama-3.1-8b-instruct",
    "medical":     "yangao381/FlowerTune-Medical-Llama-3.1-8B-Instruct-PEFT",
}
EXCLUDED = {
    "tianjun_defective":   "TianJun1/llama3.1-8b-code-reflector-lora",      # norm 0.016
    "blackroot_wrongbase": "Blackroot/Llama-3-8B-Abomination-LORA",         # norm 46.091
}

# Each config is an ordered list of (slot_label, hf_id) pairs.
AUDITED_ORDER = [(k, AUDITED[k]) for k in ["code", "math", "general_nlp", "medical"]]

CONFIGS = {
    # Parity check: should reproduce merge_main_naive.json (GSM8K 0.603, HE 0.445).
    "audited_4_baseline": AUDITED_ORDER,

    # Reviewer Q1: each failing adapter on top of the audited 4.
    "audited_plus_tianjun":   AUDITED_ORDER + [("tianjun_defective",   EXCLUDED["tianjun_defective"])],
    "audited_plus_blackroot": AUDITED_ORDER + [("blackroot_wrongbase", EXCLUDED["blackroot_wrongbase"])],

    # Both failing adapters on top of the audited 4 (worst case).
    "audited_plus_both": AUDITED_ORDER + [
        ("tianjun_defective",   EXCLUDED["tianjun_defective"]),
        ("blackroot_wrongbase", EXCLUDED["blackroot_wrongbase"]),
    ],

    # The actual pre-audit 4-adapter set: TianJun1 swapped for audited-code,
    # Blackroot swapped for audited-medical. Recovers the "~21pp" original
    # number on the FULL test set (paper currently cites an n=200 subset).
    "pre_audit_swap": [
        ("tianjun_defective",   EXCLUDED["tianjun_defective"]),
        ("math",                AUDITED["math"]),
        ("general_nlp",         AUDITED["general_nlp"]),
        ("blackroot_wrongbase", EXCLUDED["blackroot_wrongbase"]),
    ],
}


def eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    r = res["results"]["gsm8k"]
    return {"score": r["exact_match,strict-match"], "full": r}


def eval_humaneval(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    score = r.get("pass@1,create_test", r.get("pass@1", r.get("pass_at_1")))
    return {"score": score, "full": r}


BENCHMARKS = [("gsm8k", eval_gsm8k), ("humaneval", eval_humaneval)]


def build_merged_model(adapters):
    """Naive sequential merge of the given (label, hf_id) list."""
    model, tokenizer = load_base_model(BASE_MODEL)
    for label, hf_id in adapters:
        logger.info(f"  merging {label} ← {hf_id}")
        delta = extract_lora_delta(None, hf_id)
        naive_merge(model, delta)
        del delta; gc.collect()
    return model, tokenizer


def run_config(name, adapters):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    existing = load_results(path) if os.path.exists(path) else {}
    scores = existing.get("scores", {})
    fulls = existing.get("full", {})

    needed = [b for b, _ in BENCHMARKS if b not in scores]
    if not needed:
        logger.info(f"[{name}] all benchmarks cached, skipping")
        return existing

    logger.info(f"\n{'=' * 70}\n[{name}]  adapters={[a[0] for a in adapters]}\n"
                f"  pending benchmarks: {needed}\n{'=' * 70}")

    model, tokenizer = build_merged_model(adapters)
    try:
        for bench_name, eval_fn in BENCHMARKS:
            if bench_name in scores:
                continue
            logger.info(f"  -> {bench_name} ...")
            try:
                r = eval_fn(model, tokenizer)
                scores[bench_name] = r["score"]
                fulls[bench_name] = r["full"]
                logger.info(f"     {bench_name}: {r['score']:.4f}")
            except Exception as e:
                logger.error(f"     {bench_name} FAILED: {e}")
                import traceback; traceback.print_exc()
                scores[bench_name] = None
                fulls[bench_name] = {"error": str(e)}

            save_results({
                "config": name,
                "adapters": [{"slot": a[0], "id": a[1]} for a in adapters],
                "method": "naive",
                "scores": scores,
                "full": fulls,
            }, path)
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()

    return load_results(path)


def summarize():
    """Pull all per-config JSONs into a single table for the paper."""
    summary = {"configs": {}, "comparison": {}}
    baseline = load_results(os.path.join("results", "merge_main_baseline.json"))
    naive_audited = load_results(os.path.join("results", "merge_main_naive.json"))

    summary["reference"] = {
        "unmerged_baseline": baseline["scores"],
        "audited_4_naive_paper": naive_audited["scores"],
    }

    for name in CONFIGS:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if not os.path.exists(path):
            summary["configs"][name] = {"status": "not yet run"}
            continue
        d = load_results(path)
        scores = d.get("scores", {})
        summary["configs"][name] = {
            "adapters":   [a["slot"] for a in d.get("adapters", [])],
            "scores":     scores,
            "vs_baseline": {
                k: (scores[k] - baseline["scores"][k]) if scores.get(k) is not None else None
                for k in ["gsm8k", "humaneval"]
            },
            "vs_audited_4_naive": {
                k: (scores[k] - naive_audited["scores"][k]) if scores.get(k) is not None else None
                for k in ["gsm8k", "humaneval"]
            },
        }

    out_path = os.path.join(RESULTS_DIR, "SUMMARY.json")
    save_results(summary, out_path)

    # Print
    logger.info("\n" + "=" * 70)
    logger.info("EXCLUDED-ADAPTER ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Unmerged baseline: GSM8K={baseline['scores']['gsm8k']:.4f}  "
                f"HE={baseline['scores']['humaneval']:.4f}")
    logger.info(f"Audited-4 naive (paper Table 1): "
                f"GSM8K={naive_audited['scores']['gsm8k']:.4f}  "
                f"HE={naive_audited['scores']['humaneval']:.4f}")
    logger.info("-" * 70)
    for name, info in summary["configs"].items():
        if "scores" not in info:
            logger.info(f"  {name:30s}: NOT YET RUN")
            continue
        s = info["scores"]
        gs = f"{s['gsm8k']:.4f}" if s.get('gsm8k') is not None else "  —  "
        he = f"{s['humaneval']:.4f}" if s.get('humaneval') is not None else "  —  "
        d  = info["vs_baseline"]
        dgs = f"{d['gsm8k']:+.4f}" if d.get('gsm8k') is not None else "   —  "
        dhe = f"{d['humaneval']:+.4f}" if d.get('humaneval') is not None else "   —  "
        logger.info(f"  {name:30s}: GSM8K={gs} ({dgs} vs baseline)  HE={he} ({dhe})")
    logger.info("=" * 70)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("excluded_adapter_ablation", RESULTS_DIR)
    logger.info("EXCLUDED-ADAPTER ABLATION (reviewer Q1)")
    logger.info(f"  Output dir: {RESULTS_DIR}")
    logger.info(f"  Configs:    {list(CONFIGS)}")

    for name, adapters in CONFIGS.items():
        try:
            run_config(name, adapters)
        except Exception as e:
            logger.error(f"[{name}] config FAILED: {e}")
            import traceback; traceback.print_exc()

    summarize()


if __name__ == "__main__":
    main()
