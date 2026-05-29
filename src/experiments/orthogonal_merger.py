"""Runs the orthogonal-projection merger on the audited 4-adapter set.

Companion to src/merge_orthogonal.py. Evaluates on GSM8K + HumanEval
and additionally runs the SAME merger on the PRE-AUDIT pool (with
defective + wrong-base adapters) to test the reviewer's key question:

  "Does the audit alter the ranking of structure-aware mergers in the
   same way it alters TA/TIES/DARE rankings?"

If the orthogonal merger swings by ~10pp between pre-audit and post-audit
just like the magnitude-family mergers do, that is direct evidence that
the audit's effect generalizes to LoRA-aware mergers — addressing the
reviewer's gap-3 open question.

Output:
  results/orthogonal_merger/audited.json
  results/orthogonal_merger/pre_audit_swap.json
  results/orthogonal_merger/SUMMARY.json
"""
import gc
import logging
import os
import sys

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge_orthogonal import orthogonal_project_merge

RESULTS_DIR = "results/orthogonal_merger"

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

PRE_AUDIT_SWAP = [
    ("tianjun_defective",   "TianJun1/llama3.1-8b-code-reflector-lora"),
    ("math",                "kai-xu/Llama-3.1-8B-Instruct-MATH-Finetuned-LoRA"),
    ("general_nlp",         "zjudai/flowertune-general-nlp-lora-llama-3.1-8b-instruct"),
    ("blackroot_wrongbase", "Blackroot/Llama-3-8B-Abomination-LORA"),
]

CONFIGS = {
    "audited": AUDITED_ORDER,
    "pre_audit_swap": PRE_AUDIT_SWAP,
}


def eval_gsm8k(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    r = res["results"]["gsm8k"]
    return {"score": r["exact_match,strict-match"],
            "stderr": r["exact_match_stderr,strict-match"]}


def eval_humaneval(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    return {"score": r.get("pass@1,create_test"),
            "stderr": r.get("pass@1_stderr,create_test")}


BENCHMARKS = [("gsm8k", eval_gsm8k), ("humaneval", eval_humaneval)]


def run_config(name, adapters):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    existing = load_results(path) if os.path.exists(path) else {}
    scores = existing.get("scores", {})
    fulls = existing.get("full", {})
    needed = [b for b, _ in BENCHMARKS if b not in scores]
    if not needed:
        logger.info(f"[{name}] all benchmarks cached, skipping")
        return existing

    logger.info(f"\n{'='*70}\n[{name}] orthogonal-projection merge on {[a[0] for a in adapters]}\n{'='*70}")
    model, tokenizer = load_base_model(BASE_MODEL)

    history: dict = {}
    for slot, hf_id in adapters:
        logger.info(f"  merging {slot} ← {hf_id}")
        delta = extract_lora_delta(None, hf_id)
        history = orthogonal_project_merge(model, delta, cumulative_basis=history)
        del delta; gc.collect()

    try:
        for bench_name, eval_fn in BENCHMARKS:
            if bench_name in scores:
                continue
            logger.info(f"  -> {bench_name} ...")
            try:
                r = eval_fn(model, tokenizer)
                scores[bench_name] = r["score"]
                fulls[bench_name] = r
                logger.info(f"     {bench_name}: {r['score']:.4f} ± {r.get('stderr', 0):.4f}")
            except Exception as e:
                logger.error(f"     {bench_name} FAILED: {e}")
                import traceback; traceback.print_exc()
                scores[bench_name] = None
                fulls[bench_name] = {"error": str(e)}
            save_results({
                "config": name,
                "adapters": [{"slot": a[0], "id": a[1]} for a in adapters],
                "method": "orthogonal_projection",
                "scores": scores,
                "full": fulls,
            }, path)
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()

    return load_results(path)


def summarize():
    """Compare orthogonal merger across audited vs pre-audit pools."""
    summary = {"configs": {}}
    base = load_results(os.path.join("results", "merge_main_baseline.json"))
    summary["unmerged_baseline"] = base["scores"]

    for name in CONFIGS:
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if not os.path.exists(path):
            summary["configs"][name] = {"status": "not yet run"}
            continue
        d = load_results(path)
        s = d.get("scores", {})
        summary["configs"][name] = {
            "scores": s,
            "vs_baseline": {
                k: (s[k] - base["scores"][k]) if s.get(k) is not None else None
                for k in ["gsm8k", "humaneval"]
            },
        }

    # Audit-effect on orthogonal merger
    if all(c in summary["configs"] and "scores" in summary["configs"][c]
            for c in ["audited", "pre_audit_swap"]):
        a = summary["configs"]["audited"]["scores"]
        p = summary["configs"]["pre_audit_swap"]["scores"]
        summary["audit_effect_on_orthogonal_merger"] = {
            k: (a[k] - p[k]) if (a.get(k) is not None and p.get(k) is not None) else None
            for k in ["gsm8k", "humaneval"]
        }

    save_results(summary, os.path.join(RESULTS_DIR, "SUMMARY.json"))

    logger.info("\n" + "=" * 70)
    logger.info("ORTHOGONAL-PROJECTION MERGER SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Unmerged baseline:  GSM8K={base['scores']['gsm8k']:.4f}  "
                f"HE={base['scores']['humaneval']:.4f}")
    for name, info in summary["configs"].items():
        if "scores" not in info:
            logger.info(f"  {name:20s}: NOT YET RUN"); continue
        s = info["scores"]
        gs = f"{s['gsm8k']:.4f}" if s.get('gsm8k') is not None else "—"
        he = f"{s['humaneval']:.4f}" if s.get('humaneval') is not None else "—"
        logger.info(f"  {name:20s}: GSM8K={gs}  HE={he}")
    if "audit_effect_on_orthogonal_merger" in summary:
        eff = summary["audit_effect_on_orthogonal_merger"]
        logger.info(f"\nAudit effect on orthogonal merger:")
        logger.info(f"  GSM8K: {eff.get('gsm8k'):+.4f}  HE: {eff.get('humaneval'):+.4f}")
        logger.info(f"  (Compare to TA/TIES/DARE audit effect of ~10pp in Sec. 4.)")
    logger.info("=" * 70)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("orthogonal_merger", RESULTS_DIR)
    logger.info("ORTHOGONAL-PROJECTION MERGER (reviewer gap-3)")
    for name, adapters in CONFIGS.items():
        try:
            run_config(name, adapters)
        except Exception as e:
            logger.error(f"[{name}] failed: {e}")
            import traceback; traceback.print_exc()
    summarize()


if __name__ == "__main__":
    main()
