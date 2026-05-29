"""Proper sampling-robustness for Q2 (the real version).

The greedy multi-seed run (run_multiseed_robustness.py) confirmed that
seeds barely matter under greedy decoding (HumanEval seed-sigma = 0,
GSM8K seed-sigma ~ 0.003). That is reassuring but is NOT what the
reviewer asked. The reviewer asked specifically for:
  - temperature sampling for GSM8K reasoning
  - multi-sample pass@k on HumanEval
and "do TA/TIES still sit near baseline under these variations?"

This script answers exactly that:

  GSM8K self-consistency@K  — K samples at temperature T, majority-vote
    the extracted final answer (the standard SC metric, not lm-eval's
    per-sample mean). Run on a fixed N-question subset for tractability.
    Also runs single-sample greedy on the SAME subset so the SC gain
    and the absolute level are directly comparable.

  HumanEval pass@{1,5,10}  — N samples at temperature 0.2 via lm-eval's
    native pass@k metric.

Methods: baseline (unmerged), naive, TA lambda=0.5, TIES d=0.5 — the
four the reviewer cares about for the "near baseline?" question.

Per-(method, eval) caching; resumable.

Output:
  results/sampling_robustness/{method}_{eval}.json
  results/sampling_robustness/SUMMARY.json
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
from collections import Counter

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_base_model, extract_lora_delta, save_results, load_results, setup_logging
from merge import naive_merge, task_arithmetic_merge, ties_merge

RESULTS_DIR = "results/sampling_robustness"

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
    "baseline":     None,
    "naive":        lambda m, d: naive_merge(m, d),
    "ta_lambda0.5": lambda m, d: task_arithmetic_merge(m, d, lambda_val=0.5),
    "ties_d0.5":    lambda m, d: ties_merge(m, d, density=0.5),
}


def build_model(method_name):
    model, tokenizer = load_base_model(BASE_MODEL)
    fn = METHODS[method_name]
    if fn is None:
        return model, tokenizer
    for _, hf_id in AUDITED_ORDER:
        delta = extract_lora_delta(None, hf_id)
        fn(model, delta)
        del delta; gc.collect()
    return model, tokenizer


# ---------------- GSM8K self-consistency ----------------

_FEWSHOT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell altogether in April and May?
Answer: In April, Natalia sold 48 clips. In May, she sold half as many, so she sold 48 / 2 = 24 clips. Altogether she sold 48 + 24 = 72 clips. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12 / 60 = $0.2 per minute. For 50 minutes she earned 0.2 x 50 = $10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: Betty has half of $100, which is 100 / 2 = $50. Her parents give her $15. Her grandparents give twice as much, 15 x 2 = $30. Now she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5 more. #### 5

"""


def _extract_answer(text: str):
    # Prefer the value after '####'; else the last number in the text.
    m = re.search(r"####\s*([-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".").strip()
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _gold(answer: str):
    m = re.search(r"####\s*([-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else None


def gsm8k_self_consistency(model, tokenizer, n_questions, k_samples, temperature, seed):
    from datasets import load_dataset
    torch.manual_seed(seed)
    ds = load_dataset("openai/gsm8k", "main", split="test")
    n = min(n_questions, len(ds))
    model.eval()

    sc_correct = 0
    greedy_correct = 0
    for i in range(n):
        q, gold_ans = ds[i]["question"], _gold(ds[i]["answer"])
        prompt = _FEWSHOT + f"Question: {q}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2048).to(model.device)
        # K sampled generations
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                 temperature=temperature, num_return_sequences=k_samples,
                                 pad_token_id=tokenizer.pad_token_id)
        preds = []
        for o in out:
            gen = tokenizer.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            gen = gen.split("Question:")[0]  # stop at next question
            p = _extract_answer(gen)
            if p is not None:
                preds.append(p)
        # Majority vote
        if preds:
            maj = Counter(preds).most_common(1)[0][0]
            if gold_ans is not None and _num_eq(maj, gold_ans):
                sc_correct += 1
        # Greedy single-sample on same question
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        gtext = tokenizer.decode(g[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        gtext = gtext.split("Question:")[0]
        gp = _extract_answer(gtext)
        if gp is not None and gold_ans is not None and _num_eq(gp, gold_ans):
            greedy_correct += 1

        if (i + 1) % 50 == 0:
            logger.info(f"    {i+1}/{n}: running SC@{k_samples}={sc_correct/(i+1):.3f} "
                        f"greedy={greedy_correct/(i+1):.3f}")

    return {
        "sc_accuracy": sc_correct / n,
        "greedy_accuracy": greedy_correct / n,
        "n_questions": n,
        "k_samples": k_samples,
        "temperature": temperature,
        "seed": seed,
    }


def _num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return str(a) == str(b)


# ---------------- HumanEval pass@k ----------------

def humaneval_passk(model, tokenizer, n_samples, temperature, seed):
    """Pass@k via lm-eval's repeats mechanism (the `humaneval_10` task variant).

    The HFLM backend does not support num_return_sequences via gen_kwargs
    (it zips contexts 1:1 with generations). The correct lm-eval path is a
    task with `repeats: K`, which re-issues each request K times and lets
    utils.pass_at_k compute the unbiased estimator. humaneval_10 (created
    in the lm_eval task dir) sets repeats=10, k=[1,5,10], T=0.2, top_p=0.95.
    """
    torch.manual_seed(seed)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    res = lm_eval.simple_evaluate(
        model=lm, tasks=["humaneval_10"], confirm_run_unsafe_code=True,
        random_seed=seed, numpy_random_seed=seed, torch_random_seed=seed,
        fewshot_random_seed=seed,
    )
    r = res["results"]["humaneval_10"]
    # pass_at_k emits keys like "pass@1,create_test" or "pass@1,none"
    def _get(k):
        for key in (f"pass@{k},create_test", f"pass@{k},none", f"pass@{k}"):
            if key in r:
                return r[key]
        return None
    return {
        "pass@1": _get(1),
        "pass@5": _get(5),
        "pass@10": _get(10),
        "n_samples": n_samples,
        "temperature": temperature,
        "seed": seed,
        "raw_keys": list(r.keys()),
    }


def run_cell(method, eval_name, eval_fn, **kw):
    path = os.path.join(RESULTS_DIR, f"{method}_{eval_name}.json")
    if os.path.exists(path):
        logger.info(f"  [{method}_{eval_name}] CACHED")
        return load_results(path)
    logger.info(f"\n  [{method}_{eval_name}] running ...")
    model, tokenizer = build_model(method)
    try:
        r = eval_fn(model, tokenizer, **kw)
        r.update({"method": method, "eval": eval_name})
        save_results(r, path)
        logger.info(f"    {method}_{eval_name}: "
                    + json.dumps({k: v for k, v in r.items()
                                  if k in ('sc_accuracy','greedy_accuracy','pass@1','pass@5','pass@10')}))
    except Exception as e:
        logger.error(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        r = {"error": str(e), "method": method, "eval": eval_name}
        save_results(r, path)
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--gsm8k-n", type=int, default=500, help="GSM8K SC subset size")
    p.add_argument("--gsm8k-k", type=int, default=8, help="SC samples per question")
    p.add_argument("--gsm8k-temp", type=float, default=0.7)
    p.add_argument("--he-samples", type=int, default=10)
    p.add_argument("--he-temp", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--skip-gsm8k", action="store_true")
    p.add_argument("--skip-humaneval", action="store_true")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("sampling_robustness", RESULTS_DIR)
    logger.info("SAMPLING ROBUSTNESS (Q2, proper version)")
    logger.info(f"  GSM8K SC@{args.gsm8k_k} T={args.gsm8k_temp} on {args.gsm8k_n}-subset; "
                f"HumanEval pass@k n={args.he_samples} T={args.he_temp}")

    for method in args.methods:
        logger.info(f"\n{'='*70}\n[{method}]\n{'='*70}")
        if not args.skip_gsm8k:
            run_cell(method, "gsm8k_sc", gsm8k_self_consistency,
                     n_questions=args.gsm8k_n, k_samples=args.gsm8k_k,
                     temperature=args.gsm8k_temp, seed=args.seed)
        if not args.skip_humaneval:
            run_cell(method, "humaneval_passk", humaneval_passk,
                     n_samples=args.he_samples, temperature=args.he_temp, seed=args.seed)

    # Summary
    summary = {"methods": {}}
    for method in args.methods:
        entry = {}
        for ev in ["gsm8k_sc", "humaneval_passk"]:
            p_ = os.path.join(RESULTS_DIR, f"{method}_{ev}.json")
            if os.path.exists(p_):
                entry[ev] = load_results(p_)
        summary["methods"][method] = entry
    save_results(summary, os.path.join(RESULTS_DIR, "SUMMARY.json"))

    logger.info("\n" + "=" * 80)
    logger.info("SAMPLING ROBUSTNESS SUMMARY")
    logger.info("=" * 80)
    for method, entry in summary["methods"].items():
        sc = entry.get("gsm8k_sc", {})
        hk = entry.get("humaneval_passk", {})
        logger.info(f"{method:14s}  GSM8K SC@{sc.get('k_samples','?')}={sc.get('sc_accuracy')}"
                    f" (greedy {sc.get('greedy_accuracy')})  "
                    f"HE pass@1={hk.get('pass@1')} pass@10={hk.get('pass@10')}")


if __name__ == "__main__":
    main()
