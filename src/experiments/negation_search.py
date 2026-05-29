"""Negative per-adapter coefficient search.

Task-arithmetic with per-adapter λ that can take NEGATIVE values. Key
insight: if an adapter HURTS the protected task on its own (measured),
applying its delta with a negative coefficient partially CANCELS the
damage. No existing LoRA-merge baseline allows negative coefficients.

Algorithm:
  1. Apply all adapters at λ=1 (naive merge).
  2. For each adapter t in ordering:
       for λ_cand in {-0.5, 0, 0.5, 1.0}:
         compute incremental update (λ_cand - λ_current) · Δ_t
         apply, evaluate probe, decide
       set λ_t to best value
  3. Final model uses optimized per-adapter λ.

4 adapters × 4 candidates × ~1.5 min probe = ~25 min search, ~1h eval.

Compared to Option B (per-adapter α on GF-EWC), this is:
- Simpler (1 λ per adapter vs 1 λ per (adapter, block))
- Uses the negative-coefficient degree of freedom (task-arithmetic negation)
- Directly searched on probe, no importance-proxy
"""
import gc
import logging
import os
import re
import sys
import time

import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, "src")
from utils import load_config, load_base_model, extract_lora_delta, save_results, load_results, setup_logging

RESULTS_DIR = "results"

sys.modules.pop("evaluate", None)
sys.path = [p for p in sys.path if "audit_continual_learning/src" not in p and p != "src"]
import lm_eval
from lm_eval.models.huggingface import HFLM

logger = logging.getLogger(__name__)


def load_gsm8k_probe(n=200):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [{"question": ds[i]["question"], "answer": ds[i]["answer"]}
            for i in range(n)]


def probe_gsm8k_fast(model, tokenizer, examples, batch_size=16):
    model.eval()
    prompts = [ex["question"] + "\n" for ex in examples]
    correct = 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        for j, out in enumerate(outputs):
            gen = tokenizer.decode(out[inputs["input_ids"][j].shape[0]:], skip_special_tokens=True)
            gold_match = re.search(r'####\s*([-\d,\.]+)', examples[i + j]["answer"])
            gold = gold_match.group(1).replace(",", "").strip() if gold_match else None
            nums = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', gen)
            pred = nums[-1].replace(",", "") if nums else None
            if gold is not None and pred is not None:
                try:
                    if abs(float(gold) - float(pred)) < 1e-6:
                        correct += 1
                except ValueError:
                    pass
    return correct / len(examples)


def apply_adapter_scale(model, delta_dict, param_by_key, scale_delta):
    """Add `scale_delta * delta_dict[key]` to each matching param."""
    if scale_delta == 0.0:
        return
    with torch.no_grad():
        for key, d in delta_dict.items():
            if key in param_by_key:
                param = param_by_key[key]
                dt = d.to(param.device, dtype=param.dtype)
                param.data.add_(scale_delta * dt)


def eval_gsm8k_full(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k"])
    return res["results"]["gsm8k"]["exact_match,strict-match"], res["results"]["gsm8k"]


def eval_humaneval_full(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    res = lm_eval.simple_evaluate(model=lm, tasks=["humaneval"], confirm_run_unsafe_code=True)
    r = res["results"]["humaneval"]
    return r.get("pass@1,create_test", r.get("pass@1")), r


def eval_math500_full(model, tokenizer):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    res = lm_eval.simple_evaluate(model=lm, tasks=["hendrycks_math500"], num_fewshot=4)
    r = res["results"]["hendrycks_math500"]
    return r.get("exact_match,none", r.get("exact_match")), r


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    setup_logging("neg_per_adapter", RESULTS_DIR)
    config = load_config("configs/experiment_config.yaml")

    model_name = config["base_models"][0]["name"]
    short_name = config["base_models"][0]["short_name"]
    ordering = config["merge_orderings"][0]
    adapters = config["lora_adapters"][short_name]
    adapter_by_task = {a["task"]: a for a in adapters}

    probe = load_gsm8k_probe(n=200)
    logger.info(f"Loaded {len(probe)} GSM8K probe examples")

    logger.info("\n>>> STEP 1: Load adapter deltas, apply naive merge <<<")
    model, tokenizer = load_base_model(model_name)
    param_by_key = {name.replace(".weight", ""): p for name, p in model.named_parameters()}

    deltas = {}
    for t in ordering:
        deltas[t] = extract_lora_delta(None, adapter_by_task[t]["id"])

    current_lambda = {t: 1.0 for t in ordering}
    for t in ordering:
        apply_adapter_scale(model, deltas[t], param_by_key, current_lambda[t])

    t0 = time.time()
    naive_probe = probe_gsm8k_fast(model, tokenizer, probe)
    logger.info(f"Naive probe accuracy: {naive_probe:.3f} (eval {time.time()-t0:.1f}s)")

    logger.info("\n>>> STEP 2: Search per-adapter λ including negative values <<<")
    candidates = [-0.5, 0.0, 0.5, 1.0]
    best_probe = naive_probe
    history = {"naive_probe": naive_probe, "per_adapter": []}

    for t in ordering:
        d_t = deltas[t]
        tried = []
        best_cand = current_lambda[t]
        best_cand_acc = best_probe
        t_adapter = time.time()

        for cand in candidates:
            delta_scale = cand - current_lambda[t]
            if abs(delta_scale) < 1e-9:
                tried.append({"lambda": cand, "probe_acc": best_probe})
                if best_probe > best_cand_acc:
                    best_cand = cand
                    best_cand_acc = best_probe
                continue

            apply_adapter_scale(model, d_t, param_by_key, delta_scale)
            current_lambda[t] = cand
            acc = probe_gsm8k_fast(model, tokenizer, probe)
            tried.append({"lambda": cand, "probe_acc": acc})
            logger.info(f"    adapter={t:12s}  λ={cand:+.1f}  probe={acc:.3f}")
            if acc > best_cand_acc:
                best_cand = cand
                best_cand_acc = acc

        revert_scale = best_cand - current_lambda[t]
        if abs(revert_scale) > 1e-9:
            apply_adapter_scale(model, d_t, param_by_key, revert_scale)
            current_lambda[t] = best_cand
        best_probe = best_cand_acc

        logger.info(f"  [{t}] best λ={best_cand:+.1f} → probe={best_cand_acc:.3f}  ({time.time()-t_adapter:.1f}s)")
        history["per_adapter"].append({"adapter": t, "best_lambda": best_cand,
                                       "probe_acc": best_cand_acc, "tried": tried})

    save_results({"naive_probe": naive_probe, "final_probe": best_probe,
                  "lambda_per_adapter": current_lambda, "history": history,
                  "probe_n": len(probe)},
                 os.path.join(RESULTS_DIR, "negation_n200_history.json"))
    logger.info(f"\nSearch done. Probe: {naive_probe:.3f} → {best_probe:.3f}")
    logger.info(f"Final per-adapter λ: {current_lambda}")

    logger.info("\n>>> STEP 3: Full-benchmark eval on chosen λ vector <<<")
    out_path = os.path.join(RESULTS_DIR, "negation_n200.json")
    out = {"method": "neg_per_adapter", "lambda_per_adapter": current_lambda,
           "naive_probe": naive_probe, "final_probe": best_probe,
           "probe_n": len(probe),
           "scores": {}, "full": {}}

    def _save():
        save_results(out, out_path)

    try:
        logger.info("  -> MATH-500 ...")
        s, r = eval_math500_full(model, tokenizer)
        out["scores"]["math500"] = s; out["full"]["math500"] = r
        logger.info(f"     MATH-500: {s:.4f}")
        _save()

        logger.info("  -> GSM8K (full) ...")
        s, r = eval_gsm8k_full(model, tokenizer)
        out["scores"]["gsm8k"] = s; out["full"]["gsm8k"] = r
        logger.info(f"     GSM8K: {s:.4f}")
        _save()

        logger.info("  -> HumanEval ...")
        s, r = eval_humaneval_full(model, tokenizer)
        out["scores"]["humaneval"] = s; out["full"]["humaneval"] = r
        logger.info(f"     HumanEval: {s:.4f}")
        _save()
    except Exception as e:
        logger.error(f"Full-eval failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        del model; torch.cuda.empty_cache(); gc.collect()


if __name__ == "__main__":
    main()
