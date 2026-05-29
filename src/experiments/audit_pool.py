"""Expanded audit on a larger candidate pool of public LoRAs.

Tests both Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 candidates.
CPU-only, ~30 sec per adapter.
"""
import math
import json
import os

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from peft import PeftConfig

# Llama-3.1-8B-Instruct candidate pool (expanded beyond the 4 in the paper)
LLAMA_CANDIDATES = [
    # Audited 4 (already in paper)
    ("kai-xu/Llama-3.1-8B-Instruct-MATH-Finetuned-LoRA",            "math"),
    ("yangao381/FlowerTune-Code-Llama-3.1-8B-Instruct-PEFT",        "code"),
    ("zjudai/flowertune-general-nlp-lora-llama-3.1-8b-instruct",    "general_nlp"),
    ("yangao381/FlowerTune-Medical-Llama-3.1-8B-Instruct-PEFT",     "medical"),
    # Excluded 2 (already in paper)
    ("TianJun1/llama3.1-8b-code-reflector-lora",                     "code-defective-norm"),
    ("Blackroot/Llama-3-8B-Abomination-LORA",                        "creative-wrong-base"),
    # Additional candidates to expand the pool
    ("Madhuvardhanj/llama-3.1-8B-instruct-lora-text-to-SQL",         "sql"),
    ("Aakash26/llama-3.1-8B-finetuned-lora-french",                  "french"),
    ("Pranjul9/llama-3.1-8B-Instruct-Finetuned-Email-Bot",           "email"),
    ("RamishaShahid24/llama_3.1_8B_lora",                            "generic"),
    ("YudhishtraSingh/Hate-speech-detection-LoRA-LLama-3.1-8B",      "hate-detection"),
    ("yangao381/FlowerTune-NLP-Llama-3.1-8B-Instruct-PEFT",          "nlp-alt"),
    ("FrancescoBonzi/llama-3.1-8b-lora-tools",                       "tools"),
    ("0x18/llama3.1-8b-instruct-trlx-lora",                          "trl-rlhf"),
]

# Mistral-7B-Instruct-v0.3 candidate pool
MISTRAL_CANDIDATES = [
    # Original 4 (already in paper)
    ("parsak/mistralcode-7b-instruct-lora-adapters",                                "code"),
    ("xummer/mistral-7b-gsm8k-lora-en",                                             "math"),
    ("mkenfenheuer/Mistral-7B-Instruct-v0.3-ha-function-calling-lora",              "general_nlp"),
    ("svjack/Mistral7B_v2_inst_sharegpt_roleplay_chat_lora_small",                  "creative_writing"),
    # Additional Mistral candidates
    ("kaitchup/Mistral-7B-Instruct-v0.3-LoRA-test",                                 "test"),
    ("ali-tharun/Mistral-7B-Instruct-v0.3-finetuned-lora",                          "finetune-generic"),
]

LLAMA_BASE = "NousResearch/Meta-Llama-3.1-8B-Instruct"
MISTRAL_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
NORM_THRESHOLD = 1.0


def audit_one(adapter_id: str, expected_base: str):
    """Run audit on one adapter. Returns dict with verdict + diagnostics."""
    try:
        cfg = PeftConfig.from_pretrained(adapter_id)
    except Exception as e:
        return {"id": adapter_id, "error": f"config load: {e}"}

    declared_base = getattr(cfg, "base_model_name_or_path", "<missing>")
    r = getattr(cfg, "r", 0)
    alpha = getattr(cfg, "lora_alpha", 0)
    use_rslora = getattr(cfg, "use_rslora", False)
    scaling = (alpha / math.sqrt(r)) if use_rslora else (alpha / r if r else 0)

    # Base check accepts known equivalences for Llama-3.1
    base_match_targets = {expected_base}
    if expected_base.endswith("Meta-Llama-3.1-8B-Instruct"):
        base_match_targets.add("meta-llama/Llama-3.1-8B-Instruct")
        base_match_targets.add("meta-llama/Meta-Llama-3.1-8B-Instruct")
        base_match_targets.add("NousResearch/Meta-Llama-3.1-8B-Instruct")
    base_match = declared_base in base_match_targets

    # Compute reconstructed delta norm
    try:
        path = hf_hub_download(adapter_id, "adapter_model.safetensors")
        weights = load_file(path)
    except Exception as e:
        return {
            "id": adapter_id,
            "declared_base": declared_base,
            "rank": r, "alpha": alpha, "scaling": round(scaling, 3),
            "base_match": base_match,
            "error": f"weights load: {e}",
        }

    pairs = {}
    for key, tensor in weights.items():
        base_key = (key.replace(".lora_A.default.weight", "")
                       .replace(".lora_A.weight", "")
                       .replace(".lora_B.default.weight", "")
                       .replace(".lora_B.weight", "")
                       .replace("base_model.model.", ""))
        if "lora_A" in key:
            pairs.setdefault(base_key, {})["A"] = tensor.float()
        elif "lora_B" in key:
            pairs.setdefault(base_key, {})["B"] = tensor.float()

    total_norm_sq = 0.0
    n_layers = 0
    for k, d in pairs.items():
        if "A" in d and "B" in d:
            delta = (d["B"] @ d["A"]) * scaling
            total_norm_sq += delta.norm().item() ** 2
            n_layers += 1
    total_norm = round(total_norm_sq ** 0.5, 3)

    norm_pass = total_norm >= NORM_THRESHOLD
    verdict = "PASS" if (base_match and norm_pass) else "FAIL"
    fail_reasons = []
    if not base_match:
        fail_reasons.append(f"base mismatch (declared={declared_base})")
    if not norm_pass:
        fail_reasons.append(f"norm {total_norm} < {NORM_THRESHOLD}")

    return {
        "id": adapter_id,
        "declared_base": declared_base,
        "rank": r, "alpha": alpha, "scaling": round(scaling, 3),
        "n_layers": n_layers,
        "total_norm": total_norm,
        "base_match": base_match,
        "norm_pass": norm_pass,
        "verdict": verdict,
        "fail_reasons": fail_reasons,
    }


def audit_pool(family_name: str, candidates, expected_base: str):
    print(f"\n{'='*70}\nAuditing {len(candidates)} {family_name} candidates "
          f"(expected base: {expected_base})\n{'='*70}\n")
    results = []
    for adapter_id, slot_label in candidates:
        result = audit_one(adapter_id, expected_base)
        result["slot_label"] = slot_label
        results.append(result)

        # Compact print
        v = result.get("verdict", "ERROR")
        if "error" in result:
            print(f"  [{slot_label:25s}] {adapter_id}\n     ERROR: {result['error']}\n")
        else:
            base_str = f"base={'OK' if result['base_match'] else 'NO'}"
            norm_str = f"norm={result['total_norm']}"
            reasons = "; ".join(result.get("fail_reasons", [])) or "all checks passed"
            print(f"  [{slot_label:25s}] {adapter_id}")
            print(f"     {base_str}  {norm_str}  {v}  ({reasons})\n")
    return results


def main():
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    llama_results = audit_pool("Llama-3.1", LLAMA_CANDIDATES, LLAMA_BASE)
    mistral_results = audit_pool("Mistral-7B-v0.3", MISTRAL_CANDIDATES, MISTRAL_BASE)

    summary = {
        "llama": {
            "expected_base": LLAMA_BASE,
            "n_candidates": len(LLAMA_CANDIDATES),
            "n_pass": sum(1 for r in llama_results if r.get("verdict") == "PASS"),
            "n_fail": sum(1 for r in llama_results if r.get("verdict") == "FAIL"),
            "n_error": sum(1 for r in llama_results if "error" in r),
            "results": llama_results,
        },
        "mistral": {
            "expected_base": MISTRAL_BASE,
            "n_candidates": len(MISTRAL_CANDIDATES),
            "n_pass": sum(1 for r in mistral_results if r.get("verdict") == "PASS"),
            "n_fail": sum(1 for r in mistral_results if r.get("verdict") == "FAIL"),
            "n_error": sum(1 for r in mistral_results if "error" in r),
            "results": mistral_results,
        },
    }

    out_path = os.path.join(out_dir, "audit_pool.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  Llama-3.1   {summary['llama']['n_pass']}/{summary['llama']['n_candidates']} pass, "
          f"{summary['llama']['n_fail']} fail, {summary['llama']['n_error']} error")
    print(f"  Mistral-v3  {summary['mistral']['n_pass']}/{summary['mistral']['n_candidates']} pass, "
          f"{summary['mistral']['n_fail']} fail, {summary['mistral']['n_error']} error")
    print(f"\nFull results: {out_path}\n")


if __name__ == "__main__":
    main()
