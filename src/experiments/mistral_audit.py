"""Mistral adapter audit: compute norms + base-model match for the
4 Mistral LoRAs in configs/mistral_config.yaml.

Pure CPU operation — no GPU needed. Tells us in <2 minutes whether
the existing Mistral adapter set passes our audit.
"""
import math
import sys

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from peft import PeftConfig

ADAPTERS = [
    ("code",        "parsak/mistralcode-7b-instruct-lora-adapters"),
    ("math",        "xummer/mistral-7b-gsm8k-lora-en"),
    ("general_nlp", "mkenfenheuer/Mistral-7B-Instruct-v0.3-ha-function-calling-lora"),
    ("creative_writing", "svjack/Mistral7B_v2_inst_sharegpt_roleplay_chat_lora_small"),
]

EXPECTED_BASE = "mistralai/Mistral-7B-Instruct-v0.3"

print(f"\n=== Mistral adapter audit (expected base: {EXPECTED_BASE}) ===\n")
results = []

for task, adapter_id in ADAPTERS:
    try:
        cfg = PeftConfig.from_pretrained(adapter_id)
        declared_base = getattr(cfg, "base_model_name_or_path", "<missing>")
        r, alpha = cfg.r, cfg.lora_alpha
        scaling = alpha / math.sqrt(r) if getattr(cfg, "use_rslora", False) else alpha / r

        path = hf_hub_download(adapter_id, "adapter_model.safetensors")
        weights = load_file(path)

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
        total_norm = total_norm_sq ** 0.5

        base_match = (declared_base == EXPECTED_BASE)
        norm_pass = total_norm >= 1.0
        verdict = "PASS" if base_match and norm_pass else "FAIL"

        print(f"  [{task:18s}] {adapter_id}")
        print(f"     declared base: {declared_base}  match={base_match}")
        print(f"     rank={r:<3} alpha={alpha:<4} scaling={scaling:.3f}")
        print(f"     n_layers={n_layers}  total ||Δ||_F = {total_norm:.3f}")
        print(f"     >>> {verdict} (base_match={base_match}, norm_pass={norm_pass})\n")

        results.append((task, adapter_id, declared_base, total_norm, verdict))

    except Exception as e:
        print(f"  [{task}] ERROR loading {adapter_id}: {e}\n")
        results.append((task, adapter_id, "<error>", 0.0, "ERROR"))

print("\n=== Summary ===")
print(f"{'Task':20s} {'Verdict':10s} {'Norm':>8s}  {'Adapter ID'}")
for task, aid, _base, n, v in results:
    print(f"{task:20s} {v:10s} {n:8.3f}  {aid}")

passing = sum(1 for *_, v in results if v == "PASS")
print(f"\n{passing}/{len(results)} adapters pass the audit.")
