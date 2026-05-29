"""Content-based audit checks beyond metadata-string matching.

Addresses reviewer Q3 + tech-limit-1: the existing audit relies on
`adapter_config.json` strings. Adversarial or mistakenly-set metadata
could pass that check. This module adds three independent signals:

1. tokenizer_compat — Verify the adapter's effective tokenizer matches
   the declared base. Detects: adapters that tuned embed_tokens/lm_head
   for a different vocabulary, or that ship their own tokenizer files.
   CPU-only, ~5s per adapter.

2. base_signature — Compute a hash signature of the declared base
   checkpoint so it can be verified across machines. CPU-only, downloads
   the base on first run, ~30s.

3. output_divergence — Apply the adapter to the declared base, generate
   on a fixed prompt set, measure KL divergence vs the unmodified base.
   Suspiciously low → adapter is a no-op (catches the norm-0.02 case
   from a different angle). Suspiciously high on a small δ-norm → almost
   certainly a wrong-base adapter (catches the Blackroot case via
   behaviour, not metadata). Requires GPU, ~30s per adapter.

Each check returns a dict; failures don't raise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Prompts chosen to span instruction-following, factual recall, and
# reasoning. Short, deterministic. Same set across all adapters so
# divergence scores are comparable.
SANITY_PROMPTS = [
    "The capital of France is",
    "2 + 2 =",
    "Translate to French: Hello world.",
    "Python code to compute factorial:",
    "Once upon a time,",
    "The sum of 17 and 28 is",
    "def fibonacci(n):",
    "Q: What year did World War II end? A:",
    "The square root of 144 is",
    "Write a haiku about autumn:",
]


# ----------------------------------------------------------------------
# 1. Tokenizer compatibility
# ----------------------------------------------------------------------

def check_tokenizer_compat(adapter_id: str, base_model_id: str) -> dict:
    """Verify adapter's tokenizer assumptions match the declared base.

    Three sub-checks:
      a) Does the adapter target embedding/lm_head layers? (vocab-dependent)
      b) If yes, do the lora_A input/lora_B output dims match base vocab_size?
      c) If the adapter ships tokenizer files, do they hash-match the base?
    """
    from huggingface_hub import hf_hub_download
    from peft import PeftConfig
    from transformers import AutoConfig

    out: dict[str, Any] = {
        "adapter_id": adapter_id,
        "base_model_id": base_model_id,
        "targets_embeddings": False,
        "vocab_dims_match": None,
        "tokenizer_hashes": None,
        "compat": None,
        "notes": [],
    }

    try:
        cfg = PeftConfig.from_pretrained(adapter_id)
        base_cfg = AutoConfig.from_pretrained(base_model_id)
    except Exception as e:
        out["error"] = f"config load: {e}"
        out["compat"] = False
        return out

    base_vocab_size = getattr(base_cfg, "vocab_size", None)

    # (a) Does it target vocab-dependent layers?
    target_modules = getattr(cfg, "target_modules", None) or []
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    vocab_layers = {"embed_tokens", "lm_head", "wte", "wpe", "output"}
    targets_vocab = any(tm in vocab_layers for tm in target_modules)
    out["targets_embeddings"] = targets_vocab

    # (b) If we have actual adapter weights, verify dimensions
    if targets_vocab:
        try:
            from safetensors.torch import load_file
            path = hf_hub_download(adapter_id, "adapter_model.safetensors")
            weights = load_file(path)
            dim_mismatches = []
            for k, t in weights.items():
                if any(v in k for v in vocab_layers):
                    if "lora_A" in k and t.shape[-1] != base_vocab_size:
                        dim_mismatches.append(
                            (k, list(t.shape), f"expected last dim {base_vocab_size}")
                        )
                    if "lora_B" in k and t.shape[0] != base_vocab_size:
                        dim_mismatches.append(
                            (k, list(t.shape), f"expected first dim {base_vocab_size}")
                        )
            out["vocab_dims_match"] = (len(dim_mismatches) == 0)
            if dim_mismatches:
                out["notes"].append(f"vocab-layer dim mismatches: {dim_mismatches[:3]}")
        except Exception as e:
            out["vocab_dims_match"] = None
            out["notes"].append(f"weight-shape check skipped: {e}")

    # (c) Tokenizer file hashes
    tok_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
    adapter_hashes = {}
    base_hashes = {}
    for fname in tok_files:
        for repo_id, store in [(adapter_id, adapter_hashes),
                                (base_model_id, base_hashes)]:
            try:
                p = hf_hub_download(repo_id, fname)
                with open(p, "rb") as f:
                    store[fname] = hashlib.sha256(f.read()).hexdigest()[:16]
            except Exception:
                pass

    if adapter_hashes:
        match_status = {
            fname: (adapter_hashes.get(fname) == base_hashes.get(fname))
            for fname in adapter_hashes
        }
        out["tokenizer_hashes"] = {
            "adapter": adapter_hashes,
            "base": base_hashes,
            "match": match_status,
        }
        if not all(match_status.values()):
            out["notes"].append("adapter ships tokenizer files that differ from base")

    # Compat verdict
    if out["targets_embeddings"] and out["vocab_dims_match"] is False:
        out["compat"] = False
        out["notes"].append("adapter tunes vocab-layers with wrong vocab_size — FAIL")
    elif out["tokenizer_hashes"] and not all(out["tokenizer_hashes"]["match"].values()):
        out["compat"] = "warn"
        out["notes"].append("tokenizer-file mismatch — WARN (may still work)")
    else:
        out["compat"] = True

    return out


# ----------------------------------------------------------------------
# 2. Base-checkpoint signature
# ----------------------------------------------------------------------

def check_base_signature(base_model_id: str, n_files_to_hash: int = 3) -> dict:
    """Compute a stable signature of the declared base checkpoint.

    Hashes config.json + the first N safetensors shards. Sufficient to
    uniquely identify the base across hub mirrors and detect silent
    upstream changes.
    """
    from huggingface_hub import HfApi, hf_hub_download

    out = {
        "base_model_id": base_model_id,
        "config_sha256": None,
        "shard_hashes": {},
        "signature": None,
        "notes": [],
    }

    try:
        p = hf_hub_download(base_model_id, "config.json")
        with open(p, "rb") as f:
            out["config_sha256"] = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        out["notes"].append(f"config hash skipped: {e}")

    # Hash first N model shards (covers single-file and sharded layouts)
    try:
        api = HfApi()
        files = [f for f in api.list_repo_files(base_model_id)
                 if f.endswith(".safetensors") and "adapter" not in f]
        files = sorted(files)[:n_files_to_hash]
        for fname in files:
            try:
                p = hf_hub_download(base_model_id, fname)
                h = hashlib.sha256()
                # Hash in chunks to avoid loading multi-GB into RAM
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                out["shard_hashes"][fname] = h.hexdigest()[:16]
            except Exception as e:
                out["notes"].append(f"shard {fname} skipped: {e}")
    except Exception as e:
        out["notes"].append(f"shard listing failed: {e}")

    # Composite signature: hash of config + sorted shard hashes
    combo = (out["config_sha256"] or "") + "|" + "|".join(
        f"{k}:{v}" for k, v in sorted(out["shard_hashes"].items())
    )
    out["signature"] = hashlib.sha256(combo.encode("utf-8")).hexdigest()[:16]
    return out


# ----------------------------------------------------------------------
# 3. Output divergence sanity test (requires GPU/torch)
# ----------------------------------------------------------------------

def check_output_divergence(
    adapter_id: str,
    base_model_id: str,
    prompts: list[str] = SANITY_PROMPTS,
    max_new_tokens: int = 32,
    cache_dir: str | None = None,
) -> dict:
    """Compare logits/generations of (base) vs (base + naive adapter delta).

    Returns:
      kl_first_token_mean: average KL divergence on the next-token
        distribution across the prompt set. Near-0 → adapter is a no-op.
        Very large → adapter is structurally incompatible (likely wrong base).
      top1_agreement_rate: fraction of prompts where the greedy next token
        is the same with and without the adapter. Near-1 → no-op.
      generated_text_similarity: mean character-level Jaccard similarity
        between greedy generations.

    All three signals are derived from the same forward passes, so this is
    one model load + 2 × N prompt evals.
    """
    import torch

    from utils import load_base_model, extract_lora_delta
    from merge import naive_merge

    out = {
        "adapter_id": adapter_id,
        "base_model_id": base_model_id,
        "n_prompts": len(prompts),
        "kl_first_token_mean": None,
        "top1_agreement_rate": None,
        "generated_text_similarity": None,
        "interpretation": None,
        "notes": [],
    }

    # Baseline forward pass
    try:
        base_model, tokenizer = load_base_model(base_model_id)
    except Exception as e:
        out["notes"].append(f"base load failed: {e}")
        return out

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(base_model.device)

    with torch.no_grad():
        base_logits = base_model(**inputs).logits[:, -1, :]    # (N, V)
        base_probs = torch.softmax(base_logits, dim=-1)
        base_gen = base_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        base_top1 = base_logits.argmax(dim=-1)

    # Apply adapter
    try:
        delta = extract_lora_delta(None, adapter_id)
        naive_merge(base_model, delta)
        del delta
        torch.cuda.empty_cache()
    except Exception as e:
        out["notes"].append(f"adapter apply failed: {e}")
        return out

    with torch.no_grad():
        adapted_logits = base_model(**inputs).logits[:, -1, :]
        adapted_probs = torch.softmax(adapted_logits, dim=-1)
        adapted_gen = base_model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False,
                                          pad_token_id=tokenizer.pad_token_id)
        adapted_top1 = adapted_logits.argmax(dim=-1)

    # KL(base || adapted) per prompt → mean
    eps = 1e-12
    kl = (base_probs * (torch.log(base_probs + eps) - torch.log(adapted_probs + eps))).sum(dim=-1)
    out["kl_first_token_mean"] = float(kl.mean().item())

    # Top-1 agreement
    out["top1_agreement_rate"] = float((base_top1 == adapted_top1).float().mean().item())

    # Char-level Jaccard on generated suffixes
    sims = []
    for i in range(len(prompts)):
        b = tokenizer.decode(base_gen[i, inputs["input_ids"][i].shape[0]:], skip_special_tokens=True)
        a = tokenizer.decode(adapted_gen[i, inputs["input_ids"][i].shape[0]:], skip_special_tokens=True)
        a_set, b_set = set(a), set(b)
        if a_set or b_set:
            sims.append(len(a_set & b_set) / max(1, len(a_set | b_set)))
        else:
            sims.append(1.0)
    out["generated_text_similarity"] = float(sum(sims) / len(sims))

    # Heuristic interpretation
    if out["kl_first_token_mean"] < 1e-3 and out["top1_agreement_rate"] > 0.95:
        out["interpretation"] = "no-op (adapter has negligible effect)"
    elif out["kl_first_token_mean"] > 20.0:
        out["interpretation"] = "extreme divergence (likely wrong base)"
    elif out["kl_first_token_mean"] > 5.0 and out["generated_text_similarity"] < 0.2:
        out["interpretation"] = "high divergence + low gen similarity (likely incompatible)"
    else:
        out["interpretation"] = "plausible adapter effect"

    del base_model
    torch.cuda.empty_cache()
    return out
