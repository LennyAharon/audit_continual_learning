"""Real prevalence estimate: random-sample from HuggingFace Hub.

Reviewer Q5 asked for prevalence rates on a 20-50 adapter pool drawn
from "diverse hubs". Our existing pool was hand-curated by the paper
author — half PASS, half FAIL by construction — so it does NOT support
a prevalence claim. This script replaces it.

Method:
  1. Query HF Hub via `api.list_models(search=...)` for adapters
     matching `lora <base>` queries.
  2. Random-shuffle the returned list with a fixed seed for
     reproducibility.
  3. Filter out adapters already in the curated pool (so we count
     new evidence, not duplicates).
  4. Take the first N unique IDs (default N=40 per family).
  5. For each, run the existing audit_one (metadata + norm check).
  6. Report prevalence rates: P(loadable | listed),
     P(base_mismatch | loadable), P(near_zero_norm | loadable),
     P(audit_pass | loadable).

The sample is biased toward HF's default ordering (likes/relevance),
not strictly random over all Llama-3.1-8B LoRAs ever uploaded — but
it's the same bias a user would encounter when picking adapters by
hand, and is FAR more representative than the curated pool.

Output:
  results/audit_random_sample.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
from run_audit_pool import audit_one  # reuse the metadata+norm audit

OUT_PATH = "results/audit_random_sample.json"
CURATED_POOL_PATH = "configs/audit_candidates.json"


# Queries chosen to cover the natural "I want a Llama-3.1 LoRA" searches.
# Each query returns its own ranked list; we union + dedupe + shuffle.
LLAMA_QUERIES = [
    "lora llama-3.1-8b",
    "lora llama 3.1 8b",
    "lora Llama-3.1-8B-Instruct",
    "peft llama 3.1",
]
MISTRAL_QUERIES = [
    "lora mistral 7b",
    "lora Mistral-7B-Instruct-v0.3",
    "lora mistral-7b-instruct",
]


def _curated_ids() -> set[str]:
    """Return all adapter IDs already in our hand-curated pool."""
    with open(CURATED_POOL_PATH) as f:
        cfg = json.load(f)
    out = set()
    for fam in ("llama", "mistral"):
        if fam in cfg:
            for c in cfg[fam].get("candidates", []):
                if "id" in c and "PLACEHOLDER" not in c["id"]:
                    out.add(c["id"])
    return out


def search_candidates(queries: list[str], limit_per_query: int = 80) -> list[str]:
    """Run HF search queries; return union of IDs in result order."""
    from huggingface_hub import HfApi
    api = HfApi()
    seen, ordered = set(), []
    for q in queries:
        try:
            hits = api.list_models(search=q, limit=limit_per_query)
            for m in hits:
                mid = m.modelId
                if mid in seen:
                    continue
                seen.add(mid)
                ordered.append(mid)
        except Exception as e:
            print(f"  WARN: query {q!r} failed: {e}")
    return ordered


def sample_and_audit(
    family_name: str,
    queries: list[str],
    expected_base: str,
    n_to_audit: int,
    seed: int,
    skip_ids: set[str],
) -> dict:
    print(f"\n{'='*70}\n{family_name}: querying HF Hub for candidates\n  queries: {queries}\n{'='*70}")
    all_ids = search_candidates(queries)
    print(f"  found {len(all_ids)} unique adapter IDs across queries")

    # Filter out the curated pool so we count NEW evidence
    fresh = [mid for mid in all_ids if mid not in skip_ids]
    print(f"  {len(all_ids) - len(fresh)} already in curated pool; {len(fresh)} new")

    rng = random.Random(seed)
    rng.shuffle(fresh)
    sample = fresh[:n_to_audit]
    print(f"  random-sampled (seed={seed}): {len(sample)} adapters to audit\n")

    results = []
    for i, mid in enumerate(sample):
        print(f"  [{i+1:2}/{len(sample)}] {mid}")
        try:
            r = audit_one(mid, expected_base)
        except Exception as e:
            r = {"id": mid, "error": f"audit_one raised: {e}"}
        results.append(r)
        v = r.get("verdict", "ERROR")
        if "error" in r:
            print(f"       ERROR: {r['error'][:80]}")
        else:
            base = "OK" if r["base_match"] else "MISMATCH"
            print(f"       base={base}  norm={r['total_norm']}  {v}")

    # Prevalence rates
    n_total      = len(results)
    n_loadable   = sum(1 for r in results if "error" not in r)
    n_pass       = sum(1 for r in results if r.get("verdict") == "PASS")
    n_fail       = sum(1 for r in results if r.get("verdict") == "FAIL")
    n_basemismatch = sum(1 for r in results
                         if "error" not in r and not r.get("base_match", True))
    n_normfail   = sum(1 for r in results
                       if "error" not in r and not r.get("norm_pass", True))
    n_nearzero   = sum(1 for r in results
                       if "error" not in r and r.get("total_norm", 0) < 0.1)

    summary = {
        "family":         family_name,
        "expected_base":  expected_base,
        "queries":        queries,
        "seed":           seed,
        "search_total":   len(all_ids),
        "fresh_total":    len(fresh),
        "n_audited":      n_total,
        "n_loadable":     n_loadable,
        "n_pass":         n_pass,
        "n_fail":         n_fail,
        "n_base_mismatch": n_basemismatch,
        "n_norm_fail":    n_normfail,
        "n_near_zero":    n_nearzero,
        "rates_pct": {
            "loadable":        100 * n_loadable / max(n_total, 1),
            "pass":            100 * n_pass / max(n_total, 1),
            "fail":            100 * n_fail / max(n_total, 1),
            "base_mismatch_of_loadable": (100 * n_basemismatch / n_loadable) if n_loadable else None,
            "norm_fail_of_loadable":     (100 * n_normfail / n_loadable) if n_loadable else None,
            "near_zero_of_loadable":     (100 * n_nearzero / n_loadable) if n_loadable else None,
        },
        "results":        results,
    }

    print(f"\n--- {family_name} prevalence (n_audited={n_total}) ---")
    print(f"  loadable:           {n_loadable}/{n_total}  ({summary['rates_pct']['loadable']:.1f}%)")
    print(f"  audit PASS:         {n_pass}/{n_total}  ({summary['rates_pct']['pass']:.1f}%)")
    print(f"  audit FAIL:         {n_fail}/{n_total}  ({summary['rates_pct']['fail']:.1f}%)")
    if n_loadable:
        print(f"  base mismatch:      {n_basemismatch}/{n_loadable}  "
              f"({summary['rates_pct']['base_mismatch_of_loadable']:.1f}% of loadable)")
        print(f"  norm fail (<1.0):   {n_normfail}/{n_loadable}  "
              f"({summary['rates_pct']['norm_fail_of_loadable']:.1f}% of loadable)")
        print(f"  near-zero (<0.1):   {n_nearzero}/{n_loadable}  "
              f"({summary['rates_pct']['near_zero_of_loadable']:.1f}% of loadable)")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-family", type=int, default=40,
                   help="random-sample size per base-model family")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--family", choices=["llama", "mistral", "both"], default="both")
    args = p.parse_args()

    skip_ids = _curated_ids()
    print(f"Skip-list (already in curated pool): {len(skip_ids)} adapters")

    out = {
        "n_per_family": args.n_per_family,
        "seed": args.seed,
        "skip_list_size": len(skip_ids),
        "families": {},
    }

    if args.family in ("llama", "both"):
        out["families"]["llama"] = sample_and_audit(
            "llama", LLAMA_QUERIES,
            expected_base="NousResearch/Meta-Llama-3.1-8B-Instruct",
            n_to_audit=args.n_per_family, seed=args.seed, skip_ids=skip_ids,
        )

    if args.family in ("mistral", "both"):
        out["families"]["mistral"] = sample_and_audit(
            "mistral", MISTRAL_QUERIES,
            expected_base="mistralai/Mistral-7B-Instruct-v0.3",
            n_to_audit=args.n_per_family, seed=args.seed, skip_ids=skip_ids,
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")

    # Combined-pool summary line for the rebuttal text
    if len(out["families"]) == 2:
        all_loadable = sum(out["families"][f]["n_loadable"] for f in out["families"])
        all_pass     = sum(out["families"][f]["n_pass"] for f in out["families"])
        all_basem    = sum(out["families"][f]["n_base_mismatch"] for f in out["families"])
        all_normf    = sum(out["families"][f]["n_norm_fail"] for f in out["families"])
        print(f"\n=== Combined prevalence (random sample, both families) ===")
        print(f"  loadable adapters audited:  {all_loadable}")
        print(f"  PASS rate:                  {all_pass}/{all_loadable} "
              f"({100*all_pass/max(all_loadable,1):.1f}%)")
        print(f"  base-mismatch rate:         {all_basem}/{all_loadable} "
              f"({100*all_basem/max(all_loadable,1):.1f}%)")
        print(f"  norm-fail rate:             {all_normf}/{all_loadable} "
              f"({100*all_normf/max(all_loadable,1):.1f}%)")


if __name__ == "__main__":
    main()
