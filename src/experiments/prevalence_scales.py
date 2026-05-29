"""Prevalence audit across MODEL SCALES (reviewer: 'greater range of scales').

Both reviewers asked whether the audit's findings generalize beyond the
~7-8B scale. This script runs the same random-sample prevalence audit
(metadata + reconstructed-norm checks) on smaller-scale model families,
so we can report whether base-mismatch and near-zero-norm failure modes
appear at 1B / 3B scale too.

CPU + download only (the metadata+norm audit needs no GPU). Reuses the
random-sample machinery from run_audit_random_sample.py.

Output:
  results/audit_scales.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "src")
from run_audit_random_sample import sample_and_audit, _curated_ids

OUT_PATH = "results/audit_scales.json"

# Family -> (expected_base, queries). Chosen to span 1B / 3B / 8B Llama
# plus a non-Llama point (Qwen2.5-7B) for architecture diversity.
SCALE_FAMILIES = {
    "llama32_1b": (
        "meta-llama/Llama-3.2-1B-Instruct",
        ["lora llama-3.2-1b", "lora Llama-3.2-1B-Instruct", "peft llama 3.2 1b"],
    ),
    "llama32_3b": (
        "meta-llama/Llama-3.2-3B-Instruct",
        ["lora llama-3.2-3b", "lora Llama-3.2-3B-Instruct", "peft llama 3.2 3b"],
    ),
    "qwen25_7b": (
        "Qwen/Qwen2.5-7B-Instruct",
        ["lora qwen2.5-7b", "lora Qwen2.5-7B-Instruct", "peft qwen2.5 7b"],
    ),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-family", type=int, default=40)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--families", nargs="+", default=list(SCALE_FAMILIES),
                   choices=list(SCALE_FAMILIES))
    args = p.parse_args()

    skip_ids = _curated_ids()
    out = {"n_per_family": args.n_per_family, "seed": args.seed, "families": {}}

    for fam in args.families:
        base, queries = SCALE_FAMILIES[fam]
        out["families"][fam] = sample_and_audit(
            fam, queries, expected_base=base,
            n_to_audit=args.n_per_family, seed=args.seed, skip_ids=skip_ids,
        )
        # Save incrementally so a long download run is resumable-ish
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=2)

    # Cross-scale summary
    print("\n" + "=" * 70)
    print("CROSS-SCALE PREVALENCE SUMMARY")
    print("=" * 70)
    print(f"{'family':14s} {'loadable':>10s} {'pass':>8s} {'base-mism%':>12s} {'nearzero%':>11s}")
    for fam, body in out["families"].items():
        pv = body["prevalence"]
        rp = pv["rates_pct"]
        bm = f"{rp['base_mismatch_of_loadable']:.0f}" if rp['base_mismatch_of_loadable'] is not None else "—"
        nz = f"{rp['near_zero_of_loadable']:.0f}" if rp['near_zero_of_loadable'] is not None else "—"
        print(f"{fam:14s} {pv['n_loadable']:>4}/{pv['n_audited']:<5} "
              f"{pv['n_pass']:>8} {bm:>12} {nz:>11}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
