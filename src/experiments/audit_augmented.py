"""Run the augmented audit on the existing candidate pool.

Addresses reviewer Q3 + tech-limit-1: per-candidate runs the three new
checks from src/audit_checks.py and writes a combined audit JSON.

Two modes:
  --cpu-only   tokenizer + base-signature checks only (no GPU needed).
               Quick smoke pass; ~5-10s per adapter.
  (default)    plus output-divergence sanity test (~30s/adapter on GPU).

Per-adapter caching: each candidate's augmented audit is written
incrementally so the run is resumable.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, "src")
from utils import setup_logging, save_results, load_results
from audit_checks import (
    check_tokenizer_compat,
    check_base_signature,
    check_output_divergence,
)

logger = logging.getLogger(__name__)

POOL_PATH = "results/audit_pool.json"
OUT_DIR   = "results/audit_augmented"


def augment_one(adapter_id: str, base_model_id: str, gpu: bool) -> dict:
    out: dict = {"adapter_id": adapter_id, "base_model_id": base_model_id}

    logger.info(f"  [tokenizer_compat] ...")
    try:
        out["tokenizer_compat"] = check_tokenizer_compat(adapter_id, base_model_id)
    except Exception as e:
        out["tokenizer_compat"] = {"error": str(e)}

    if gpu:
        logger.info(f"  [output_divergence] ...")
        try:
            out["output_divergence"] = check_output_divergence(adapter_id, base_model_id)
        except Exception as e:
            out["output_divergence"] = {"error": str(e)}
            import traceback; traceback.print_exc()
    else:
        out["output_divergence"] = {"skipped": "cpu-only mode"}

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-only", action="store_true",
                   help="Skip the GPU-only output_divergence check.")
    p.add_argument("--family", choices=["llama", "mistral", "both"], default="both")
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    setup_logging("audit_augmented", OUT_DIR)

    if not os.path.exists(POOL_PATH):
        logger.error(f"Required: {POOL_PATH} (run src/run_audit_pool.py first)")
        sys.exit(1)

    pool = load_results(POOL_PATH)

    families = []
    if args.family in ("llama", "both") and "llama" in pool:
        families.append(("llama", pool["llama"]))
    if args.family in ("mistral", "both") and "mistral" in pool:
        families.append(("mistral", pool["mistral"]))

    # Compute base signatures once per family
    base_signatures = {}
    for fam_name, fam in families:
        base_id = fam["expected_base"]
        logger.info(f"\n=== {fam_name}: base_signature for {base_id} ===")
        try:
            base_signatures[fam_name] = check_base_signature(base_id)
            logger.info(f"  signature: {base_signatures[fam_name].get('signature')}")
        except Exception as e:
            base_signatures[fam_name] = {"error": str(e)}
            logger.error(f"  base signature failed: {e}")

    for fam_name, fam in families:
        base_id = fam["expected_base"]
        logger.info(f"\n{'='*70}\n[{fam_name}] augmented audit on {len(fam['results'])} adapters"
                    f"\n  base: {base_id}  mode: {'CPU+GPU' if not args.cpu_only else 'CPU-only'}"
                    f"\n{'='*70}")

        family_out_path = os.path.join(OUT_DIR, f"{fam_name}_augmented.json")
        existing = load_results(family_out_path) if os.path.exists(family_out_path) else {
            "expected_base": base_id,
            "base_signature": base_signatures[fam_name],
            "results": {},
        }
        # Refresh signature each run in case the user updated the base
        existing["base_signature"] = base_signatures[fam_name]

        for entry in fam["results"]:
            adapter_id = entry["id"]
            slot = entry.get("slot_label", "?")

            # Skip if previous audit couldn't even load the adapter config
            if "error" in entry:
                existing["results"][adapter_id] = {
                    "slot_label": slot,
                    "skipped": f"original audit error: {entry['error']}",
                }
                continue

            # Skip if cached for this mode
            cached = existing["results"].get(adapter_id)
            if cached and "tokenizer_compat" in cached:
                if args.cpu_only or "output_divergence" in cached and "skipped" not in cached["output_divergence"]:
                    logger.info(f"  [{slot:25s}] {adapter_id}  CACHED")
                    continue

            logger.info(f"\n[{slot:25s}] {adapter_id}")
            aug = augment_one(adapter_id, base_id, gpu=not args.cpu_only)
            aug["slot_label"] = slot
            aug["metadata_audit"] = {
                "declared_base": entry.get("declared_base"),
                "base_match":    entry.get("base_match"),
                "total_norm":    entry.get("total_norm"),
                "norm_pass":     entry.get("norm_pass"),
                "verdict":       entry.get("verdict"),
            }

            # Compose a content-aware verdict
            metadata_verdict = entry.get("verdict")
            tc = aug.get("tokenizer_compat", {})
            od = aug.get("output_divergence", {})

            content_flags = []
            if tc.get("compat") is False:
                content_flags.append("tokenizer_incompat")
            if od.get("interpretation") and "no-op" in od["interpretation"]:
                content_flags.append("content_noop")
            if od.get("interpretation") and ("wrong base" in od["interpretation"]
                                              or "incompatible" in od["interpretation"]):
                content_flags.append("content_incompat")

            if metadata_verdict == "PASS" and not content_flags:
                aug["augmented_verdict"] = "PASS"
            elif content_flags:
                aug["augmented_verdict"] = f"FAIL (content: {','.join(content_flags)})"
            else:
                aug["augmented_verdict"] = f"FAIL (metadata: {metadata_verdict})"

            existing["results"][adapter_id] = aug
            save_results(existing, family_out_path)

            # Quick print
            kl  = od.get("kl_first_token_mean")
            agr = od.get("top1_agreement_rate")
            interp = od.get("interpretation", "—")
            logger.info(
                f"  metadata: {metadata_verdict}  tokenizer: {tc.get('compat')}  "
                f"divergence: KL={kl if kl is None else f'{kl:.3f}'}  "
                f"top1agree={agr if agr is None else f'{agr:.2f}'}  → {interp}"
            )
            logger.info(f"  augmented_verdict: {aug['augmented_verdict']}")

        # Family summary
        results = existing["results"]
        n_pass = sum(1 for v in results.values()
                     if isinstance(v, dict) and v.get("augmented_verdict") == "PASS")
        n_fail = sum(1 for v in results.values()
                     if isinstance(v, dict) and v.get("augmented_verdict", "").startswith("FAIL"))
        n_skip = sum(1 for v in results.values()
                     if isinstance(v, dict) and "skipped" in v)
        logger.info(f"\n[{fam_name}] PASS={n_pass}  FAIL={n_fail}  SKIP={n_skip}  "
                    f"total={len(results)}")

    logger.info(f"\nOutput: {OUT_DIR}/")


if __name__ == "__main__":
    main()
