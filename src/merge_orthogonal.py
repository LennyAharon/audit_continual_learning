"""Orthogonal-projection merger — a representative structure-aware baseline.

Addresses reviewer gap-3 (partial): adds one LoRA-aware merger to the
comparison family. Inspired by OSRM (Orthogonal Subspace Routing for
Merging) — for each layer, the t-th adapter's delta is projected onto
the subspace orthogonal to the span of the previously-applied deltas
before being added. This avoids interference between adapters that
target overlapping low-rank directions.

This is NOT a faithful reproduction of OSRM, SA-LoRA, LoRA-LEGO, or
LoRI. It is a controlled implementation of the family's central idea
(orthogonalize before merge) using only public LoRA factorizations.
Findings are presented as evidence that "the audit alters the ranking
among structure-aware mergers in the same way it alters TA/TIES" — the
specific question the reviewer raised.

Faithful reproductions are deferred to future work; this merger establishes
the comparison shape.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def orthogonal_project_merge(
    model,
    lora_delta: dict,
    *,
    cumulative_basis: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Orthogonalize new delta against previously-merged deltas, then add.

    Tracks a cumulative orthonormal basis ``U_cum`` per layer of shape
    ``(out_dim, k)`` where ``k`` is the cumulative effective rank.
    This is the column space of all previously-applied deltas.

    Per step::

        delta_orth = delta_new - U_cum @ (U_cum^T @ delta_new)
        param += delta_orth                                            # apply
        U_d  = thin-SVD(delta_orth, keep significant singular vectors) # new basis
        U_cum_new = [U_cum | U_d]                                       # concat

    Math is equivalent to stacking every prior delta and SVD'ing the
    full stack, but storage is only ``out_dim * cumulative_rank``
    instead of ``out_dim * in_dim * T``. For a 4-adapter merge with
    rank-64 adapters this is ~500MB total instead of ~30GB.

    The cumulative basis stays on CPU in fp32 between calls; it's
    materialised to GPU only for the per-layer projection step.
    """
    if cumulative_basis is None:
        cumulative_basis = {}

    applied = 0
    orthogonalised = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key not in lora_delta:
                continue
            delta_new = lora_delta[key].to(param.device, dtype=torch.float32)

            U_cum = cumulative_basis.get(key)
            if U_cum is None:
                # First merge for this layer: no orthogonalisation to do
                delta_orth = delta_new
            else:
                try:
                    U_cum_gpu = U_cum.to(param.device, dtype=torch.float32)
                    proj = U_cum_gpu @ (U_cum_gpu.T @ delta_new)
                    delta_orth = delta_new - proj
                    orthogonalised += 1
                    del proj, U_cum_gpu
                except Exception as e:
                    logger.warning(f"  proj failed on {key}: {e}; falling back to naive add")
                    delta_orth = delta_new

            # Apply to model
            param.data.add_(delta_orth.to(param.dtype))

            # Update cumulative basis: thin-SVD of delta_orth -> new directions
            try:
                U_d, S_d, _ = torch.linalg.svd(delta_orth, full_matrices=False)
                tol = max(delta_orth.shape) * S_d.max().item() * 1e-7
                keep = (S_d > tol).sum().item()
                U_d_eff = U_d[:, :keep].contiguous()  # (out_dim, k_new)
            except Exception as e:
                logger.warning(f"  basis-SVD failed on {key}: {e}; basis unchanged")
                U_d_eff = None
                U_d = S_d = None  # for cleanup below

            if U_d_eff is not None:
                if U_cum is None:
                    new_basis = U_d_eff.detach().to("cpu", dtype=torch.float32)
                else:
                    new_basis = torch.cat([
                        U_cum,
                        U_d_eff.detach().to("cpu", dtype=torch.float32),
                    ], dim=1)
                cumulative_basis[key] = new_basis

            # Free per-layer GPU memory
            del delta_new, delta_orth
            if U_d is not None: del U_d, S_d
            if U_d_eff is not None: del U_d_eff
            applied += 1

        torch.cuda.empty_cache()

    logger.info(f"  orthogonal-projection merge: applied to {applied} layers "
                f"({orthogonalised} had previous-history projection)")
    return cumulative_basis
