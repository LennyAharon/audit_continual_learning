"""LoRA merge strategies used in the paper: naive, Task Arithmetic, TIES, DARE, MagMax.

Each function mutates the model's weights in place.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def naive_merge(model, lora_delta: dict) -> None:
    """Naive merge: directly add LoRA delta to base model weights in-place."""
    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in lora_delta:
                delta = lora_delta[key].to(param.device, dtype=param.dtype)
                param.data.add_(delta)
                applied += 1
    logger.info(f"Naive merge: applied delta to {applied} layers")


def task_arithmetic_merge(model, lora_delta: dict, lambda_val: float) -> None:
    """Task Arithmetic merge: add scaled LoRA delta to base model.

    Reference: Ilharco et al., "Editing Models with Task Arithmetic" (ICLR 2023).
    """
    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in lora_delta:
                delta = lora_delta[key].to(param.device, dtype=param.dtype)
                param.data.add_(lambda_val * delta)
                applied += 1
    logger.info(f"Task arithmetic merge (lambda={lambda_val}): applied to {applied} layers")


def ties_merge(model, lora_delta: dict, density: float = 0.5) -> None:
    """TIES merge: trim, resolve sign conflicts, then apply.

    Reference: Yadav et al., "Resolving Interference When Merging Models" (NeurIPS 2023).
    """
    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in lora_delta:
                delta = lora_delta[key].clone()

                # Step 1: Trim - keep only top-density fraction by magnitude
                flat = delta.flatten()
                num_keep = max(1, int(density * flat.numel()))
                threshold = flat.abs().topk(num_keep).values[-1]
                mask = delta.abs() >= threshold
                delta = delta * mask.float()

                # Step 2: Resolve sign conflicts (majority-vote on signed mass)
                nonzero_mask = delta != 0
                if nonzero_mask.any():
                    positive_mass = (delta * (delta > 0).float()).sum()
                    negative_mass = (delta * (delta < 0).float()).sum().abs()
                    majority_sign = 1.0 if positive_mass >= negative_mass else -1.0

                    if majority_sign > 0:
                        delta = delta * (delta >= 0).float()
                    else:
                        delta = delta * (delta <= 0).float()

                delta = delta.to(param.device, dtype=param.dtype)
                param.data.add_(delta)
                applied += 1

    logger.info(f"TIES merge (density={density}): applied to {applied} layers")


def dare_merge(model, lora_delta: dict, density: float = 0.5) -> None:
    """DARE merge: random drop + rescale, then apply.

    Reference: Yu et al., "Language Models are Super Mario" (ICML 2024).
    """
    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in lora_delta:
                delta = lora_delta[key].clone()

                drop_mask = torch.bernoulli(
                    torch.full_like(delta, density)
                ).bool()
                delta = delta * drop_mask.float() / density

                delta = delta.to(param.device, dtype=param.dtype)
                param.data.add_(delta)
                applied += 1

    logger.info(f"DARE merge (density={density}): applied to {applied} layers")


def magmax_merge(model, lora_delta: dict) -> None:
    """MagMax merge: element-wise keep whichever has larger magnitude (merged vs base).

    Reference: Marczak et al., "MagMax: Leveraging Model Merging for Seamless
    Continual Learning" (ECCV 2024).
    """
    applied = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in lora_delta:
                d = lora_delta[key].to(param.device, dtype=param.dtype)
                merged = param.data + d
                mask = merged.abs() >= param.data.abs()
                param.data = torch.where(mask, merged, param.data)
                applied += 1
    logger.info(f"MagMax merge: applied to {applied} layers")


def apply_merge(model, lora_delta: dict, method: str, params: dict | None = None) -> None:
    """Dispatch to the appropriate merge method.

    Args:
        model: Base model to modify in-place.
        lora_delta: Dict mapping layer-name (without ``.weight``) to delta tensor.
        method: One of 'naive', 'task_arithmetic', 'ties', 'dare', 'magmax'.
        params: Method-specific parameters (``lambda_val`` for TA, ``density`` for TIES/DARE).
    """
    params = params or {}
    if method == "naive":
        naive_merge(model, lora_delta)
    elif method == "task_arithmetic":
        task_arithmetic_merge(model, lora_delta, lambda_val=params.get("lambda_val", 0.5))
    elif method == "ties":
        ties_merge(model, lora_delta, density=params.get("density", 0.5))
    elif method == "dare":
        dare_merge(model, lora_delta, density=params.get("density", 0.5))
    elif method == "magmax":
        magmax_merge(model, lora_delta)
    else:
        raise ValueError(f"Unknown merge method: {method}")
