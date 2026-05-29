"""Utility functions for model loading, weight extraction, and logging."""

import json
import logging
import os
from pathlib import Path
from copy import deepcopy

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_base_model(model_name: str, quantize_4bit: bool = False):
    """Load a base instruct model and tokenizer.

    Returns:
        (model, tokenizer) tuple
    """
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # required for decoder-only generation

    model.eval()
    return model, tokenizer


def load_lora_adapter(base_model, adapter_id: str):
    """Load a LoRA adapter onto a base model using PEFT.

    Returns:
        PeftModel with adapter loaded
    """
    peft_model = PeftModel.from_pretrained(base_model, adapter_id)
    return peft_model


def extract_lora_delta(base_model_or_name, adapter_id: str) -> dict:
    """Extract the effective weight delta from a LoRA adapter.

    Loads adapter weights directly from safetensors — no PeftModel needed,
    uses ~100MB RAM instead of ~16GB. Computes delta = lora_B @ lora_A * scaling.

    Args:
        base_model_or_name: Accepted for API compat but not used.
        adapter_id: HuggingFace adapter ID.

    Returns:
        Dict of {target_layer_name: delta_tensor}
    """
    import math
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    # Load adapter config
    adapter_config = PeftConfig.from_pretrained(adapter_id)
    lora_alpha = adapter_config.lora_alpha
    lora_r = adapter_config.r

    use_rslora = getattr(adapter_config, "use_rslora", False)
    scaling = lora_alpha / math.sqrt(lora_r) if use_rslora else lora_alpha / lora_r

    # Download and load adapter weights directly from safetensors
    adapter_path = hf_hub_download(adapter_id, "adapter_model.safetensors")
    adapter_weights = load_file(adapter_path)

    # Group lora_A and lora_B weights by their target layer
    lora_pairs = {}
    for key, tensor in adapter_weights.items():
        if "lora_A" in key:
            base_key = key.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
            base_key = base_key.replace("base_model.model.", "")
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif "lora_B" in key:
            base_key = key.replace(".lora_B.default.weight", "").replace(".lora_B.weight", "")
            base_key = base_key.replace("base_model.model.", "")
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    # Compute deltas: delta = lora_B @ lora_A * scaling
    deltas = {}
    with torch.no_grad():
        for layer_name, pair in lora_pairs.items():
            if "A" in pair and "B" in pair:
                lora_a = pair["A"].float()   # (r, in_features)
                lora_b = pair["B"].float()   # (out_features, r)
                delta = (lora_b @ lora_a) * scaling
                deltas[layer_name] = delta.to(torch.bfloat16).cpu()

    del adapter_weights
    return deltas


def get_layer_names(model) -> list:
    """Return ordered list of layer names that have weight matrices.

    Returns names like 'model.layers.0.self_attn.q_proj', etc.
    """
    layer_names = []
    for name, param in model.named_parameters():
        if "weight" in name and any(
            proj in name
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ):
            layer_names.append(name.replace(".weight", ""))
    return sorted(layer_names)


def extract_weight_snapshot(model) -> dict:
    """Extract a snapshot of all weight matrices for comparison.

    Returns dict of {layer_name: weight_tensor_clone}.
    """
    snapshot = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and any(
                proj in name
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ):
                key = name.replace(".weight", "")
                snapshot[key] = param.data.clone().cpu()
    return snapshot


def apply_delta_to_model(model, deltas: dict):
    """Apply weight deltas directly to a model's parameters in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in deltas:
                delta = deltas[key].to(param.device, dtype=param.dtype)
                param.data.add_(delta)


def save_results(results_dict: dict, path: str):
    """Save experiment results as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    logging.info(f"Results saved to {path}")


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)


def setup_logging(experiment_name: str, results_dir: str = "results"):
    """Set up logging to both console and file."""
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"{experiment_name}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging to {log_path}")
