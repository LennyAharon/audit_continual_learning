"""HumanEval with sampling for bootstrap CIs.
Use 10 repeats at temp=0.2 -- gives pass@1, pass@5, pass@10 with variance.
Only 10*164 = 1640 generations per config, ~2 hours per config, ~8 hours total.
"""

import gc, os, sys, json, math, yaml
import torch

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

sys.path.insert(0, 'src')
from utils import load_config, load_base_model, extract_lora_delta, save_results, load_results
from gf_ewc import gf_ewc_merge, compute_block_scales
from merge import naive_merge, task_arithmetic_merge, ties_merge

RESULTS_DIR = "results"

sys.modules.pop('evaluate', None)
sys.path = [p for p in sys.path if 'audit_continual_learning/src' not in p and p != 'src']
import lm_eval
from lm_eval.models.huggingface import HFLM

# Create a custom 10-repeat humaneval task config
# We put this in a temp yaml and point lm-eval to it
TASK_DIR = os.path.join(os.path.dirname(lm_eval.__file__), 'tasks', 'humaneval')


def ensure_humaneval_10_task():
    """Create humaneval_10 task file if not present."""
    task_path = os.path.join(TASK_DIR, 'humaneval_10.yaml')
    if not os.path.exists(task_path):
        content = """include: humaneval.yaml
task: humaneval_10
repeats: 10
metric_list:
  - metric: !function utils.pass_at_k
    aggregation: mean
    higher_is_better: true
    k: [1, 5, 10]
generation_kwargs:
  until:
    - "\\nclass"
    - "\\ndef"
    - "\\n#"
    - "\\nif"
    - "\\nprint"
  max_gen_toks: 1024
  do_sample: true
  temperature: 0.2
  top_p: 0.95
"""
        with open(task_path, 'w') as f:
            f.write(content)
        print(f"Created {task_path}")


def evaluate_humaneval_passk(model, tokenizer):
    """HumanEval with 10 samples/problem at temp=0.2 -- gives pass@1,5,10."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=4)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["humaneval_10"],
        confirm_run_unsafe_code=True,
    )
    return results["results"]["humaneval_10"]


def gfewc_ties_merge(model, lora_delta, importance_scores, alpha, density=0.5):
    scales = compute_block_scales(importance_scores, alpha, "linear")
    trimmed = {}
    for key, delta in lora_delta.items():
        flat = delta.flatten().abs().float()
        k = max(1, int(density * flat.numel()))
        threshold = flat.kthvalue(k).values.item()
        mask = delta.abs() >= threshold
        trimmed[key] = delta * mask
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = name.replace(".weight", "")
            if key in trimmed:
                scale = scales.get(key, 1.0)
                d = trimmed[key].to(param.device, dtype=param.dtype)
                param.data.add_(scale * d)


def run_config(config, merge_spec, run_id):
    result_path = os.path.join(RESULTS_DIR, f"passk_{run_id}.json")
    if os.path.exists(result_path):
        existing = load_results(result_path)
        if 'humaneval_results' in existing:
            print(f"Skipping {run_id} (exists)")
            return existing

    print(f"\n{'='*60}\n{run_id}\n{'='*60}")

    short_name = config["base_models"][0]["short_name"]
    model_name = config["base_models"][0]["name"]
    ordering = config["merge_orderings"][0]
    adapters = config["lora_adapters"][short_name]
    adapter_by_task = {a["task"]: a for a in adapters}

    model, tokenizer = load_base_model(model_name)

    if merge_spec is not None:
        for task_name in ordering:
            delta = extract_lora_delta(model, adapter_by_task[task_name]["id"])
            merge_spec['fn'](model, delta, **merge_spec.get('kwargs', {}))
            del delta; gc.collect()

    print("  HumanEval pass@{1,5,10} (10 samples/problem, temp=0.2)...")
    try:
        he_results = evaluate_humaneval_passk(model, tokenizer)
        print(f"    {he_results}")
        results = {"run_id": run_id, "humaneval_results": he_results}
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        results = {"run_id": run_id, "error": str(e)}

    del model; torch.cuda.empty_cache(); gc.collect()
    save_results(results, result_path)
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = load_config("configs/experiment_config.yaml")
    importance_scores = load_results("results/task_importance_layers.json")

    ensure_humaneval_10_task()

    configs = [
        ("baseline", None),
        ("naive", {"fn": naive_merge}),
        ("gfewc_alpha5", {"fn": lambda m, d: gf_ewc_merge(m, d, importance_scores, alpha=5.0, scaling_fn="linear")}),
        ("gfewc_ties_alpha3", {"fn": lambda m, d: gfewc_ties_merge(m, d, importance_scores, alpha=3.0, density=0.5)}),
    ]

    for run_id, merge_spec in configs:
        run_config(config, merge_spec, run_id)


if __name__ == "__main__":
    main()
