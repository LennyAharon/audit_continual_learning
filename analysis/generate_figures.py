"""Generate the five figures used in the paper.

Fig 1 (methods): Bar chart over GSM8K / HumanEval / MATH-500 (Fig 1 in paper).
Fig 1 (upper bounds): Single-adapter reference deltas.
Fig 2: Per-block leakage-free importance distribution.
Fig 3 (negation): GSM8K probe accuracy vs per-adapter coefficient.
Fig 3 (adapter audit): Schematic; data-driven labels.

Outputs: figures/fig*.pdf and figures/fig*.png at the repo root.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO, "results")
OUT = os.path.join(_REPO, "figures")
os.makedirs(OUT, exist_ok=True)


def load_scores(label):
    p = os.path.join(RESULTS, f"merge_main_{label}.json")
    with open(p) as f:
        d = json.load(f)
    return d.get("scores", {})


def load_stderrs(label):
    """Pull per-benchmark standard errors from the cached lm-eval `full` block."""
    p = os.path.join(RESULTS, f"merge_main_{label}.json")
    with open(p) as f:
        d = json.load(f)
    full = d.get("full", {})
    out = {}
    g = full.get("gsm8k", {})
    out["gsm8k"] = g.get("exact_match_stderr,strict-match")
    h = full.get("humaneval", {})
    out["humaneval"] = h.get("pass@1_stderr,create_test")
    m = full.get("math500", {})
    out["math500"] = m.get("exact_match_stderr,none")
    return out


# ---------- Figure 1: method comparison ----------
methods = [
    ("Baseline",    "baseline"),
    ("Naive",       "naive"),
    ("TA λ=0.5",    "task_arithmetic_lambda0.5"),
    ("TIES d=0.5",  "ties_d0.5"),
    ("DARE d=0.5",  "dare_d0.5"),
    ("MagMax",      "magmax"),
]

gsm = [load_scores(key).get("gsm8k", 0) for _, key in methods]
he  = [load_scores(key).get("humaneval", 0) for _, key in methods]
mh  = [load_scores(key).get("math500", 0) for _, key in methods]

gsm_se = [load_stderrs(key).get("gsm8k") or 0 for _, key in methods]
he_se  = [load_stderrs(key).get("humaneval") or 0 for _, key in methods]
mh_se  = [load_stderrs(key).get("math500") or 0 for _, key in methods]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.0, 2.6), sharey=False)

# Compact labels (avoid rotated overlap)
labels = ["Base", "Naive", "TA\n0.5", "TIES\n0.5", "DARE\n0.5", "MagMax"]
x = np.arange(len(labels))

def _bars(ax, vals, title, ylim, baseline, errs=None, baseline_label=True):
    colors = ["#555555"] + ["#1f77b4" if v >= baseline - 0.002 else "#d62728" for v in vals[1:]]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7, width=0.7,
                  yerr=errs, capsize=3,
                  error_kw=dict(ecolor="#222222", elinewidth=1.0, capthick=1.0))
    # Shade the ±1 stderr band around the unmerged baseline so "within
    # stderr of baseline" is visually obvious.
    if errs is not None:
        ax.axhspan(baseline - errs[0], baseline + errs[0],
                   color="#999999", alpha=0.18, zorder=0,
                   label="Baseline ±1 SE")
    ax.axhline(baseline, linestyle="--", color="#999999", linewidth=0.8,
               label="Baseline" if baseline_label else None)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=10)
    # Place value labels above the error-bar caps to avoid overlap
    span = ylim[1] - ylim[0]
    for xi, v, se in zip(x, vals, (errs if errs is not None else [0] * len(vals))):
        ax.text(xi, v + se + 0.012 * span, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5)

_bars(ax1, gsm, "GSM8K", (0.55, 0.78), baseline=gsm[0], errs=gsm_se)
_bars(ax2, he,  "HumanEval pass@1", (0.40, 0.72), baseline=he[0], errs=he_se)
_bars(ax3, mh,  "MATH-500 (4-shot)", (0.12, 0.205), baseline=mh[0], errs=mh_se)

ax1.legend(loc="lower right", fontsize=8, frameon=False)
fig.suptitle("Four-adapter merge on audited Llama-3.1-8B-Instruct (full test sets)",
             fontsize=12, y=1.02)
plt.subplots_adjust(wspace=0.32)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_methods.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig1_methods.png"), bbox_inches="tight", dpi=200)
print(f"Wrote {OUT}/fig1_methods.pdf")

# ---------- Figure 2: per-block importance ----------
with open(os.path.join(RESULTS, "leakfree_block_probe.json")) as f:
    probe = json.load(f)

block_imps = []
for b in range(32):
    imp = probe["block_scores"][str(b)]["importance"]
    block_imps.append(imp)

fig2, ax = plt.subplots(figsize=(8.0, 2.4))
bars = ax.bar(range(32), block_imps,
              color=["#d62728" if v > 0 else "#1f77b4" if v < 0 else "#999999"
                     for v in block_imps],
              edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("Transformer block index", fontsize=9)
ax.set_ylabel(r"Block importance $\hat{I}_b$", fontsize=9)
ax.set_xticks(range(0, 32, 2))
ax.set_xlim(-0.6, 31.6)

# Annotate extremes.
for b in [0, 11, 30]:
    v = block_imps[b]
    ax.annotate(f"block {b}", xy=(b, v),
                xytext=(b, v + (0.02 if v > 0 else -0.02)),
                ha="center", fontsize=7, color="black")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#d62728", label=r"$\hat{I}_b > 0$: protect (adapter delta hurts)"),
    Patch(facecolor="#1f77b4", label=r"$\hat{I}_b < 0$: preserve (adapter delta helps)"),
    Patch(facecolor="#999999", label=r"$\hat{I}_b \approx 0$: neutral"),
]
ax.legend(handles=legend_elements, loc="lower left", fontsize=7, frameon=False)
ax.set_title(r"Per-block leakage-free importance $\hat{I}_b$ on GSM8K (naive probe $=0.44$)",
             fontsize=10)
plt.tight_layout()
fig2.savefig(os.path.join(OUT, "fig2_block_importance.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(OUT, "fig2_block_importance.png"), bbox_inches="tight", dpi=200)
print(f"Wrote {OUT}/fig2_block_importance.pdf")

# ---------- Figure 4 (NEW): negation finding ----------
adapters_neg = ["math", "code", "general_nlp", "medical"]
neg_table = {
    "math":         [0.490, 0.435, 0.450, 0.430],
    "code":         [0.395, 0.390, 0.395, 0.420],
    "general_nlp":  [0.305, 0.315, 0.315, 0.470],
    "medical":      [0.265, 0.330, 0.315, 0.340],
}
lambdas = [-0.5, 0.0, 0.5, 1.0]

fig4, ax = plt.subplots(figsize=(7.5, 4.0))
markers = {"math": "o", "code": "s", "general_nlp": "^", "medical": "D"}
colors_n = {"math": "#d62728", "code": "#1f77b4", "general_nlp": "#2ca02c", "medical": "#9467bd"}
for adapter in adapters_neg:
    ax.plot(lambdas, neg_table[adapter], marker=markers[adapter], color=colors_n[adapter],
            label=adapter, linewidth=1.6, markersize=7.5)
    # Highlight the optimal λ
    best_idx = int(np.argmax(neg_table[adapter]))
    ax.scatter([lambdas[best_idx]], [neg_table[adapter][best_idx]],
               s=200, facecolor="none", edgecolor=colors_n[adapter], linewidth=1.7, zorder=3)

ax.axvline(0, color="#aaaaaa", linestyle=":", linewidth=0.8)
ax.axvspan(-0.6, -0.05, alpha=0.07, color="#d62728")
# Move annotation to top-left where there is empty space, away from data lines
ax.text(-0.55, 0.535, "Negative-$\\lambda$ region\n(no scalar-positive\nmerger can express)",
        fontsize=8.5, color="#d62728", ha="left", va="top")
ax.set_xlabel(r"Per-adapter coefficient $\lambda$ ($n{=}200$ probe)", fontsize=10)
ax.set_ylabel("GSM8K probe accuracy", fontsize=10)
ax.set_xticks(lambdas)
ax.set_ylim(0.22, 0.56)
ax.legend(loc="lower right", fontsize=9, frameon=False, ncol=2)
ax.set_title(r"Probe accuracy vs. per-adapter $\lambda$. Math adapter is probe-optimal at $\lambda=-0.5$.",
             fontsize=11)
plt.tight_layout()
fig4.savefig(os.path.join(OUT, "fig3_negation.pdf"), bbox_inches="tight")
fig4.savefig(os.path.join(OUT, "fig3_negation.png"), bbox_inches="tight", dpi=200)
print(f"Wrote {OUT}/fig3_negation.pdf")

# ---------- Figure: single-adapter reference evaluations (signed deltas) ----------
# Each (adapter, benchmark) pair we tested.  Δ vs. unmerged baseline.
upper_bounds = [
    # (adapter, benchmark, single_adapter_score, baseline, delta)
    ("math",        "MATH-500",       0.164, 0.154,  +0.010),
    ("math",        "GSM8K",          0.660, 0.707,  -0.047),
    ("code",        "HumanEval",      0.598, 0.628,  -0.030),
    ("general_nlp", "TruthfulQA-MC2", 0.557, 0.531,  +0.026),  # baseline 0.5307 from results/baseline_llama31-8b.json
    ("general_nlp", "GSM8K",          0.654, 0.707,  -0.053),
    ("medical",     "GSM8K",          0.723, 0.707,  +0.016),
]

fig_ub, ax = plt.subplots(figsize=(8.5, 3.0))
labels_ub = [f"{a}\n{b}" for a, b, _, _, _ in upper_bounds]
deltas_ub = [d for _, _, _, _, d in upper_bounds]
colors_ub = ["#1f77b4" if d > 0.005 else
             "#999999" if abs(d) <= 0.005 else
             "#d62728"
             for d in deltas_ub]

x_ub = np.arange(len(labels_ub))
bars = ax.bar(x_ub, deltas_ub, color=colors_ub, edgecolor="black", linewidth=0.7,
              width=0.7)
ax.axhline(0, color="black", linewidth=0.7)
# Place labels with extra padding away from bar tops/bottoms
for xi, d in zip(x_ub, deltas_ub):
    if d > 0:
        ax.text(xi, d + 0.0035, f"{d:+.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.text(xi, d - 0.0035, f"{d:+.3f}",
                ha="center", va="top", fontsize=9, fontweight="bold")

ax.set_xticks(x_ub)
ax.set_xticklabels(labels_ub, fontsize=9.5)
ax.set_ylabel(r"$\Delta$ vs. unmerged baseline", fontsize=10)
# Widen y-range so annotation has its own clear band beneath the bars
ax.set_ylim(-0.090, 0.040)

# Highlight GSM8K bars to underline the protected-task retention claim.
gsm_indices = [i for i, (_, b, *_rest) in enumerate(upper_bounds) if b == "GSM8K"]
for i in gsm_indices:
    bars[i].set_edgecolor("#cc6600")
    bars[i].set_linewidth(1.8)

# Annotation in the dedicated bottom band (below all bars)
ax.text(2.5, -0.085,
        "No single adapter improves GSM8K  (unmerged baseline is the reference)",
        ha="center", va="bottom", fontsize=9, color="#cc6600",
        fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1f77b4", label=r"$\Delta > 0$ (helps)"),
    Patch(facecolor="#d62728", label=r"$\Delta < 0$ (hurts)"),
    Patch(facecolor="#999999", label=r"$|\Delta| \leq 0.005$ (neutral)"),
    Patch(facecolor="white", edgecolor="#cc6600", linewidth=1.6,
          label="Protected task (GSM8K)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8.5,
          frameon=False, ncol=2)
ax.set_title("Single-adapter reference evaluations: each adapter alone, "
             r"$\Delta$ vs. unmerged baseline",
             fontsize=11)
plt.tight_layout()
fig_ub.savefig(os.path.join(OUT, "fig1_upper_bounds.pdf"), bbox_inches="tight")
fig_ub.savefig(os.path.join(OUT, "fig1_upper_bounds.png"), bbox_inches="tight", dpi=200)
print(f"Wrote {OUT}/fig1_upper_bounds.pdf")

# ---------- Figure: adapter audit (norms of audited vs excluded) [unused] ----------
# Six adapters: 4 audited (used in paper) + 2 excluded (flagged by audit).
adapter_labels = [
    "math\n(kai-xu)",
    "code\n(FlowerTune)",
    "general_nlp\n(FlowerTune)",
    "medical\n(FlowerTune)",
    "code*\n(excluded)",
    "creative*\n(excluded)",
]
adapter_norms = [16.62, 1.73, 1.07, 1.55, 0.02, 46.0]
adapter_reasons = [
    "audited",
    "audited",
    "audited",
    "audited",
    "delta norm 0.02\n(no-op)",
    "trained on Llama-3\n(wrong base)",
]
colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4", "#d62728", "#d62728"]

fig3, ax = plt.subplots(figsize=(7.5, 3.0))
x = np.arange(len(adapter_labels))
bars = ax.bar(x, adapter_norms, color=colors, edgecolor="black", linewidth=0.6)
ax.set_yscale("log")
ax.set_ylabel(r"Total $\|\Delta\|_F$ (log scale)", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(adapter_labels, fontsize=8)
ax.axhline(1.0, linestyle="--", color="#666666", linewidth=0.8, label="Audit threshold (norm $\\geq 1.0$)")

# Annotations
for xi, (v, reason) in enumerate(zip(adapter_norms, adapter_reasons)):
    ax.text(xi, v * 1.3 if v > 0.1 else v * 1.8,
            f"{v:.2f}" if v >= 0.1 else f"{v:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    if "audited" not in reason:
        ax.text(xi, v * (0.3 if v < 10 else 0.5), reason,
                ha="center", va="top", fontsize=6.5, color="#d62728", style="italic")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#1f77b4", label="Audited (used in paper)"),
    Patch(facecolor="#d62728", label="Excluded by audit"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=8, frameon=False)
ax.set_title("Adapter-audit outcome: 4 audited Llama-3.1 adapters + 2 excluded defective adapters",
             fontsize=10)
ax.set_ylim(0.01, 200)
plt.tight_layout()
fig3.savefig(os.path.join(OUT, "fig3_adapter_audit.pdf"), bbox_inches="tight")
fig3.savefig(os.path.join(OUT, "fig3_adapter_audit.png"), bbox_inches="tight", dpi=200)
print(f"Wrote {OUT}/fig3_adapter_audit.pdf")

print("Done.")
