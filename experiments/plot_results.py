# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Regenerate the report figures from the checked-in, curated W&B export.

Run from any directory: uv run experiments/plot_results.py
This is reporting code, not model or training code.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RUNS = {r["id"]: r for r in json.loads((ROOT / "data/runs.json").read_text())["runs"]}
COLORS = ["#2357a4", "#c35c18", "#16816c", "#954b9e", "#c43e48", "#718096"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.labelcolor": "#243448",
                     "text.color": "#243448", "axes.grid": True,
                     "grid.alpha": .15, "figure.facecolor": "white"})


def points(rid, metric="val_loss", tokens=False):
    r = RUNS[rid]
    pairs = sorted({row["_step"]: row[metric] for row in r["history"]
                    if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])}.items())
    factor = r["config"]["batch_size"] * r["config"]["context_length"] / 1e6 if tokens else 1
    return [x * factor for x, _ in pairs], [y for _, y in pairs]


def curve(ax, rid, label, color, tokens=False, metric="val_loss", style="-"):
    x, y = points(rid, metric, tokens)
    ax.plot(x, y, label=label, color=color, lw=1.8, linestyle=style)
    ax.scatter(x[-1], y[-1], color=color, s=20, zorder=3)


def finish(fig, name, footnote):
    fig.text(.025, .025, footnote, fontsize=9, color="#536375", va="bottom")
    fig.tight_layout(rect=(0, .09, 1, .94))
    for extension in ["png", "svg"]:
        fig.savefig(ROOT / "figures" / f"{name}.{extension}", dpi=180)
    plt.close(fig)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Learning-rate experiments, grouped by hardware", fontsize=18, fontweight="bold", y=.985)
groups = [
    ("MacBook · Apple MPS · B=640", [("rnoc2pxe", "0.008 → 0.001"), ("hiytf04j", "0.009 → 0.001"), ("wx9o9huh", "0.01 → 0.001")]),
    ("A100 40 GB · B=200", [("rukvkasi", "0.005 → 0.0005"), ("jwu7v6h0", "0.0025 → 0.00025"), ("bse95z37", "0.01 → 0.001 · stopped")]),
    ("H100 PCIe · B=640", [("1sjic7ym", "0.005 → 0.001"), ("4lq9tn3g", "0.009 → 0.001"), ("n0a7jkyg", "0.01 → 0.001 · earlier"), ("753n2zbh", "0.01 → 0.001 · later"), ("5inlca2c", "0.1 → 0.001")]),
    ("B200 · B=640", [("l5dba839", "0.01 → 0.001"), ("tiuyf0cn", "0.02 → 0.002"), ("cwe96lt3", "0.03 → 0.003")]),
]
for ax, (title, entries) in zip(axes.flat, groups):
    for i, (rid, label) in enumerate(entries): curve(ax, rid, label, COLORS[i], style="--" if rid == "753n2zbh" else "-")
    ax.set(title=title, xlabel="Training step", ylabel="Validation loss (nats/token)", ylim=(1.2, 4.2))
    ax.legend(fontsize=8, frameon=False)
finish(fig, "hardware-lr", "Raw validation samples; curves end at their last logged evaluation. Initial loss above 4.2 is outside the display.\nPanels have different training budgets. Hardware is a grouping variable, not an isolated causal explanation.")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
fig.suptitle("Batch size at approximately 327.68M training tokens", fontsize=18, fontweight="bold", y=.98)
for ax, title, entries in [
    (axes[0], "A100 · LR 0.005 → 0.0005", [("rukvkasi", "B=200 · 6,400 steps"), ("wy5g0wte", "B=300 · 4,267 steps")]),
    (axes[1], "H100 · LR 0.01 → 0.001", [("753n2zbh", "B=640 · later run"), ("5xkb83pk", "B=128 · 10,000 steps"), ("6tx3lj83", "B=64 · 20,000 steps"), ("n0a7jkyg", "B=640 · earlier repeat")]),
]:
    for i, (rid, label) in enumerate(entries): curve(ax, rid, label, COLORS[i], tokens=True, style="--" if rid == "n0a7jkyg" else "-")
    ax.set(title=title, xlabel="Training tokens processed (millions)", ylabel="Validation loss (nats/token)", ylim=(1.2, 3.3))
    ax.legend(fontsize=8, frameon=False)
finish(fig, "hardware-batch", "Warmup and cosine durations scale with the step budget; context length is 256. Evaluation cadence stays at 50 steps.\nH100's earlier B=640 repeat exposes run variation; these are not repeated-seed estimates of a batch-size optimum.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.9), gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle("A100 architecture ablations · B=200 · context=256", fontsize=18, fontweight="bold", y=.98)
for i, (rid, label) in enumerate([("rukvkasi", "Baseline: pre-norm + RoPE + SwiGLU"), ("szyqbc8p", "Post-norm"), ("p4pdtnnc", "NoPE"), ("2yayrk3n", "SiLU FFN · stopped early")]):
    curve(axes[0], rid, label, COLORS[i])
axes[0].set(title="Validation loss · LR 0.005 → 0.0005", xlabel="Training step", ylabel="Validation loss (nats/token)", ylim=(1.25, 2.2))
axes[0].legend(fontsize=8, frameon=False)
for i, (rid, label) in enumerate([("rukvkasi", "Baseline · peak LR 0.005"), ("ogzm51yj", "No RMSNorm · peak LR 0.005"), ("ol070jon", "No RMSNorm · peak LR 0.001")]):
    curve(axes[1], rid, label, COLORS[i], metric="loss")
axes[1].set_yscale("log")
axes[1].set(title="Removing RMSNorm destabilized training", xlabel="Training step", ylabel="Training loss · logarithmic scale", xlim=(0, 6400))
axes[1].annotate("First logged NaN: step 230\nHigher-LR run has no finite loss afterward", xy=(230, 4e9), xytext=(1450, 2e11), fontsize=8,
                 arrowprops={"arrowstyle": "->", "color": "#536375"})
axes[1].legend(fontsize=8, frameon=False, loc="upper right")
finish(fig, "a100-ablations", "Raw metrics; no smoothing or extrapolation. SiLU's last validation is at step 3,050; training stopped after logged step 3,090.\nNo-RMSNorm curves show finite training losses only; NaNs remain explicitly recorded in data/runs.json.")
print("Wrote hardware-lr, hardware-batch, and a100-ablations figures (PNG + SVG).")
