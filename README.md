# Language Modeling from Scratch — CS336

My from-scratch implementation of Stanford CS336 Assignment 1, building and training a Transformer LM in PyTorch.

## Experiments

The [experiment archive](experiments/README.md) collects **27 informative runs**, with linked W&B results, exact logged configurations, and reproducible plots.

- **[Hardware tuning](experiments/hardware-tuning.md):** LR sweeps and batch-size comparisons grouped by **MacBook / MPS, A100, H100, and B200**. The GPU previously referred to as “GB200” is recorded as NVIDIA B200 in W&B metadata.
- **[A100 architecture ablations](experiments/a100-ablations.md):** RMSNorm removal, pre-norm versus post-norm, RoPE versus NoPE, and SwiGLU versus SiLU. Model variants and their training entry points live in [`experiments/`](experiments/README.md#files).

![Learning-rate experiments grouped by hardware](experiments/figures/hardware-lr.png)

The [A100 baseline](https://wandb.ai/alephnott/assignment1/runs/rukvkasi) reached **1.3308 validation loss** with B=200, LR 0.005→0.0005, and 6,400 steps. At the same budget, post-norm reached **1.3771** and NoPE **1.4183**. The SiLU run was stopped early and is compared at matching steps in the [ablation report](experiments/a100-ablations.md#swiglu--silu-an-early-stopped-comparison).

## Current progress

| Component | Status |
| --- | --- |
| Linear projection and embedding lookup | Built |
| RMSNorm and SwiGLU feed-forward network | Built |
| Causal multi-head attention and RoPE | Built |
| Transformer blocks and complete language model | Built |
| Cross-entropy loss and AdamW optimizer | Built |
| Cosine learning-rate schedule and gradient clipping | Built |
| Data loading, training loop, and checkpointing | Built |
| Text generation with temperature and top-p sampling | Built |
| Experiments | [Hardware tuning and A100 ablations documented](experiments/README.md) |
| Leaderboard run | In progress |

## Reproduce the reports

```sh
uv run experiments/plot_results.py
```

Figures use the curated [run data](experiments/data/runs.json); no GPU or W&B login is needed to regenerate them. See the [A100 report](experiments/a100-ablations.md#code-and-reproduction) for training commands and dataset requirements.

All training code was written by me. AI was used for conceptual understanding and general administrative work, including experiment documentation and organization.

