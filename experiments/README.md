# Experiment archive

| Report | What it answers | Visuals |
|---|---|---|
| [Hardware tuning](hardware-tuning.md) | Which LR and batch-size choices were useful on MacBook/MPS, A100, H100 and B200? | [LR curves by hardware](figures/hardware-lr.png), [batch-size curves](figures/hardware-batch.png) |
| [A100 architecture ablations](a100-ablations.md) | What changed when removing RMSNorm, moving normalization, removing RoPE, or replacing SwiGLU with SiLU? | [Ablation curves](figures/a100-ablations.png) |

**Baseline:** [valiant-yogurt-171](https://wandb.ai/alephnott/assignment1/runs/rukvkasi), A100 40 GB, B=200, LR 0.005→0.0005, 6,400 steps, final validation loss **1.3308**.

The reports link every included run and distinguish completed results, early-stopped probes and numerical failures. Twenty-seven informative runs were selected from 184 audited runs. Hardware metadata identifies the previously named “GB200” sweep as NVIDIA B200; the report preserves that distinction.

## Files

- `models/`: the user-written post-norm, NoPE and SiLU model variants, moved from `cs336_basics/`.
- `pretrain_*.py`: variant-specific reproduction entry points using the existing training loop. See the full commands and qualifications in the [A100 report](a100-ablations.md#code-and-reproduction).
- `data/runs.json`: selected W&B configs, sanitized hardware metadata and raw metric histories. String `"NaN"` values are preserved; plots omit non-finite points rather than treating them as zero.
- `plot_results.py`: reproducible reporting code; dependencies are isolated with uv script metadata.
- `figures/`: PNG previews and scalable SVG versions.

Regenerate the figures without W&B credentials or GPU access:

```sh
uv run experiments/plot_results.py
```

Curated data was exported on October 21, 2026. Model code is the author's implementation; report generation, file organization and plotting are administrative tooling.
