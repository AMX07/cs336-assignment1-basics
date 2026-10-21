# Language Modeling from Scratch — CS336

My from-scratch implementation of Stanford CS336 Assignment 1, building and training a Transformer LM in PyTorch.

![TinyStories LR sweep comparing validation loss for peak learning rates 0.01, 0.02, and 0.03](docs/images/lr-sweep.png)

Across three 2,000-step TinyStories runs, peak LR **0.01** achieved the lowest final validation loss: **1.378**, compared with **2.400** at 0.02 and **2.198** at 0.03. All three used 100 warmup steps followed by cosine decay to 10% of peak LR, with the same model, batch size, and context length.



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
| Experiments | In progress |
| Leaderboard run | In progress |

