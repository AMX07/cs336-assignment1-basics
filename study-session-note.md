# CS336 Assignment 1 Prep Note

Date: 2026-06-13

## What Assignment 1 Is Asking Me To Build

Assignment 1 is about building a tokenizer and a small LLM from scratch.

Tokenizer work:

- Train a BPE tokenizer.
- Build tokenizer encode/decode behavior.
- Use TinyStories as one of the datasets.

LLM work:

- Implement the Transformer model components.
- Train the model on TinyStories.
- Learn and use `einsum` notation.
- Implement RoPE.
- Implement AdamW.

## Current Understanding

I have built transformer models before, but I have forgotten enough of the architecture that I should reload the basics into working memory before trying to push through CS336.

The transformer I previously built used learned positional embeddings: pass position indices through an embedding table and get a tensor of shape like `(block_size, n_embd)`, then use that in the model computation.

CS336 uses RoPE instead, so I should pay attention to how positional information enters attention differently.

## Best First Move Tomorrow

Before coding CS336, quickly review the transformer model I previously built:

`/Users/anshmittal/projects/KarpathyGPT-HW/notebooks/gpt_hw-2.ipynb`

Focus especially on tracing the shapes of `x` and `y` through the forward pass.

Tiny success condition for tomorrow:

Write a short shape trace for the previous transformer model, enough to reload:

- token IDs in
- token embeddings
- positional information
- attention input/output
- MLP input/output
- logits
- loss

Do not start by trying to finish the assignment. Start by getting the transformer shape story back into working memory.
