# Language Modeling from Scratch — CS336

Ansh Mittal's work in progress on Stanford CS336 Assignment 1: building the pieces of a language model in Python and PyTorch.

This repository records the implementation, experiments, and learning process. It is based on the [Stanford CS336 assignment starter](https://github.com/stanford-cs336/assignment1-basics), whose history and MIT license are preserved. This is a personal learning project, not an official Stanford solution or a completed language model.

## Current progress

| Component | Status |
| --- | --- |
| Linear projection | Implemented; selected course test passes |
| Embedding lookup | Implemented; selected course test passes |
| RMSNorm | Implemented; selected course test passes |
| SwiGLU feed-forward network | Implemented; selected course test passes |
| BPE | Earlier implementation and exploratory notebooks; adapter integration still pending |
| Attention, RoPE, complete Transformer, optimizer, and training | Still in progress |

Validation on September 13, 2026: **4 passed, 9 deselected** in the model test file. This is a targeted check, not a claim that the full assignment suite passes.

## Explore the work

- [Model components](cs336_basics/gpt.py): Linear, Embedding, RMSNorm, and SwiGLU.
- [BPE work](cs336_basics/bpe.py) and [training notebook](cs336_basics/train_bpe.ipynb).
- [Bigram experiments](cs336_basics/makemore_part1_bigrams.ipynb), following the makemore learning exercise.
- [Test adapters](tests/adapters.py) connecting the implementations to the supplied tests.
- [Development timeline](docs/DEVELOPMENT.md), with the evidence behind the dates.

Scratch notebooks preserve exploration, including unfinished work. Their saved outputs have been cleared for publication. Some experiments contain paths from the original local setup and are not portable entry points.

## Run with uv

Use one `.venv` for this assignment, managed by uv:

```sh
uv sync --locked
uv run pytest tests/test_model.py -k 'test_linear or test_embedding or test_rmsnorm or test_swiglu'
```

The full course suite is available with `uv run pytest`, but unimplemented components are expected to fail. Large training datasets, virtual environments, and local experiment artifacts are excluded from Git.

## Course context

See the [original course README](docs/COURSE_README.md), [assignment handout](cs336_assignment1_basics.pdf), and [course website](https://cs336.stanford.edu/) for assignment requirements and data-download instructions. Students following the course should follow its collaboration and AI-use policies and produce their own implementations.

Development has included conversational AI assistance for conceptual explanations, code review, environment setup, and repository organization. The assignment remains an ongoing learning project.

Original course materials retain their attribution and [MIT license](LICENSE).
