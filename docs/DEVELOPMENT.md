# Development timeline

This timeline separates existing historical evidence from commits created to publish work on September 13, 2026. New commits use their actual creation dates; the original Git history has not been rewritten.

| Date | Evidence | What it establishes |
| --- | --- | --- |
| May–June 2026 | [Captured notebook filesystem timestamps](file-timestamps.json) | Notebook files carry timestamps from this period. These are supporting metadata, not verified development dates. |
| June 13, 2026 | [Dated study note](../study-session-note.md) | A note dated June 13 describes preparation for tokenizer and Transformer implementation. |
| June 22, 2026 | Existing commit `9857b898c7f9bc0def286a2a474e4e69243285dc` | BPE code was committed with this author and committer date. The original commit is preserved. |
| September 13, 2026 | Model-component validation and publication commits | Linear, Embedding, RMSNorm, and SwiGLU pass four selected course tests; current work is organized for GitHub. |

Filesystem timestamps were captured before notebook output cleanup. They may be affected by copying, moves, or later edits and cannot reconstruct individual code changes. Historical code snapshots are available only where Git actually recorded them. Earlier upstream commits belong to the course starter, not to this personal implementation.

## Validation at publication

```sh
uv run pytest tests/test_model.py -k 'test_linear or test_embedding or test_rmsnorm or test_swiglu'
```

Result: **4 passed, 9 deselected**. The full assignment is not complete, and this validation makes no claim about BPE integration, attention, training, or other unfinished components.
