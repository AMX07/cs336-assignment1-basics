# BPE Tokenizer — Writeup

Code: `cs336_basics/tokenizer.py` (`train_bpe`, `Tokenizer`, `encode_file`, CLI).
Tests: `uv run pytest tests/test_train_bpe.py tests/test_tokenizer.py` → 26 passed, 2 skipped (Linux-only memory tests).
Machine: Apple Silicon Mac, 18 cores, 128 GB RAM, CPU only.

## Implementation
- **Pre-tokenization:** the file is split into ~64 MB chunks, each starting at `<|endoftext|>`. Each chunk is split on the special tokens, so no merge crosses a document boundary. The GPT-2 regex (`regex` package) then runs over the text in 18 worker processes, and the per-chunk pre-token counts are combined.
- **Merges:** each distinct pre-token is stored once with its count. Merging is incremental: an index maps each pair to the words that contain it, so only words containing the merged pair are recounted. A max-heap, with stale entries skipped, picks the most frequent pair; ties go to the lexicographically greater pair.
- **Encoding:** merges are applied to each pre-token in creation order, with a cache of already-encoded pre-tokens. Whole files are encoded in parallel over chunks that end at document boundaries and written out as flat `uint16` `.npy` arrays.

## Reproduce
```bash
uv run python -m cs336_basics.tokenizer train data/TinyStoriesV2-GPT4-train.txt 10000 out/tinystories --profile
uv run python -m cs336_basics.tokenizer train data/owt_train.txt 32000 out/owt --profile
uv run python -m cs336_basics.tokenizer encode data/TinyStoriesV2-GPT4-train.txt out/tinystories out/tinystories_train.npy
```
Load with `Tokenizer.from_files("out/owt_vocab.pkl", "out/owt_merges.pkl", ["<|endoftext|>"])`; data with `np.load("out/owt_train.npy", mmap_mode="r")`.

## train_bpe_tinystories
**(a)** Training took 14 s, with a peak of about 0.5 GiB per worker process and 0.14 GiB in the main process. The longest tokens are 15 bytes (`" accomplishment"`, `" disappointment"`, `" responsibility"`); that makes sense because they are common whole words, with their leading space, in a corpus of simple English stories.

**(b)** Pre-tokenization (running the GPT-2 regex over 2.2 GB of text) takes the most time, about 10 of the 14 s even with 18 processes; the merge loop takes about 4 s. On OpenWebText the balance flips: about 52 s is pre-tokenization and about 990 s is the merge loop, which is dominated by heap operations and updating the pair-to-word index.

## train_bpe_expts_owt
**(a)** Training took about 26 minutes wall-clock (about 17.5 min of training time plus about 5 min of laptop sleep), with peak memory of 11.2 GiB in the main process and 2.5 GiB per worker. The longest token is 64 bytes: `ÃÂ` repeated 16 times, which is mojibake, i.e. text that was mis-decoded more than once. Next are runs of 64 `-`, 16 `—` and 32 `_`. It makes sense in a frequency sense, since scraped web pages repeat these byte sequences thousands of times, but these tokens carry no meaning and waste vocabulary slots.

**(b)** The TinyStories tokenizer spends its vocabulary on whole, simple English words, and its longest tokens are ordinary words. The OpenWebText tokenizer is larger and covers far more varied text: rare words, numbers, code, URLs and non-English text. It also picks up web junk such as mojibake and separator lines, so it is more general but spends some slots on degenerate tokens.

## Tokenized data (`out/`, uint16)
| file | tokens | encode time |
|---|---|---|
| tinystories_train.npy | 541,229,347 | 15.5 s |
| tinystories_valid.npy | 5,465,883 | 0.3 s |
| owt_train.npy | 2,727,120,452 | 82.9 s |
| owt_valid.npy | 66,401,098 | 2.7 s |

Compression, measured on the valid sets: TinyStories 22.5 MB / 5.47M tokens ≈ **4.12 bytes/token**; OWT 290 MB / 66.4M tokens ≈ **4.37 bytes/token**.
