"""Byte-level BPE tokenizer training."""

from __future__ import annotations

import heapq
import os
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO, Iterable, Iterator

import regex as re

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Split a file into chunks whose boundaries sit at the start of a special token."""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            mini = file.read(4096)
            if not mini:
                boundaries[bi] = file_size
                break
            found = mini.find(split_special_token)
            if found != -1:
                boundaries[bi] = pos + found
                break
            pos += 4096
    return sorted(set(boundaries))


def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> Counter[bytes]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    if special_tokens:
        split_pat = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        docs = re.split(split_pat, text)
    else:
        docs = [text]
    counts: Counter[bytes] = Counter()
    for doc in docs:
        counts.update(m.group().encode("utf-8") for m in PAT.finditer(doc))
    return counts


class _HeapItem:
    """Max-heap entry: highest count first, ties broken by lexicographically greatest pair."""

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]):
        self.count = count
        self.pair = pair

    def __lt__(self, other: _HeapItem) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def pretokenize(input_path: str | os.PathLike, special_tokens: list[str], num_processes: int | None = None) -> Counter[bytes]:
    num_processes = num_processes or os.cpu_count() or 1
    split_token = (special_tokens[0] if special_tokens else "<|endoftext|>").encode("utf-8")
    file_size = os.path.getsize(input_path)
    # Many small chunks (~64MB) bound per-worker memory on large corpora.
    num_chunks = max(num_processes, file_size // (64 * 1024 * 1024))
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_token)
    jobs = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]

    total: Counter[bytes] = Counter()
    if len(jobs) == 1 or file_size < 4 * 1024 * 1024:
        for job in jobs:
            total.update(_pretokenize_chunk(job))
    else:
        with Pool(num_processes) as pool:
            for c in pool.imap_unordered(_pretokenize_chunk, jobs):
                total.update(c)
    return total


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    num_merges = max(0, vocab_size - len(vocab))

    word_counts = pretokenize(input_path, special_tokens, num_processes)
    words: list[list[bytes]] = [[bytes([b]) for b in w] for w in word_counts]
    freqs: list[int] = list(word_counts.values())
    del word_counts

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wi, (w, f) in enumerate(zip(words, freqs)):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += f
            pair_to_words[pair].add(wi)

    heap = [_HeapItem(c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    while len(merges) < num_merges and heap:
        item = heapq.heappop(heap)
        cur = pair_counts.get(item.pair, 0)
        if cur != item.count:  # stale entry
            if cur > 0:
                heapq.heappush(heap, _HeapItem(cur, item.pair))
            continue
        if cur <= 0:
            break
        a, b = best = item.pair
        new_tok = a + b
        merges.append(best)
        vocab[len(vocab)] = new_tok

        changed: set[tuple[bytes, bytes]] = set()
        for wi in list(pair_to_words.pop(best, ())):
            w, f = words[wi], freqs[wi]
            for pair in zip(w, w[1:]):
                pair_counts[pair] -= f
                changed.add(pair)
            merged: list[bytes] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    merged.append(new_tok)
                    i += 2
                else:
                    merged.append(w[i])
                    i += 1
            words[wi] = merged
            for pair in zip(merged, merged[1:]):
                pair_counts[pair] += f
                pair_to_words[pair].add(wi)
                changed.add(pair)
        pair_counts.pop(best, None)
        for pair in changed:
            c = pair_counts.get(pair, 0)
            if c > 0:
                heapq.heappush(heap, _HeapItem(c, pair))
            else:
                pair_counts.pop(pair, None)
                pair_to_words.pop(pair, None)

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer: encodes text to token IDs and decodes IDs back to text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)  # longest first for overlaps
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.vocab.values():
                self.vocab[len(self.vocab)] = b
        self.token_to_id = {t: i for i, t in self.vocab.items()}
        self.merge_ranks = {pair: r for r, pair in enumerate(self.merges)}
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")") if self.special_tokens else None
        )
        self._cache: dict[bytes, list[int]] = {}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Tokenizer:
        """Load a tokenizer from pickled vocab/merges files (as written by `save`)."""
        import pickle

        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _encode_pretoken(self, word: bytes) -> list[int]:
        """Apply merges to one pre-token in creation order (lowest rank first)."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        parts = [bytes([b]) for b in word]
        while len(parts) > 1:
            best_rank, best_i = None, -1
            for i in range(len(parts) - 1):
                r = self.merge_ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            a, b = parts[best_i], parts[best_i + 1]
            merged, i = [], 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        ids = [self.token_to_id[p] for p in parts]
        if len(self._cache) < 1_000_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        segments = self._special_re.split(text) if self._special_re else [text]
        for seg in segments:
            if not seg:
                continue
            if seg in self.special_tokens:
                ids.append(self.token_to_id[seg.encode("utf-8")])
                continue
            for m in PAT.finditer(seg):
                ids.extend(self._encode_pretoken(m.group().encode("utf-8")))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g. a file handle) with bounded memory."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, prefix: str) -> None:
        """Write `<prefix>_vocab.pkl`, `<prefix>_merges.pkl`, plus human-readable JSON/TXT copies."""
        import json
        import pickle

        with open(f"{prefix}_vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        with open(f"{prefix}_merges.pkl", "wb") as f:
            pickle.dump(self.merges, f)
        with open(f"{prefix}_vocab.json", "w") as f:
            json.dump({i: t.decode("utf-8", errors="replace") for i, t in self.vocab.items()}, f, ensure_ascii=False, indent=0)
        with open(f"{prefix}_merges.txt", "w") as f:
            for a, b in self.merges:
                f.write(f"{a!r} {b!r}\n")


# ---------------------------------------------------------------------------
# Encoding whole datasets to uint16 token arrays (parallel over document-aligned chunks)
# ---------------------------------------------------------------------------

_worker_tok: Tokenizer | None = None


def _init_encode_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global _worker_tok
    _worker_tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)


def _encode_chunk(args: tuple[str, int, int, str]) -> tuple[str, int]:
    import numpy as np

    path, start, end, out_path = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    arr = np.asarray(_worker_tok.encode(text), dtype=np.uint16)
    np.save(out_path, arr)
    return out_path, len(arr)


def encode_file(
    input_path: str, prefix: str, out_path: str, special_tokens: list[str], num_processes: int | None = None
) -> int:
    """Tokenize `input_path` into a flat uint16 `.npy` at `out_path`. Returns number of tokens."""
    import tempfile

    import numpy as np

    num_processes = num_processes or os.cpu_count() or 1
    num_chunks = max(num_processes, os.path.getsize(input_path) // (32 * 1024 * 1024))
    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out_path) or ".") as tmp:
        jobs = [(input_path, s, e, os.path.join(tmp, f"{i:06d}.npy")) for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:]))]
        with Pool(
            num_processes,
            initializer=_init_encode_worker,
            initargs=(f"{prefix}_vocab.pkl", f"{prefix}_merges.pkl", special_tokens),
        ) as pool:
            results = dict(pool.imap_unordered(_encode_chunk, jobs))
        total = sum(results.values())
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total,))
        pos = 0
        for _, _, _, p in jobs:
            a = np.load(p)
            out[pos : pos + len(a)] = a
            pos += len(a)
        out.flush()
    return total


def main() -> None:
    """CLI.

    Train:   uv run python -m cs336_basics.tokenizer train  <input.txt> <vocab_size> <out_prefix>
    Encode:  uv run python -m cs336_basics.tokenizer encode <input.txt> <out_prefix> <out.npy>
    """
    import argparse
    import resource
    import time

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("input")
    t.add_argument("vocab_size", type=int)
    t.add_argument("prefix")
    t.add_argument("--profile", action="store_true")
    e = sub.add_parser("encode")
    e.add_argument("input")
    e.add_argument("prefix")
    e.add_argument("out")
    args = ap.parse_args()
    special = ["<|endoftext|>"]

    t0 = time.time()
    if args.cmd == "train":
        if args.profile:
            import cProfile
            import pstats

            prof = cProfile.Profile()
            prof.enable()
        vocab, merges = train_bpe(args.input, args.vocab_size, special)
        if args.profile:
            prof.disable()
            pstats.Stats(prof).sort_stats("cumulative").print_stats(15)
        elapsed = time.time() - t0
        Tokenizer(vocab, merges, special).save(args.prefix)
        longest = sorted(vocab.values(), key=len, reverse=True)[:5]
        print("longest tokens:", [(len(x), x.decode("utf-8", errors="replace")) for x in longest])
    else:
        n = encode_file(args.input, args.prefix, args.out, special)
        elapsed = time.time() - t0
        print(f"tokens: {n:,}")
    # ru_maxrss is bytes on macOS
    main_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    child_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(f"time: {elapsed:.1f}s  peak RSS main: {main_gb:.2f} GiB  largest worker: {child_gb:.2f} GiB")


if __name__ == "__main__":
    main()
