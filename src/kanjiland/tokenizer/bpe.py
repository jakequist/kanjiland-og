"""Byte-level BPE — the algorithm, from scratch (M1).

Base alphabet is the 256 byte values, so *any* UTF-8 string round-trips with
zero unknown tokens (this is the whole reason for byte-level: Japanese,
emoji, and stray control chars all decompose to bytes we already have). BPE
then learns merges greedily: repeatedly fuse the most frequent adjacent
symbol pair into a new symbol.

This module works purely in **byte-id space**: ids 0–255 are the raw bytes,
id ``256 + k`` is the k-th learned merge. It knows nothing about special
tokens or pre-tokenization — callers hand it a frequency table of
already-pre-tokenized byte strings, and the Tokenizer layers special tokens
and the id offset on top.

The trainer keeps an incremental index (pair → containing words) and only
re-scans the words a merge actually touches, rather than recomputing every
pair count from scratch each step — the difference between minutes and hours
on a real corpus. It is fully deterministic: ties for "most frequent pair"
break by smallest id pair, independent of dict ordering.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

# A merge is (id_a, id_b); the new id it produces is implied by its position:
# the k-th merge (0-based) creates id BYTE_BASE + k.
BYTE_BASE = 256
Pair = tuple[int, int]


def word_freqs_from_pretokens(pretokens: Iterable[str]) -> dict[bytes, int]:
    """Count identical pre-tokens, keyed by their UTF-8 bytes.

    Deduplicating into a frequency table is what makes BPE tractable:
    particles like は and common English words collapse to one entry with a
    large count instead of thousands of copies.
    """
    freqs: dict[bytes, int] = defaultdict(int)
    for tok in pretokens:
        if tok:
            freqs[tok.encode("utf-8")] += 1
    return dict(freqs)


def _adjacent_pairs(seq: list[int]) -> Iterable[Pair]:
    return zip(seq, seq[1:])


def _merge_seq(seq: list[int], a: int, b: int, new_id: int) -> list[int]:
    """Replace every non-overlapping adjacent (a, b) in ``seq`` with new_id."""
    out: list[int] = []
    i, n = 0, len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(
    word_freqs: dict[bytes, int],
    num_merges: int,
    *,
    verbose: bool = False,
) -> list[Pair]:
    """Learn ``num_merges`` merges from a pre-token frequency table.

    Returns the merges in learned order; the k-th merge produces id
    ``BYTE_BASE + k``. Deterministic for a given ``word_freqs``.
    """
    # Each unique pre-token becomes a mutable list of byte ids, with a
    # parallel frequency. All indices below are into these two lists.
    words: list[list[int]] = [list(w) for w in word_freqs]
    freqs: list[int] = list(word_freqs.values())

    pair_counts: dict[Pair, int] = defaultdict(int)
    pair_words: dict[Pair, set[int]] = defaultdict(set)
    for wi, seq in enumerate(words):
        f = freqs[wi]
        for p in _adjacent_pairs(seq):
            pair_counts[p] += f
            pair_words[p].add(wi)

    merges: list[Pair] = []
    steps = range(num_merges)
    if verbose:
        try:
            from tqdm import tqdm

            steps = tqdm(steps, desc=f"bpe {num_merges} merges")
        except ImportError:
            pass

    for k in steps:
        if not pair_counts:
            break
        # Most frequent pair; ties broken by smallest (a, b) for determinism.
        best = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
        new_id = BYTE_BASE + k
        a, b = best
        merges.append(best)

        # Only words that currently contain `best` can change. For each, pull
        # its pair contributions out of the index, merge, then put the new
        # contributions back — surgically local, so cost scales with how many
        # words the merge actually touches, not the whole corpus.
        for wi in list(pair_words[best]):
            seq = words[wi]
            f = freqs[wi]
            for p in _adjacent_pairs(seq):
                c = pair_counts.get(p, 0) - f
                if c <= 0:
                    pair_counts.pop(p, None)
                else:
                    pair_counts[p] = c
                pw = pair_words.get(p)
                if pw is not None:
                    pw.discard(wi)
                    if not pw:
                        pair_words.pop(p, None)

            new_seq = _merge_seq(seq, a, b, new_id)
            words[wi] = new_seq
            for p in _adjacent_pairs(new_seq):
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_words.setdefault(p, set()).add(wi)

    return merges
