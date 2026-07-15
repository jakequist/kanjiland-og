"""Training data: memmapped token dataset, token-budget batching, collate.

The corpus is pre-tokenized once to a compact binary (see scripts/pretokenize.py)
so training never re-runs the Python BPE encoder. Token ids live in two flat
uint16 arrays (the joint vocab is <65536) — one for source, one for target —
plus an ``(N, 4)`` index of ``[src_off, src_len, tgt_off, tgt_len]``. We
``np.memmap`` the arrays so a 22M-pair corpus is paged from disk on demand
instead of loaded into RAM.

Two efficiency ideas that matter as much as the model (ADR-009):

- **Token-budget batching**, not fixed batch size. A batch fills until its
  *padded* token count (sequences × longest sequence) hits ``tokens_per_batch``.
  Long sentences therefore get small batches and short ones large batches, so
  every step does roughly constant work and fills the GPU evenly.
- **Length bucketing.** Padding every sequence to the batch maximum wastes
  compute on padding. We sort *locally* (within shuffled "megabatches") by
  length so each batch groups similar lengths — near-zero padding — while the
  megabatch shuffle keeps training order random enough. This is the standard
  fairseq-style trick.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    """Memmapped (src_ids, tgt_ids) pairs. Sequences already carry their special
    tokens (src: ``... EOS``; tgt: ``BOS ... EOS``) — baked in at pre-tokenize
    time, so ``__getitem__`` is a pure slice."""

    def __init__(self, prefix: str | Path):
        prefix = str(prefix)
        self.src = np.memmap(f"{prefix}.src.bin", dtype=np.uint16, mode="r")
        self.tgt = np.memmap(f"{prefix}.tgt.bin", dtype=np.uint16, mode="r")
        self.idx = np.load(f"{prefix}.idx.npy")  # (N, 4) int64

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        so, sl, to, tl = self.idx[i]
        # Copy out of the memmap and widen to int64 for embedding lookup.
        src = np.asarray(self.src[so : so + sl], dtype=np.int64)
        tgt = np.asarray(self.tgt[to : to + tl], dtype=np.int64)
        return src, tgt

    def lengths(self) -> np.ndarray:
        """Per-example length used for bucketing = max(src_len, tgt_len)."""
        return np.maximum(self.idx[:, 1], self.idx[:, 3])


class TokenBatchSampler:
    """Yields lists of indices; each batch's padded token count ≈ a budget.

    Batches are formed by: shuffle → cut into megabatches → sort each megabatch
    by length → greedily pack batches up to ``tokens_per_batch`` padded tokens →
    shuffle the batch order. Reshuffled every epoch via ``set_epoch``.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        tokens_per_batch: int,
        seed: int = 0,
        shuffle: bool = True,
        megabatch_mult: int = 50,
    ):
        self.lengths = lengths
        self.tokens_per_batch = tokens_per_batch
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        # A megabatch is ~megabatch_mult batches' worth of examples — big enough
        # to bucket well, small enough to preserve randomness.
        avg_len = max(1, int(lengths.mean()))
        self.megabatch_size = max(1, megabatch_mult * tokens_per_batch // avg_len)
        self._batches = self._build()  # cache so __len__ is stable for one epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = self._build()

    def _build(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))

        batches: list[list[int]] = []
        for mb_start in range(0, len(order), self.megabatch_size):
            mb = order[mb_start : mb_start + self.megabatch_size]
            mb = mb[np.argsort(self.lengths[mb], kind="stable")]  # bucket by length
            cur: list[int] = []
            cur_max = 0
            for i in mb:
                new_max = max(cur_max, int(self.lengths[i]))
                # Would adding this example blow the padded-token budget? If so,
                # flush the current batch first (unless it's empty).
                if cur and (len(cur) + 1) * new_max > self.tokens_per_batch:
                    batches.append(cur)
                    cur, cur_max = [int(i)], int(self.lengths[i])
                else:
                    cur.append(int(i))
                    cur_max = new_max
            if cur:
                batches.append(cur)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def collate(batch, pad_id: int):
    """Pad a list of (src, tgt) arrays into (src, tgt) LongTensors."""
    srcs, tgts = zip(*batch)
    src = _pad_stack(srcs, pad_id)
    tgt = _pad_stack(tgts, pad_id)
    return src, tgt


def _pad_stack(seqs, pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.from_numpy(np.asarray(s))
    return out


def make_dataloader(
    dataset: TranslationDataset,
    tokens_per_batch: int,
    pad_id: int,
    seed: int = 0,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, TokenBatchSampler]:
    # pin_memory speeds host->GPU copies but only makes sense for CUDA; leave it
    # off on MPS/CPU (where it just warns or wastes memory).
    sampler = TokenBatchSampler(dataset.lengths(), tokens_per_batch, seed, shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler
