"""Tests for the training data layer (M3): dataset, token batching, collate."""

from __future__ import annotations

import numpy as np

from kanjiland.train.data import TokenBatchSampler, TranslationDataset, collate

PAD = 0


def _make_dataset(tmp_path, seqs):
    """seqs: list of (src_ids, tgt_ids). Writes the binary format on disk."""
    prefix = tmp_path / "d"
    src_all, tgt_all, idx = [], [], []
    so = to = 0
    for s, t in seqs:
        src_all += s
        tgt_all += t
        idx.append((so, len(s), to, len(t)))
        so += len(s)
        to += len(t)
    np.asarray(src_all, dtype=np.uint16).tofile(f"{prefix}.src.bin")
    np.asarray(tgt_all, dtype=np.uint16).tofile(f"{prefix}.tgt.bin")
    np.save(f"{prefix}.idx.npy", np.asarray(idx, dtype=np.int64))
    return TranslationDataset(prefix)


def test_dataset_getitem_and_lengths(tmp_path):
    seqs = [([5, 6, 7], [9, 8]), ([1], [2, 3, 4, 5]), ([4, 4], [4, 4])]
    ds = _make_dataset(tmp_path, seqs)
    assert len(ds) == 3
    s, t = ds[0]
    assert list(s) == [5, 6, 7] and list(t) == [9, 8]
    assert list(ds.lengths()) == [3, 4, 2]  # max(src,tgt) each


def test_collate_pads_to_batch_max(tmp_path):
    seqs = [([5, 6, 7], [9, 8]), ([1], [2, 3, 4, 5])]
    ds = _make_dataset(tmp_path, seqs)
    src, tgt = collate([ds[0], ds[1]], PAD)
    assert src.shape == (2, 3) and tgt.shape == (2, 4)
    assert src[1].tolist() == [1, PAD, PAD]  # short src padded
    assert tgt[0].tolist() == [9, 8, PAD, PAD]


def test_token_batch_sampler_covers_all_and_respects_budget(tmp_path):
    rng = np.random.default_rng(0)
    seqs = [([1] * n, [1] * (n + 1)) for n in rng.integers(1, 20, size=200)]
    ds = _make_dataset(tmp_path, seqs)
    sampler = TokenBatchSampler(ds.lengths(), tokens_per_batch=64, seed=1)
    batches = list(sampler)

    # every example exactly once
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(200))
    # padded token budget respected for multi-item batches
    for b in batches:
        max_len = max(int(ds.lengths()[i]) for i in b)
        if len(b) > 1:
            assert len(b) * max_len <= 64


def test_token_batch_sampler_buckets_similar_lengths(tmp_path):
    # Mixed short/long; bucketing should keep each batch's length spread small.
    seqs = [([1] * n, [1] * n) for n in ([2] * 50 + [18] * 50)]
    ds = _make_dataset(tmp_path, seqs)
    sampler = TokenBatchSampler(ds.lengths(), tokens_per_batch=64, seed=1)
    lengths = ds.lengths()
    for b in sampler:
        spread = max(lengths[i] for i in b) - min(lengths[i] for i in b)
        assert spread <= 4  # never mixes the 2s with the 18s


def test_sampler_deterministic_and_epoch_varies(tmp_path):
    seqs = [([1] * n, [1] * n) for n in range(1, 60)]
    ds = _make_dataset(tmp_path, seqs)
    a = TokenBatchSampler(ds.lengths(), 32, seed=3)
    b = TokenBatchSampler(ds.lengths(), 32, seed=3)
    assert list(a) == list(b)  # same seed/epoch -> identical
    before = list(a)
    a.set_epoch(1)
    assert list(a) != before  # different epoch -> reshuffled
