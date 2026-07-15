"""Tests for greedy / beam decoding with the KV cache (M3)."""

from __future__ import annotations

import torch

from kanjiland.model import ModelConfig, Transformer, beam_search, greedy_decode

VOCAB, PAD, BOS, EOS = 64, 0, 1, 2


def _model():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=VOCAB,
        d_model=32,
        n_heads=4,
        d_ff=64,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.0,
        pad_id=PAD,
    )
    return Transformer(cfg).eval()


def _ref_greedy_no_cache(model, src, max_len):
    """Reference greedy that recomputes the full decode each step (no cache)."""
    src_mask = model._src_key_mask(src)
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), BOS, dtype=torch.long)
    finished = torch.zeros(src.size(0), dtype=torch.bool)
    for _ in range(max_len):
        logits = model.decode(ys, memory, model._tgt_self_mask(ys), src_mask)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        finished = finished | (nxt == EOS)
        if bool(finished.all()):
            break
    return ys


def test_greedy_shape_and_starts_with_bos():
    model = _model()
    src = torch.randint(3, VOCAB, (4, 7))
    out = greedy_decode(model, src, BOS, EOS, PAD, max_len=10)
    assert out.size(0) == 4
    assert (out[:, 0] == BOS).all()
    assert out.size(1) <= 11  # BOS + up to max_len


def test_greedy_cache_matches_no_cache():
    # The whole point of the KV cache: identical output to full recomputation.
    model = _model()
    src = torch.randint(3, VOCAB, (3, 6))
    cached = greedy_decode(model, src, BOS, EOS, PAD, max_len=12)
    ref = _ref_greedy_no_cache(model, src, max_len=12)
    n = min(cached.size(1), ref.size(1))
    assert torch.equal(cached[:, :n], ref[:, :n])


def test_beam1_matches_greedy():
    model = _model()
    src = torch.randint(3, VOCAB, (3, 6))
    greedy = greedy_decode(model, src, BOS, EOS, PAD, max_len=12)
    beam1 = beam_search(model, src, BOS, EOS, PAD, beam=1, max_len=12)
    n = min(greedy.size(1), beam1.size(1))
    assert torch.equal(greedy[:, :n], beam1[:, :n])


def test_beam_search_runs_and_is_valid():
    model = _model()
    src = torch.randint(3, VOCAB, (2, 5))
    out = beam_search(model, src, BOS, EOS, PAD, beam=4, max_len=10)
    assert out.size(0) == 2
    assert (out[:, 0] == BOS).all()
