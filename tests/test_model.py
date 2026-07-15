"""Tests for the from-scratch transformer (M3).

Small models on CPU. dropout=0 everywhere so forward passes are deterministic
and the equivalence checks (manual vs sdpa, cache vs full) are exact-ish.
"""

from __future__ import annotations

import torch

from kanjiland.model import ModelConfig, RotaryEmbedding, Transformer

VOCAB = 64
PAD = 0


def small_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=VOCAB,
        d_model=32,
        n_heads=4,
        d_ff=64,
        encoder_layers=2,
        decoder_layers=2,
        dropout=0.0,
        pad_id=PAD,
    )
    base.update(kw)
    return ModelConfig(**base)


def _batch(b=3, s_src=7, s_tgt=5):
    torch.manual_seed(0)
    src = torch.randint(1, VOCAB, (b, s_src))
    tgt = torch.randint(1, VOCAB, (b, s_tgt))
    src[0, -2:] = PAD  # some padding to exercise masks
    return src, tgt


# --- shapes & params ----------------------------------------------------


def test_forward_shapes_rope_and_sinusoidal():
    src, tgt = _batch()
    for pe in ("rope", "sinusoidal"):
        model = Transformer(small_cfg(pos_encoding=pe)).eval()
        logits = model(src, tgt)
        assert logits.shape == (src.size(0), tgt.size(1), VOCAB)


def test_base_config_param_count_is_transformer_base():
    # The real M3 config: ~60M "transformer-base".
    p = Transformer(ModelConfig(vocab_size=16000)).num_parameters()
    assert 40_000_000 < p < 80_000_000, p


def test_three_way_tying_shares_one_tensor():
    m = Transformer(small_cfg(tie_embeddings="three_way"))
    assert m.src_embed.weight is m.tgt_embed.weight
    assert m.output.weight is m.src_embed.weight
    # decoder_only ties only output<->tgt
    m2 = Transformer(small_cfg(tie_embeddings="decoder_only"))
    assert m2.output.weight is m2.tgt_embed.weight
    assert m2.src_embed.weight is not m2.tgt_embed.weight


# --- correctness --------------------------------------------------------


def test_decoder_is_causal_no_peeking_ahead():
    # Logits at position i must not depend on target tokens after i.
    model = Transformer(small_cfg()).eval()
    src, tgt = _batch(b=1, s_tgt=6)
    with torch.no_grad():
        base = model(src, tgt)
        tgt2 = tgt.clone()
        tgt2[:, 3:] = torch.randint(1, VOCAB, tgt2[:, 3:].shape)  # change the future
        changed = model(src, tgt2)
    # positions 0..2 (strictly before the change at index 3) must be identical
    assert torch.allclose(base[:, :3], changed[:, :3], atol=1e-5)
    assert not torch.allclose(base[:, 3:], changed[:, 3:])  # sanity: change had effect


def test_manual_and_sdpa_attention_agree():
    src, tgt = _batch()
    model = Transformer(small_cfg(attn_impl="manual")).eval()
    with torch.no_grad():
        manual = model(src, tgt)
        for mod in model.modules():
            if hasattr(mod, "attn_impl"):
                mod.attn_impl = "sdpa"
        sdpa = model(src, tgt)
    assert torch.allclose(manual, sdpa, atol=1e-4)


def test_rope_rotation_preserves_norm():
    rope = RotaryEmbedding(head_dim=16, max_len=32)
    x = torch.randn(2, 4, 10, 16)  # (batch, heads, seq, head_dim)
    rotated = rope.rotate(x)
    # A rotation preserves vector length.
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_kv_cache_incremental_matches_full_decode():
    # The KV-cache generation path must reproduce a full parallel decode.
    for pe in ("rope", "sinusoidal"):
        model = Transformer(small_cfg(pos_encoding=pe)).eval()
        src, tgt = _batch(b=2, s_tgt=6)
        with torch.no_grad():
            src_mask = model._src_key_mask(src)
            memory = model.encode(src, src_mask)
            full = model.decode(tgt, memory, model._tgt_self_mask(tgt), src_mask)

            cache = model.init_cache()
            steps = []
            for t in range(tgt.size(1)):
                lg = model.decode(
                    tgt[:, t : t + 1], memory, None, src_mask, cache=cache, self_pos=t
                )
                steps.append(lg)
            incremental = torch.cat(steps, dim=1)
        assert torch.allclose(full, incremental, atol=1e-4), pe


def test_backward_runs_and_grads_flow():
    model = Transformer(small_cfg())
    src, tgt = _batch()
    logits = model(src, tgt)
    loss = logits.float().log_softmax(-1).mean()
    loss.backward()
    # tied weight should have received gradient
    assert model.src_embed.weight.grad is not None
    assert model.src_embed.weight.grad.abs().sum() > 0
