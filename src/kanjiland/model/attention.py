"""Multi-head attention, from scratch — self- and cross-attention, RoPE, KV cache.

Attention lets every position build a weighted summary of other positions. With
``h`` heads the model runs ``h`` such summaries in parallel over different
learned subspaces of the ``d_model`` vector, then concatenates them — so one
head can track subject-verb agreement while another tracks, say, coreference.

Two implementations are provided and are numerically equivalent (tested):

- ``manual``: the textbook softmax(QKᵀ/√d)·V, written out so the mechanism is
  legible. Used by default in tests.
- ``sdpa``: ``F.scaled_dot_product_attention``, PyTorch's fused kernel (Flash
  Attention when available) — far faster and lower-memory, used for training.
  It's a raw-PyTorch primitive, not a modeling framework, so it stays within the
  "from-scratch, no HF transformers" rule (ADR-010).

Masking is done with an *additive* mask (0 where allowed, −inf where forbidden)
because it composes cleanly: a decoder's causal mask and a batch's padding mask
just add together, and the same tensor feeds either implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .positional import RotaryEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        rotary: RotaryEmbedding | None = None,
        attn_impl: str = "sdpa",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.rotary = rotary
        self.attn_impl = attn_impl
        self.dropout_p = dropout

        # One projection each for query/key/value, then an output projection.
        # Biases omitted — standard for transformer attention and one fewer
        # thing to learn.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, S, d_model) -> (B, n_heads, S, head_dim)
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (B, n_heads, S, head_dim) -> (B, S, d_model)
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        attn_mask: Tensor | None = None,
        *,
        apply_rope: bool = False,
        q_pos: int = 0,
        kv_pos: int = 0,
        cache: dict | None = None,
        static: bool = False,
    ) -> Tensor:
        """Attend from ``x_q`` (queries) to ``x_kv`` (keys/values).

        - self-attention: ``x_q is x_kv``.
        - cross-attention: ``x_q`` = decoder states, ``x_kv`` = encoder memory.
        - ``apply_rope``: rotate Q/K by position (self-attention only; relative
          position across two different sequences isn't meaningful).
        - ``cache``: per-layer dict for incremental decoding. For self-attention
          the new K/V are appended to the cached past; for cross-attention
          (``static=True``) the encoder K/V are computed once and reused.
        - ``q_pos``/``kv_pos``: absolute position offsets for RoPE during
          incremental decoding (the new token sits after the cached ones).
        """
        q = self._split_heads(self.q_proj(x_q))
        if apply_rope and self.rotary is not None:
            q = self.rotary.rotate(q, offset=q_pos)

        if static and cache is not None and "k" in cache:
            # Cross-attention with a warm cache: encoder K/V never change across
            # decoding steps, so reuse them instead of re-projecting the memory.
            k, v = cache["k"], cache["v"]
        else:
            k = self._split_heads(self.k_proj(x_kv))
            v = self._split_heads(self.v_proj(x_kv))
            if apply_rope and self.rotary is not None:
                # Rotate only the *new* keys; cached keys were already rotated
                # when they were first computed.
                k = self.rotary.rotate(k, offset=kv_pos)
            if cache is not None:
                if static:
                    cache["k"], cache["v"] = k, v
                else:
                    if "k" in cache:  # append this step's K/V to the running past
                        k = torch.cat([cache["k"], k], dim=2)
                        v = torch.cat([cache["v"], v], dim=2)
                    cache["k"], cache["v"] = k, v

        out = self._attend(q, k, v, attn_mask)
        return self.out_proj(self._merge_heads(out))

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None) -> Tensor:
        if self.attn_impl == "sdpa":
            # F.sdpa applies the 1/√d scaling internally and fuses softmax+matmul.
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
            )
        # manual reference path. attn_mask is boolean (True = attend), matching
        # F.sdpa's convention; forbidden positions get the most-negative finite
        # value so softmax drives them to ~0 (finite, not -inf, to avoid NaN on a
        # fully-masked row).
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return weights @ v
