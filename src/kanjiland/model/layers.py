"""Transformer sublayers: feed-forward, and pre-LN encoder / decoder blocks.

**Pre-LN** (LayerNorm *inside* each residual branch, before the sublayer) rather
than post-LN (norm after adding the residual). Pre-LN keeps a clean identity
path from input to output, which makes deep transformers train stably without a
delicate learning-rate warmup dance and without the loss spikes post-LN is prone
to. The price is a final LayerNorm after the whole stack (added in
transformer.py) to normalize the accumulated residual before the output head.

Each residual branch is: ``x = x + dropout(sublayer(LayerNorm(x)))``.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from .attention import MultiHeadAttention
from .positional import RotaryEmbedding


class FeedForward(nn.Module):
    """Position-wise MLP: expand to d_ff, non-linearity, project back."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Pre-LN block: self-attention then feed-forward, each residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        rotary: RotaryEmbedding | None,
        attn_impl: str,
        activation: str,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rotary, attn_impl)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, pad_mask: Tensor | None, apply_rope: bool) -> Tensor:
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, pad_mask, apply_rope=apply_rope))
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class DecoderLayer(nn.Module):
    """Pre-LN block: masked self-attention, cross-attention to the encoder
    memory, then feed-forward — each a residual branch."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        rotary: RotaryEmbedding | None,
        attn_impl: str,
        activation: str,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, rotary, attn_impl)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, rotary, attn_impl)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Tensor | None,
        cross_mask: Tensor | None,
        apply_rope: bool,
        cache: dict | None = None,
        self_pos: int = 0,
    ) -> Tensor:
        # Masked self-attention. During incremental decoding, `cache["self"]`
        # holds the past K/V and `self_pos` is the position of the new token(s).
        self_cache = cache["self"] if cache is not None else None
        h = self.norm1(x)
        x = x + self.dropout(
            self.self_attn(
                h,
                h,
                self_mask,
                apply_rope=apply_rope,
                q_pos=self_pos,
                kv_pos=self_pos,
                cache=self_cache,
            )
        )
        # Cross-attention to the (fixed) encoder memory — RoPE never applies here
        # (query and key live in different sequences), and the encoder K/V are
        # cached once via static=True.
        cross_cache = cache["cross"] if cache is not None else None
        h = self.norm2(x)
        x = x + self.dropout(
            self.cross_attn(h, memory, cross_mask, apply_rope=False, cache=cross_cache, static=True)
        )
        h = self.norm3(x)
        x = x + self.dropout(self.ff(h))
        return x
