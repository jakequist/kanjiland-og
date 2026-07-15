"""Positional encodings: sinusoidal (added to embeddings) and RoPE (rotates Q/K).

A transformer's attention is permutation-invariant — it has no built-in notion
of token order — so we inject position information explicitly. We implement the
two variants M5 ablates (ROADMAP M5, ADR-004 is unrelated; see roadmap):

- **Sinusoidal** (Vaswani et al. 2017): a fixed, non-learned signal *added* to
  the token embeddings once, before the stack. Position k, dimension i gets
  sin/cos of k / 10000^(i/d). Cheap, and extrapolates to unseen lengths in
  principle because it's a smooth function of position.

- **RoPE** (Rotary Position Embedding, Su et al. 2021): instead of adding a
  signal, it *rotates* the query and key vectors by an angle proportional to
  their absolute position, applied inside attention. The dot product q·k then
  depends only on the *relative* offset (k_pos − q_pos), which is exactly what
  attention wants — so RoPE tends to generalize better to longer sequences and
  is the modern default. Nothing is added to the embeddings.

The key consequence for the rest of the code: with sinusoidal, positions are
baked in at embedding time; with RoPE, the attention module must apply the
rotation to Q and K itself (see attention.py).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos positional signal added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        # Precompute the (max_len, d_model) table once. position k, dim 2i uses
        # sin(k / 10000^(2i/d)), dim 2i+1 uses cos of the same argument. The
        # geometric spread of frequencies lets the model attend by relative
        # position via linear combinations of these basis functions.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        # div_term = 10000^(-2i/d) computed in log space for numerical stability.
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a buffer (moves with .to(device), saved in state_dict but
        # not a trained parameter).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Add positional encoding to ``x`` of shape (batch, seq, d_model).

        ``offset`` shifts the starting position — nonzero during incremental
        decoding, where the token being generated sits after the cached ones."""
        return x + self.pe[offset : offset + x.size(1)].unsqueeze(0)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding applied to Q/K inside attention.

    Precomputes cos/sin tables for the head dimension and exposes ``rotate`` to
    apply the position-dependent rotation. Uses the "half-split" convention
    (GPT-NeoX / HF style): the head_dim is split in two halves that rotate
    together, which is mathematically equivalent to rotating adjacent pairs but
    friendlier to contiguous memory.
    """

    def __init__(self, head_dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        # inv_freq[i] = base^(-2i/head_dim): the rotation speed of the i-th pair.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        pos = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # (max_len, head_dim/2): angle per pos,pair
        # Duplicate across the two halves so cos/sin line up with rotate_half.
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        # Split into halves [a, b] and return [-b, a] — the 90° rotation partner.
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        return torch.cat([-b, a], dim=-1)

    def rotate(self, x: Tensor, offset: int = 0) -> Tensor:
        """Rotate ``x`` of shape (batch, heads, seq, head_dim) by absolute
        position. ``offset`` is the position of the first token — nonzero during
        incremental decoding, where each new token sits after the cached ones."""
        seq = x.size(-2)
        cos = self.cos[offset : offset + seq].to(x.dtype)  # (seq, head_dim)
        sin = self.sin[offset : offset + seq].to(x.dtype)
        # Broadcast over (batch, heads): (1, 1, seq, head_dim).
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin
