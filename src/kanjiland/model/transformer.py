"""The encoder-decoder transformer, assembled from scratch (raw PyTorch).

Ja→En translation model (ROADMAP M3). Encoder reads the source sentence into a
sequence of context vectors ("memory"); the decoder generates the target token
by token, attending both to what it has produced so far (masked self-attention)
and to the source memory (cross-attention).

Design points worth knowing:

- **Tied embeddings** (ADR-012 uses a *joint* Ja+En vocab, which makes this
  natural): with ``three_way`` the source-embedding, target-embedding, and
  output-projection matrices are the *same* weights. That removes ~2 vocab×d
  matrices of parameters and couples "recognizing a token" with "producing it".
- **Two positional schemes** selectable at config time (M5 ablation): sinusoidal
  adds a signal to embeddings; RoPE rotates Q/K inside attention (see
  positional.py). ``apply_rope`` threads through to every attention call.
- **Boolean masks** (True = attend): a source *padding* mask, and a decoder mask
  that is causal ∧ target-padding. Building them once here keeps the layers
  clean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import DecoderLayer, EncoderLayer
from .positional import RotaryEmbedding, SinusoidalPositionalEncoding


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    encoder_layers: int = 6
    decoder_layers: int = 6
    dropout: float = 0.1
    pos_encoding: str = "rope"  # "rope" | "sinusoidal"
    tie_embeddings: str = "three_way"  # "three_way" | "decoder_only" | "none"
    norm: str = "pre"  # only pre-LN implemented
    activation: str = "gelu"  # "gelu" | "relu"
    attn_impl: str = "sdpa"  # "sdpa" (fast) | "manual" (reference)
    max_len: int = 4096
    pad_id: int = 0

    @classmethod
    def from_dict(cls, d: dict, vocab_size: int) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(vocab_size=vocab_size, **{k: v for k, v in d.items() if k in known})


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.norm != "pre":
            raise ValueError("only pre-LN is implemented")
        self.cfg = cfg
        self.embed_scale = math.sqrt(cfg.d_model)

        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # RoPE is one shared table (same rotation for every layer); sinusoidal is
        # added once to the embeddings.
        rotary = None
        self.sinusoidal = None
        if cfg.pos_encoding == "rope":
            rotary = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_len=cfg.max_len)
        elif cfg.pos_encoding == "sinusoidal":
            self.sinusoidal = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        else:
            raise ValueError(f"unknown pos_encoding {cfg.pos_encoding}")
        self.apply_rope = cfg.pos_encoding == "rope"

        args = (
            cfg.d_model,
            cfg.n_heads,
            cfg.d_ff,
            cfg.dropout,
            rotary,
            cfg.attn_impl,
            cfg.activation,
        )
        self.encoder = nn.ModuleList(EncoderLayer(*args) for _ in range(cfg.encoder_layers))
        self.decoder = nn.ModuleList(DecoderLayer(*args) for _ in range(cfg.decoder_layers))
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self._init_weights()
        self._tie_weights()

    # --- setup --------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.cfg.d_model**-0.5)
                if m.padding_idx is not None:
                    nn.init.zeros_(m.weight[m.padding_idx])

    def _tie_weights(self) -> None:
        # Weight tying = literal parameter sharing (same tensor object), so a
        # gradient to any view updates them all.
        if self.cfg.tie_embeddings == "three_way":
            self.tgt_embed.weight = self.src_embed.weight
            self.output.weight = self.src_embed.weight
        elif self.cfg.tie_embeddings == "decoder_only":
            self.output.weight = self.tgt_embed.weight
        elif self.cfg.tie_embeddings != "none":
            raise ValueError(f"unknown tie_embeddings {self.cfg.tie_embeddings}")

    def num_parameters(self) -> int:
        # Count unique tensors so tied weights aren't double-counted.
        seen = {id(p): p for p in self.parameters()}
        return sum(p.numel() for p in seen.values())

    # --- masks (boolean, True = attend) -------------------------------------

    def _src_key_mask(self, src: Tensor) -> Tensor:
        # (B, S) -> (B, 1, 1, S): which source keys are real (not padding).
        return (src != self.cfg.pad_id)[:, None, None, :]

    def _tgt_self_mask(self, tgt: Tensor) -> Tensor:
        # Causal ∧ target-padding, shape (B, 1, S, S). A query at position i may
        # attend to key j iff j <= i (no peeking ahead) and j is not padding.
        s = tgt.size(1)
        causal = torch.tril(torch.ones(s, s, dtype=torch.bool, device=tgt.device))
        key_ok = (tgt != self.cfg.pad_id)[:, None, None, :]  # (B,1,1,S)
        return causal[None, None] & key_ok

    # --- forward ------------------------------------------------------------

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        x = self.src_embed(src) * self.embed_scale
        if self.sinusoidal is not None:
            x = self.sinusoidal(x)
        x = self.dropout(x)
        for layer in self.encoder:
            x = layer(x, src_mask, self.apply_rope)
        return self.enc_norm(x)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        self_mask: Tensor | None,
        cross_mask: Tensor,
        cache: list | None = None,
        self_pos: int = 0,
    ) -> Tensor:
        x = self.tgt_embed(tgt) * self.embed_scale
        if self.sinusoidal is not None:
            x = self.sinusoidal(x, offset=self_pos)
        x = self.dropout(x)
        for i, layer in enumerate(self.decoder):
            x = layer(
                x,
                memory,
                self_mask,
                cross_mask,
                self.apply_rope,
                cache=cache[i] if cache is not None else None,
                self_pos=self_pos,
            )
        x = self.dec_norm(x)
        return self.output(x)

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        """Teacher-forced forward for training. ``tgt_in`` is the target shifted
        right (BOS ... token_{n-1}); the caller compares logits to (token_1 ...
        EOS). Returns logits (B, S_tgt, vocab)."""
        src_mask = self._src_key_mask(src)
        memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, self._tgt_self_mask(tgt_in), src_mask)

    def init_cache(self) -> list:
        """Empty per-layer KV cache for incremental decoding."""
        return [{"self": {}, "cross": {}} for _ in range(len(self.decoder))]
