"""From-scratch encoder-decoder transformer (M3), raw PyTorch only (ADR-010).

from kanjiland.model import Transformer, ModelConfig
model = Transformer(ModelConfig(vocab_size=16000))
logits = model(src, tgt_in)            # training (teacher forcing)
"""

from .attention import MultiHeadAttention
from .layers import DecoderLayer, EncoderLayer, FeedForward
from .positional import RotaryEmbedding, SinusoidalPositionalEncoding
from .transformer import ModelConfig, Transformer

__all__ = [
    "Transformer",
    "ModelConfig",
    "MultiHeadAttention",
    "EncoderLayer",
    "DecoderLayer",
    "FeedForward",
    "RotaryEmbedding",
    "SinusoidalPositionalEncoding",
]
