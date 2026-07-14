"""From-scratch byte-level BPE tokenizer (M1).

See docs/ROADMAP.md (M1) and docs/DECISIONS.md (ADR-012). Public surface:

    from kanjiland.tokenizer import Tokenizer
    tok = Tokenizer.train(lines, vocab_size=16000)
    ids = tok.encode(text); text == tok.decode(ids)
"""

from .bpe import train_bpe, word_freqs_from_pretokens
from .pretokenize import pretokenize
from .tokenizer import Tokenizer

__all__ = ["Tokenizer", "pretokenize", "train_bpe", "word_freqs_from_pretokens"]
