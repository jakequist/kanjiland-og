"""Tests for the M3 eval baseline (word-substitution)."""

from __future__ import annotations

from kanjiland.eval.baseline import build_lexicon, translate
from kanjiland.tokenizer import Tokenizer


def test_baseline_lexicon_maps_and_translates():
    # Self-contained tiny tokenizer + a toy aligned corpus with a clear signal.
    tok = Tokenizer.train(["猫 犬 鳥 です", "cat dog bird is"] * 30, vocab_size=320)
    pairs = [("猫です", "cat"), ("犬です", "dog"), ("鳥です", "bird")] * 20
    lex = build_lexicon(pairs, tok, max_pairs=100)
    assert len(lex) > 0
    out = translate("猫です", tok, lex)
    assert isinstance(out, str)  # produces some string, no crash
