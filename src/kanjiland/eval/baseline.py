"""Word-substitution baseline — the "beat this by a mile" reference (M3).

A translation model has to earn its keep, so we compare it against the dumbest
thing that still uses the parallel data: a **lexical substitution** model. From
the training pairs we learn, for each source (sub)token, the single target token
it most co-occurs with, then "translate" by mapping each source token through
that table and stitching the results together. No reordering, no context, no
fluency — just a bag of dictionary hits.

To avoid the table degenerating to English stopwords ("the", "of") for every
source token, associations are scored by ``count(s, t) / freq(t)`` — a crude
pointwise-association weighting that rewards target tokens *distinctively* tied
to the source token rather than merely frequent. (This is a one-line stand-in
for IBM Model 1; good enough for a floor.)

The real model should crush this on chrF; if it doesn't, the model isn't
learning translation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from kanjiland.tokenizer import Tokenizer


def build_lexicon(
    pairs: Iterable[tuple[str, str]], tok: Tokenizer, max_pairs: int
) -> dict[int, int]:
    """Learn src-token -> best tgt-token from up to ``max_pairs`` training pairs."""
    cooc: dict[int, Counter] = defaultdict(Counter)
    tgt_freq: Counter = Counter()
    for i, (ja, en) in enumerate(pairs):
        if i >= max_pairs:
            break
        src_toks = set(tok.encode(ja))  # presence, not count, on the source side
        tgt_toks = tok.encode(en)
        tgt_freq.update(tgt_toks)
        tgt_set = set(tgt_toks)
        for s in src_toks:
            cooc[s].update(tgt_set)

    lexicon: dict[int, int] = {}
    for s, ctr in cooc.items():
        # argmax_t count(s,t)/freq(t): distinctive association, not raw frequency.
        lexicon[s] = max(ctr, key=lambda t: ctr[t] / tgt_freq[t])
    return lexicon


def translate(ja: str, tok: Tokenizer, lexicon: dict[int, int]) -> str:
    """Map each source token through the lexicon and detokenize; drop misses and
    collapse immediate repeats (the only 'fluency' this baseline gets)."""
    out: list[int] = []
    for s in tok.encode(ja):
        t = lexicon.get(s)
        if t is not None and (not out or out[-1] != t):
            out.append(t)
    return tok.decode(out)
