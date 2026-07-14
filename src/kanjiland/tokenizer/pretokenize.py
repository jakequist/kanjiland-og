"""Script-aware pre-tokenization for the byte-level BPE (ADR-012, M1).

BPE quality depends on *where merges are allowed to happen*. We split raw
text into pre-tokens along Unicode script/category boundaries so that a
single merge can never span, say, a kanji and the punctuation after it, or
a Japanese particle and an English word. Within a pre-token, byte-level BPE
is free to learn whatever merges the data supports (学生, です, ' world').

Categories (maximal same-category runs become one pre-token):

    HAN     CJK ideographs        学生, 日本語
    HIRA    Hiragana              です, かった
    KATA    Katakana (+ ー)       コーヒー
    LATIN   ASCII letters         Hello   (GPT-2 style: one leading space
                                            attaches to the following word)
    DIGIT   ASCII digits          2026
    SPACE   whitespace runs       (a lone leading space is peeled off onto a
                                   following LATIN/DIGIT run, GPT-2 style)
    OTHER   everything else       。、!?, symbols — grouped in same-category runs

This module is deliberately dependency-free and operates per-character in
pure Python (Python's ``re`` lacks ``\\p{Han}``); it is fast enough because
the corpus is pre-tokenized once at train time and inputs are short at
inference time. It sees only PUA-separator-free text — the Tokenizer strips
special tokens out before calling it (they are atomic and must never merge).
"""

from __future__ import annotations

# Category labels. LATIN/DIGIT are the only ones a leading space attaches to.
HAN, HIRA, KATA, LATIN, DIGIT, SPACE, OTHER = (
    "HAN",
    "HIRA",
    "KATA",
    "LATIN",
    "DIGIT",
    "SPACE",
    "OTHER",
)

_ATTACH_LEADING_SPACE = frozenset({LATIN, DIGIT})


def _category(ch: str) -> str:
    cp = ord(ch)
    if ch.isspace():
        return SPACE
    if "a" <= ch <= "z" or "A" <= ch <= "Z":
        return LATIN
    if "0" <= ch <= "9":
        return DIGIT
    # Hiragana block (U+3040–U+309F).
    if 0x3040 <= cp <= 0x309F:
        return HIRA
    # Katakana block + the prolonged-sound mark ー (U+30FC lives in this block)
    # and halfwidth katakana (U+FF66–U+FF9D).
    if 0x30A0 <= cp <= 0x30FF or 0xFF66 <= cp <= 0xFF9D:
        return KATA
    # CJK ideographs: main (U+4E00–U+9FFF), Extension A (U+3400–U+4DBF),
    # and compatibility ideographs (U+F900–U+FAFF).
    if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0xF900 <= cp <= 0xFAFF:
        return HAN
    return OTHER


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into script-aware pre-tokens (see module docstring).

    The concatenation of the returned pre-tokens equals ``text`` exactly —
    this is relied upon for the reconstruction invariant.
    """
    if not text:
        return []

    # 1. Group maximal same-category runs.
    runs: list[tuple[str, str]] = []  # (category, substring)
    start = 0
    cur = _category(text[0])
    for i in range(1, len(text)):
        cat = _category(text[i])
        if cat != cur:
            runs.append((cur, text[start:i]))
            start, cur = i, cat
    runs.append((cur, text[start:]))

    # 2. GPT-2-style leading space: peel the final char off a SPACE run when
    #    the next run is LATIN/DIGIT, attaching it to that run. A SPACE run
    #    that becomes empty is dropped.
    out: list[str] = []
    for idx, (cat, chunk) in enumerate(runs):
        if cat == SPACE and idx + 1 < len(runs) and runs[idx + 1][0] in _ATTACH_LEADING_SPACE:
            if len(chunk) > 1:
                out.append(chunk[:-1])
            runs[idx + 1] = (runs[idx + 1][0], chunk[-1] + runs[idx + 1][1])
            continue
        out.append(chunk)
    return out
