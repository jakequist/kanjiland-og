"""Text normalization for the M2 data pipeline.

Every sentence from every source passes through here before filtering. The
goal is to remove *spurious* variation — encoding quirks, width variants,
stray whitespace, control characters — without touching *meaningful* content.
Consistent normalization matters twice over: (1) it stops the tokenizer from
wasting vocab on fullwidth-vs-halfwidth duplicates of the same word, and
(2) it guarantees the format layer never sees a Private-Use-Area codepoint,
which it reserves for its own separators (FORMAT_SPEC §2).

Design notes (why each step exists):

- **NFKC** (Normalization Form KC): folds compatibility variants to a canonical
  form. In Japanese text this is the big one — it maps fullwidth ASCII (Ａ→A,
  ３→3), halfwidth katakana (ｶﾀｶﾅ→カタカナ), and various compatibility symbols
  onto their standard code points. Without it, "AI" and "ＡＩ" are different
  strings and the model would have to learn both.
- **PUA stripping**: the format uses U+E000–U+E01F as structural separators and
  forbids them inside field content. Real text almost never contains PUA code
  points, but scraped web data occasionally does (custom icon fonts, mojibake).
  We drop the whole Private Use Area and log the (rare) occurrences.
- **Control-char stripping**: newlines/tabs/NULs inside a "sentence" are always
  noise here (our records are one sentence per line); we drop C0/C1 controls
  except nothing — everything goes.
- **Whitespace collapse**: runs of whitespace → a single ASCII space, trimmed.
  Japanese has no inter-word spaces, so internal runs are almost always
  formatting artifacts.
"""

from __future__ import annotations

import unicodedata

# Private Use Area ranges to strip. The BMP PUA (U+E000–U+F8FF) covers the
# format separators and then some; the two supplementary planes are stripped
# for completeness. Content must never carry these (FORMAT_SPEC §2).
_PUA_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)


def _is_pua(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _PUA_RANGES)


def _is_control(ch: str) -> bool:
    # Unicode category "Cc" = control (C0/C1). We keep no controls; even \n and
    # \t are meaningless inside a single-sentence record.
    return unicodedata.category(ch) == "Cc"


def count_pua(text: str) -> int:
    """How many PUA code points are in ``text`` (for logging/monitoring)."""
    return sum(_is_pua(ord(ch)) for ch in text)


def normalize_text(text: str) -> str:
    """Normalize one sentence: NFKC, strip PUA + control chars, collapse space.

    Idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    # NFKC first, so width-folding happens before we inspect characters.
    text = unicodedata.normalize("NFKC", text)
    # Drop PUA and control characters in a single pass.
    text = "".join(ch for ch in text if not _is_pua(ord(ch)) and not _is_control(ch))
    # Collapse any whitespace run (now that tabs/newlines are gone, this is
    # spaces and ideographic spaces) to one ASCII space, and trim the ends.
    text = " ".join(text.split())
    return text
