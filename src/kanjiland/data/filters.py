"""Pair-level filters for the M2 data pipeline.

Parallel-corpus quality dominates translation quality far more than model
tweaks do (ROADMAP M2 learning target), so the cleaning funnel is where a lot
of the real work is. These are the cheap, deterministic filters — length,
length-ratio, script presence, and deduplication. The two *model-based*
filters (fastText language ID and LaBSE semantic similarity) live in
``langid.py`` and ``similarity.py`` because they carry heavy, lazily-loaded
dependencies.

Every threshold here is a knob, gathered into ``FilterConfig`` so experiments
set them from YAML rather than hardcoding. The chosen defaults and their
reasoning are recorded in ADR-013.

A note on measuring length: Japanese is written without spaces and packs
roughly one morpheme per 1–2 characters, so a Japanese sentence has far fewer
*characters* than its English translation has. We therefore measure both sides
in **characters** and expect en/ja char ratios well above 1 (typically ~1.5–3).
Word-counting English against character-counting Japanese would be apples to
oranges.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    """Thresholds for the deterministic filters (see ADR-013 for rationale)."""

    # Absolute length bounds (characters). Lower bounds kill empty/near-empty
    # fragments; upper bounds kill run-on paragraphs and boilerplate dumps that
    # blow past the model's 128-token context and are usually misaligned anyway.
    ja_min_chars: int = 1
    ja_max_chars: int = 250
    en_min_chars: int = 1
    en_max_chars: int = 500

    # en_chars / ja_chars must fall in this band. Japanese is compact, so a
    # *legitimate* Ja->En pair has more English characters than Japanese ones —
    # empirically the median is ~2.5–3.5 and formal text (KFTT) runs to ~5–6.
    # The band is therefore asymmetric and the upper bound generous: en up to 6x
    # ja is kept; beyond that it's almost always misalignment or a one-sided
    # truncation. (Measured on KFTT; see ADR-013. Prime ablation candidate.)
    len_ratio_min: float = 0.5
    len_ratio_max: float = 6.0

    # Require the Japanese side to actually contain Japanese script and the
    # English side to be predominantly Latin. Catches swapped columns, romaji,
    # and all-symbol junk before we pay for language ID.
    require_japanese: bool = True
    require_english: bool = True
    min_english_latin_ratio: float = 0.5


# --- script detection ---------------------------------------------------


def _is_japanese_char(ch: str) -> bool:
    """True for hiragana, katakana, or CJK ideographs — the scripts that make
    text unambiguously Japanese. (Kanji alone is shared with Chinese, but any
    kana settles it, and pure-kanji Japanese sentences are rare and still worth
    keeping.)"""
    cp = ord(ch)
    return (
        0x3040 <= cp <= 0x309F  # hiragana
        or 0x30A0 <= cp <= 0x30FF  # katakana
        or 0xFF66 <= cp <= 0xFF9D  # halfwidth katakana
        or 0x4E00 <= cp <= 0x9FFF  # CJK unified ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK extension A
    )


def has_japanese(text: str) -> bool:
    return any(_is_japanese_char(ch) for ch in text)


def english_latin_ratio(text: str) -> float:
    """Fraction of *letters* that are ASCII Latin. Ignores spaces/digits/punct
    so that "Mr. Smith, 42." isn't penalised for its non-letters."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return latin / len(letters)


# --- individual predicates (return True == KEEP) ------------------------


def length_ok(ja: str, en: str, cfg: FilterConfig) -> bool:
    return (
        cfg.ja_min_chars <= len(ja) <= cfg.ja_max_chars
        and cfg.en_min_chars <= len(en) <= cfg.en_max_chars
    )


def ratio_ok(ja: str, en: str, cfg: FilterConfig) -> bool:
    # Guard the divide; length_ok already excludes truly empty sides, but this
    # keeps the function safe to call on its own.
    if not ja or not en:
        return False
    ratio = len(en) / len(ja)
    return cfg.len_ratio_min <= ratio <= cfg.len_ratio_max


def script_ok(ja: str, en: str, cfg: FilterConfig) -> bool:
    if cfg.require_japanese and not has_japanese(ja):
        return False
    if cfg.require_english and english_latin_ratio(en) < cfg.min_english_latin_ratio:
        return False
    return True


# --- deduplication ------------------------------------------------------


def dedup_key(ja: str, en: str) -> tuple[str, str]:
    """Key for exact-duplicate removal.

    We casefold and drop internal spaces so that pairs differing only in
    capitalization or spacing collapse together — subtitle corpora in
    particular repeat the same line thousands of times with cosmetic variation.
    Kept deliberately simple (exact after light canonicalization); fuzzy
    near-dup detection would need MinHash/LSH and isn't worth it at this stage.
    """
    return (ja.replace(" ", ""), en.casefold().replace(" ", ""))
