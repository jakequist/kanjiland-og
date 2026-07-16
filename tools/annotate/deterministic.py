"""Deterministic ⟨T⟩ labels from MeCab + UniDic (M7, offline — ADR-007).

The mechanical layer of the hybrid supervision plan: segmentation, ruby, lemma,
and coarse POS come from UniDic (deterministic, rule #6), NOT from the LLM teacher
(which hallucinates readings and drifts on morpheme boundaries). The teacher only
adds the *judgment* layer later — glosses, translation, grammar.

This lives under tools/ and is NEVER imported by src/kanjiland (rule #1): the
shipped model must learn these labels, not call MeCab at inference. The
import-guard test enforces that.

The one non-trivial algorithm here is **furigana alignment** (align_ruby): UniDic
gives a whole-morpheme reading, but ADR-003 wants ruby on *kanji runs only*, no
okurigana. We recover the per-run readings by using the surface's kana as fixed
anchors in the reading and letting each kanji run absorb the reading between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Kanji = CJK unified ideographs + the iteration mark 々 (as in 人々). Everything
# else (kana, punctuation, latin) is a non-kanji "anchor" for alignment.
_KANJI = re.compile(r"[一-鿿々]")


def kata_to_hira(s: str) -> str:
    """カタカナ → ひらがな (ruby is conventionally hiragana). Leaves ー / marks as-is."""
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _segments(surface: str) -> list[tuple[bool, str]]:
    """Split a surface into maximal (is_kanji_run, text) segments."""
    segs: list[tuple[bool, str]] = []
    for ch in surface:
        k = bool(_KANJI.match(ch))
        if segs and segs[-1][0] == k:
            segs[-1] = (k, segs[-1][1] + ch)
        else:
            segs.append((k, ch))
    return segs


def align_ruby(surface: str, reading_hira: str) -> tuple[str, ...] | None:
    """Reading → one ruby string per maximal kanji run (ADR-003).

    Returns () when the surface has no kanji (no ruby needed), or None when the
    reading can't be aligned (irregular sound change etc.) so the caller can flag
    it rather than emit a wrong reading. Builds a regex where each kanji run is a
    non-greedy capture and each kana run is a literal anchor, then reads the runs'
    readings out of the match groups:  取り引き / とりひき  ->  "(.+?)り(.+?)き"  ->  (と, ひ)
    """
    segs = _segments(surface)
    if not any(k for k, _ in segs):
        return ()
    pattern = ""
    for is_kanji, text in segs:
        pattern += "(.+?)" if is_kanji else re.escape(kata_to_hira(text))
    m = re.fullmatch(pattern, reading_hira)
    if not m:
        return None
    return tuple(m.groups())


# UniDic pos1 (+ pos2 for a couple of splits) → the closed §6 tagset.
_POS1 = {
    "代名詞": "PRON", "動詞": "VERB", "形容詞": "ADJ_I", "形状詞": "ADJ_NA",
    "副詞": "ADV", "助詞": "PART", "接続詞": "CONJ", "連体詞": "DET",
    "感動詞": "INTERJ", "接頭辞": "PREFIX", "接尾辞": "SUFFIX", "記号": "SYM",
    "空白": "OTHER", "フィラー": "INTERJ",
}
_COPULA_LEMMAS = {"だ", "です"}  # である etc. handled at grammar-1.1 (register)


def map_pos(pos1: str, pos2: str, lemma: str) -> str:
    if pos1 == "名詞":
        return "NUM" if pos2 == "数詞" else "NOUN"
    if pos1 == "助動詞":
        return "COP" if lemma in _COPULA_LEMMAS else "AUX"
    if pos1 == "補助記号":
        return "PUNCT" if pos2 in ("句点", "読点") else "SYM"
    return _POS1.get(pos1, "OTHER")


@dataclass
class Morph:
    """A deterministic ⟨T⟩ skeleton — everything but the (teacher-supplied) gloss."""

    surface: str
    ruby: tuple[str, ...]
    dictionary_form: str
    pos: str
    ruby_ok: bool  # False if alignment failed (ruby left empty; flag for audit)


_TAGGER = None


def tag(sentence: str) -> list[Morph]:
    """Morpheme-segment a sentence into deterministic ⟨T⟩ skeletons."""
    global _TAGGER
    if _TAGGER is None:
        import fugashi

        _TAGGER = fugashi.Tagger()
    out: list[Morph] = []
    for w in _TAGGER(sentence):
        f = w.feature
        pos = map_pos(f.pos1, f.pos2 or "", f.lemma or "")
        dict_form = f.orthBase or w.surface
        reading = kata_to_hira(f.kana) if f.kana else ""
        ruby = align_ruby(w.surface, reading) if reading else ()
        out.append(Morph(
            surface=w.surface,
            ruby=ruby if ruby is not None else (),
            dictionary_form=dict_form,
            pos=pos,
            ruby_ok=ruby is not None,
        ))
    return out


if __name__ == "__main__":  # quick manual check
    import sys

    for m in tag(sys.argv[1] if len(sys.argv) > 1 else "私は昨日、京都で古い寺を訪れた。"):
        print(f"{m.surface:8s} {m.pos:6s} {m.dictionary_form:8s} ruby={m.ruby} ok={m.ruby_ok}")
