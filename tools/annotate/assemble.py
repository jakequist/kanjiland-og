"""Assemble deterministic ⟨T⟩ skeletons + teacher JSON into a linter-checked
Document (M7, offline). tools/ may import src/kanjiland (the guard only forbids
the reverse); we reuse the real format records + linter so the data gate here is
the SAME code that will police training data.

Assembly is deliberately defensive: any structural problem in the teacher JSON
(missing gloss, bad span, unknown rule) raises, is caught by the caller, and the
sentence is dropped. The linter then catches semantic problems (reconstruction,
tiling, ruby count, required roles). Only sentences that pass become silver data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kanjiland.format import (  # noqa: E402
    Document, GrammarAnnotation, Header, Paragraph, Sentence, Span, Token, Word,
)
from kanjiland.format.linter import lint  # noqa: E402


def _valid_span(v, n: int):
    """A [start,end] that is in-bounds and non-empty, else None."""
    if isinstance(v, list) and len(v) == 2:
        a, b = int(v[0]), int(v[1])
        if 0 <= a < b <= n:
            return Span(a, b)
    return None


def build_document(morphs, ann: dict) -> Document:
    """Combine deterministic morphs (order = token id) with the teacher's JSON,
    AUTO-REPAIRING the teacher's most common structural slips so a good annotation
    isn't thrown away over a fixable defect (this lifts gate yield a lot):
      - ⟨S⟩: each input is ONE line, so we force a single sentence tiling all
        tokens (0:n) with the teacher's translation — eliminates non-tiling.
      - ⟨W⟩: drop zero-width / out-of-bounds / overlapping words (keep first).
      - ⟨G⟩: skip roles whose span target is invalid; drop an annotation only if a
        required piece is unusable. Semantic correctness is still the human audit's
        job — repair only touches structure, never meaning.
    """
    n = len(morphs)
    glosses = {int(t["id"]): t.get("gloss", "") for t in ann.get("tokens", [])}
    tokens = [
        Token(i, m.surface, m.ruby, m.dictionary_form, m.pos, glosses.get(i, ""))
        for i, m in enumerate(morphs)
    ]

    # ⟨W⟩ — sort by start, keep non-overlapping valid spans.
    words, last_end = [], 0
    for w in sorted(ann.get("words", []), key=lambda w: (w.get("span") or [0])[0]):
        sp = _valid_span(w.get("span"), n)
        if sp is not None and sp.start >= last_end:
            words.append(Word(sp, w.get("dict", ""), w.get("gloss", "")))
            last_end = sp.end

    # ⟨S⟩ — one span over the whole line; join any split translations the teacher gave.
    trans = " ".join(s.get("translation", "").strip() for s in ann.get("sentences", []) if s.get("translation"))
    sentences = [Sentence(Span(0, n), trans)] if n else []

    # ⟨G⟩ — validate each target; a token-id target must be in range, a span valid.
    grammar = []
    for g in ann.get("grammar", []):
        roles, ok = [], True
        for r, t in g.get("roles", {}).items():
            if isinstance(t, list):
                sp = _valid_span(t, n)
                if sp is None:
                    ok = False; break
                roles.append((r, sp))
            else:
                ti = int(t)
                if not 0 <= ti < n:
                    ok = False; break
                roles.append((r, ti))
        if ok and roles:
            grammar.append(GrammarAnnotation(g["rule"], tuple(roles)))

    return Document(
        header=Header("0.2", "grammar-1.0"),
        paragraphs=[Paragraph(tokens=tokens, words=words, sentences=sentences, grammar=grammar)],
    )


def assemble_and_lint(sentence: str, morphs, ann: dict):
    """Return (document, violations). violations == [] means it passes the gate.
    Raises on structurally-broken JSON (caller drops the sentence)."""
    doc = build_document(morphs, ann)
    return doc, lint(doc, source_paragraphs=[sentence])
