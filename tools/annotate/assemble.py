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


def _target(v):
    """A ⟨G⟩ target is a token id (int) or a [start,end] span."""
    if isinstance(v, list) and len(v) == 2:
        return Span(int(v[0]), int(v[1]))
    return int(v)


def build_document(morphs, ann: dict) -> Document:
    """Combine deterministic morphs (order = token id) with the teacher's JSON."""
    glosses = {int(t["id"]): t.get("gloss", "") for t in ann.get("tokens", [])}
    tokens = [
        Token(i, m.surface, m.ruby, m.dictionary_form, m.pos, glosses.get(i, ""))
        for i, m in enumerate(morphs)
    ]
    words = [
        Word(Span(int(w["span"][0]), int(w["span"][1])), w.get("dict", ""), w.get("gloss", ""))
        for w in ann.get("words", [])
    ]
    sentences = [
        Sentence(Span(int(s["span"][0]), int(s["span"][1])), s.get("translation", ""))
        for s in ann.get("sentences", [])
    ]
    grammar = [
        GrammarAnnotation(g["rule"], tuple((r, _target(t)) for r, t in g.get("roles", {}).items()))
        for g in ann.get("grammar", [])
    ]
    return Document(
        header=Header("0.2", "grammar-1.0"),
        paragraphs=[Paragraph(tokens=tokens, words=words, sentences=sentences, grammar=grammar)],
    )


def assemble_and_lint(sentence: str, morphs, ann: dict):
    """Return (document, violations). violations == [] means it passes the gate.
    Raises on structurally-broken JSON (caller drops the sentence)."""
    doc = build_document(morphs, ann)
    return doc, lint(doc, source_paragraphs=[sentence])
