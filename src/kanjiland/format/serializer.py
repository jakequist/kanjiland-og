"""Document -> wire-format serializer (FORMAT_SPEC.md §2–§4).

Contract:
    serialize(doc: Document) -> str
    - One record per line (newline after each RECORD_END).
    - Must satisfy: parse(serialize(doc)) == doc.
    - Raises ValueError if any content field contains a reserved PUA
      codepoint (invariant 9). The serializer refuses to produce wire text
      the parser would reject.
"""

from __future__ import annotations

from .records import Document, GrammarAnnotation, Sentence, Token, Word
from .separators import (
    FIELD_SEP,
    GRAMMAR,
    HEADER,
    LIST_SEP,
    PARAGRAPH,
    RECORD_END,
    SENTENCE,
    TOKEN,
    WORD,
    contains_reserved,
)


def serialize(doc: Document) -> str:
    _check(doc.header.format_version, "header.format_version")
    _check(doc.header.ruleset_version, "header.ruleset_version")

    parts: list[str] = [
        f"{HEADER}{doc.header.format_version}"
        f"{FIELD_SEP}{doc.header.ruleset_version}{RECORD_END}\n"
    ]

    for para in doc.paragraphs:
        parts.append(f"{PARAGRAPH}{RECORD_END}\n")
        parts.extend(_serialize_token(t) for t in para.tokens)
        parts.extend(_serialize_word(w) for w in para.words)
        parts.extend(_serialize_sentence(s) for s in para.sentences)
        parts.extend(_serialize_grammar(g) for g in para.grammar)

    return "".join(parts)


def _serialize_token(t: Token) -> str:
    _check(t.surface, f"token[{t.id}].surface")
    for r in t.ruby:
        _check(r, f"token[{t.id}].ruby")
    _check(t.dictionary_form, f"token[{t.id}].dictionary_form")
    _check(t.pos, f"token[{t.id}].pos")
    _check(t.gloss, f"token[{t.id}].gloss")
    ruby_s = LIST_SEP.join(t.ruby)
    return (
        f"{TOKEN}{t.id}{FIELD_SEP}{t.surface}{FIELD_SEP}{ruby_s}"
        f"{FIELD_SEP}{t.dictionary_form}{FIELD_SEP}{t.pos}"
        f"{FIELD_SEP}{t.gloss}{RECORD_END}\n"
    )


def _serialize_word(w: Word) -> str:
    _check(w.dictionary_form, "word.dictionary_form")
    _check(w.gloss, "word.gloss")
    return f"{WORD}{w.span}{FIELD_SEP}{w.dictionary_form}{FIELD_SEP}{w.gloss}{RECORD_END}\n"


def _serialize_sentence(s: Sentence) -> str:
    _check(s.translation, "sentence.translation")
    return f"{SENTENCE}{s.span}{FIELD_SEP}{s.translation}{RECORD_END}\n"


def _serialize_grammar(g: GrammarAnnotation) -> str:
    _check(g.rule_id, "grammar.rule_id")
    role_parts: list[str] = []
    for name, target in g.roles:
        _check(name, f"grammar[{g.rule_id}].role_name")
        role_parts.append(f"{name}={target}")
    body = FIELD_SEP.join([g.rule_id, *role_parts]) if role_parts else g.rule_id
    return f"{GRAMMAR}{body}{RECORD_END}\n"


def _check(value: str, where: str) -> None:
    if contains_reserved(value):
        raise ValueError(f"{where}: field contains reserved PUA codepoint: {value!r}")
