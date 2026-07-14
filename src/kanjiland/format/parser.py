"""Wire-format parser (FORMAT_SPEC.md §2–§4).

Contract:
    parse(wire: str) -> Document
    - Raises ParseError with a position and reason on malformed input.
    - Must satisfy: parse(serialize(doc)) == doc  (round-trip property).

Strategy: split on RECORD_END, then dispatch on the leading tag character
of each record. Newlines outside records are insignificant.
"""

from __future__ import annotations

from .records import (
    Document,
    GrammarAnnotation,
    Header,
    Paragraph,
    Sentence,
    Span,
    Token,
    Word,
)
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
)


class ParseError(ValueError):
    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message if position is None else f"@{position}: {message}")
        self.position = position


def parse(wire: str) -> Document:
    if not wire:
        raise ParseError("empty input")

    header: Header | None = None
    paragraphs: list[Paragraph] = []
    current: Paragraph | None = None
    offset = 0

    for raw in wire.split(RECORD_END):
        record_offset = offset
        offset += len(raw) + 1  # +1 for the RECORD_END we split on
        record = raw.lstrip("\n\r\t ")
        if not record:
            continue

        tag = record[0]
        body = record[1:]

        if tag == HEADER:
            if header is not None:
                raise ParseError("multiple header records", record_offset)
            header = _parse_header(body, record_offset)
        elif tag == PARAGRAPH:
            if body:
                raise ParseError("paragraph record must have no body", record_offset)
            current = Paragraph()
            paragraphs.append(current)
        elif tag == TOKEN:
            if current is None:
                raise ParseError("token record before any paragraph", record_offset)
            current.tokens.append(_parse_token(body, record_offset))
        elif tag == WORD:
            if current is None:
                raise ParseError("word record before any paragraph", record_offset)
            current.words.append(_parse_word(body, record_offset))
        elif tag == SENTENCE:
            if current is None:
                raise ParseError("sentence record before any paragraph", record_offset)
            current.sentences.append(_parse_sentence(body, record_offset))
        elif tag == GRAMMAR:
            if current is None:
                raise ParseError("grammar record before any paragraph", record_offset)
            current.grammar.append(_parse_grammar(body, record_offset))
        else:
            raise ParseError(
                f"unknown record tag U+{ord(tag):04X} ({tag!r})", record_offset
            )

    if header is None:
        raise ParseError("missing header record")
    return Document(header=header, paragraphs=paragraphs)


def _parse_header(body: str, pos: int) -> Header:
    fields = body.split(FIELD_SEP)
    if len(fields) != 2:
        raise ParseError(f"header expects 2 fields, got {len(fields)}", pos)
    return Header(format_version=fields[0], ruleset_version=fields[1])


def _parse_token(body: str, pos: int) -> Token:
    fields = body.split(FIELD_SEP)
    if len(fields) != 6:
        raise ParseError(f"token expects 6 fields, got {len(fields)}", pos)
    tok_id_s, surface, ruby_s, dict_form, pos_tag, gloss = fields
    try:
        tok_id = int(tok_id_s)
    except ValueError as e:
        raise ParseError(f"token id not an int: {tok_id_s!r}", pos) from e
    ruby = tuple(ruby_s.split(LIST_SEP)) if ruby_s else ()
    return Token(
        id=tok_id,
        surface=surface,
        ruby=ruby,
        dictionary_form=dict_form,
        pos=pos_tag,
        gloss=gloss,
    )


def _parse_word(body: str, pos: int) -> Word:
    fields = body.split(FIELD_SEP)
    if len(fields) != 3:
        raise ParseError(f"word expects 3 fields, got {len(fields)}", pos)
    span_s, dict_form, gloss = fields
    return Word(span=Span.parse(span_s), dictionary_form=dict_form, gloss=gloss)


def _parse_sentence(body: str, pos: int) -> Sentence:
    fields = body.split(FIELD_SEP)
    if len(fields) != 2:
        raise ParseError(f"sentence expects 2 fields, got {len(fields)}", pos)
    span_s, translation = fields
    return Sentence(span=Span.parse(span_s), translation=translation)


def _parse_grammar(body: str, pos: int) -> GrammarAnnotation:
    fields = body.split(FIELD_SEP)
    if not fields or not fields[0]:
        raise ParseError("grammar record missing rule_id", pos)
    rule_id = fields[0]
    roles: list[tuple[str, int | Span]] = []
    for role_field in fields[1:]:
        name, sep, target_s = role_field.partition("=")
        if not sep or not name or not target_s:
            raise ParseError(f"malformed grammar role {role_field!r}", pos)
        target: int | Span
        if ":" in target_s:
            target = Span.parse(target_s)
        else:
            try:
                target = int(target_s)
            except ValueError as e:
                raise ParseError(
                    f"grammar role target not int/span: {target_s!r}", pos
                ) from e
        roles.append((name, target))
    return GrammarAnnotation(rule_id=rule_id, roles=tuple(roles))
