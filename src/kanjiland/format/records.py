"""Record types for the annotation format (FORMAT_SPEC.md §4).

These dataclasses are the in-memory representation; parser.py and
serializer.py convert to/from the wire format, and linter.py validates
invariants (SPEC §7).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Span:
    """Half-open token span: start:end covers tokens start..end-1."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid span {self.start}:{self.end}")

    def __str__(self) -> str:
        return f"{self.start}:{self.end}"

    @classmethod
    def parse(cls, text: str) -> "Span":
        start, _, end = text.partition(":")
        return cls(int(start), int(end))


@dataclass(frozen=True)
class Header:
    format_version: str
    ruleset_version: str


@dataclass(frozen=True)
class Token:
    id: int
    surface: str
    ruby: tuple[str, ...]  # one entry per maximal kanji run; empty if no kanji
    dictionary_form: str
    pos: str
    gloss: str


@dataclass(frozen=True)
class Word:
    span: Span
    dictionary_form: str
    gloss: str


@dataclass(frozen=True)
class Sentence:
    span: Span
    translation: str


@dataclass(frozen=True)
class GrammarAnnotation:
    rule_id: str
    # role -> token id (int) or Span
    roles: tuple[tuple[str, int | Span], ...]


@dataclass
class Paragraph:
    tokens: list[Token] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    sentences: list[Sentence] = field(default_factory=list)
    grammar: list[GrammarAnnotation] = field(default_factory=list)

    @property
    def surface_text(self) -> str:
        """Concatenated surfaces — must equal the source paragraph text
        exactly (linter invariant 3)."""
        return "".join(t.surface for t in self.tokens)


@dataclass
class Document:
    header: Header
    paragraphs: list[Paragraph] = field(default_factory=list)
