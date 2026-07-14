"""Annotation format layer: records, parser, serializer, linter (M0)."""

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
from .separators import from_debug, to_debug

__all__ = [
    "Document", "GrammarAnnotation", "Header", "Paragraph", "Sentence",
    "Span", "Token", "Word", "from_debug", "to_debug",
]
