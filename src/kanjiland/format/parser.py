"""Wire-format parser (M0 — implement together with Claude Code).

Contract:
    parse(wire: str) -> Document
    - Raises ParseError with a position and reason on malformed input.
    - Must satisfy: parse(serialize(doc)) == doc  (round-trip property).

Design notes to discuss before implementing:
    - Single left-to-right scan vs. splitting on RECORD_END first?
    - Error recovery: for `report`/`repair` linter modes we eventually want
      a lenient mode that yields (records, errors) instead of raising.
      Start strict; add lenient later.
"""

from .records import Document


class ParseError(ValueError):
    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message if position is None else f"@{position}: {message}")
        self.position = position


def parse(wire: str) -> Document:
    raise NotImplementedError("M0: implement with Claude Code (see tests/test_format.py)")
