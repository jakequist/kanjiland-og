"""Format linter — the executable form of FORMAT_SPEC.md §7 (M0).

Modes:
    strict  — any violation is fatal (gate for training data)
    report  — return all violations (eval / debugging)
    repair  — best-effort fixes for a defined subset (inference salvage);
              define the repairable subset explicitly as it grows.

Each invariant in SPEC §7 gets its own check function and its own test.
"""

from dataclasses import dataclass
from enum import Enum, auto

from .records import Document


class Severity(Enum):
    ERROR = auto()
    WARNING = auto()


@dataclass(frozen=True)
class Violation:
    invariant: int          # SPEC §7 number
    severity: Severity
    message: str
    paragraph_index: int | None = None


def lint(doc: Document, source_paragraphs: list[str] | None = None) -> list[Violation]:
    """Run all invariant checks. source_paragraphs enables the
    reconstruction check (invariant 3) when available."""
    raise NotImplementedError("M0: implement with Claude Code")
