"""Format linter — the executable form of FORMAT_SPEC.md §7.

Every invariant in §7 has a check function here and (at least) one test in
tests/test_format.py. Adding an invariant means: bump the spec, add the
check, add the test.

Modes:
    strict  — any violation is fatal (gate for training data)
    report  — return all violations (eval / debugging)
    repair  — best-effort fixes for a defined subset (inference salvage);
              define the repairable subset explicitly as it grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .grammar import load_ruleset
from .records import Document, Paragraph, Span
from .separators import contains_reserved

KNOWN_FORMAT_VERSIONS = frozenset({"0.2"})
KNOWN_RULESET_VERSIONS = frozenset({"grammar-1.0"})

# FORMAT_SPEC.md §6 — closed POS tagset.
POS_TAGS = frozenset({
    "NOUN", "VERB", "ADJ_I", "ADJ_NA", "ADV", "PRON", "PART", "COP", "AUX",
    "CONJ", "DET", "NUM", "INTERJ", "PREFIX", "SUFFIX", "PUNCT", "SYM", "OTHER",
})


class Severity(Enum):
    ERROR = auto()
    WARNING = auto()


@dataclass(frozen=True)
class Violation:
    invariant: int  # SPEC §7 number
    severity: Severity
    message: str
    paragraph_index: int | None = None


class LintError(ValueError):
    """Raised by lint(..., mode='strict') when any ERROR violation is found."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        super().__init__(f"{len(violations)} lint violation(s)")


def lint(
    doc: Document,
    source_paragraphs: list[str] | None = None,
    mode: str = "report",
) -> list[Violation]:
    """Run all invariant checks. source_paragraphs enables the
    reconstruction check (invariant 3) when available."""
    if mode not in {"strict", "report", "repair"}:
        raise ValueError(f"unknown lint mode {mode!r}")
    if mode == "repair":
        raise NotImplementedError("repair mode not implemented yet")

    violations: list[Violation] = []
    _check_header(doc, violations)
    ruleset = load_ruleset(doc.header.ruleset_version)

    for i, para in enumerate(doc.paragraphs):
        source = (
            source_paragraphs[i]
            if source_paragraphs is not None and i < len(source_paragraphs)
            else None
        )
        _check_paragraph(para, i, source, ruleset, violations)

    if mode == "strict":
        errors = [v for v in violations if v.severity is Severity.ERROR]
        if errors:
            raise LintError(errors)
    return violations


# --- Invariant checks ------------------------------------------------------

def _check_header(doc: Document, out: list[Violation]) -> None:
    # Invariant 1
    if doc.header.format_version not in KNOWN_FORMAT_VERSIONS:
        out.append(Violation(1, Severity.ERROR,
            f"unknown format_version {doc.header.format_version!r}"))
    if doc.header.ruleset_version not in KNOWN_RULESET_VERSIONS:
        out.append(Violation(1, Severity.ERROR,
            f"unknown ruleset_version {doc.header.ruleset_version!r}"))


def _check_paragraph(
    para: Paragraph,
    idx: int,
    source: str | None,
    ruleset: dict,
    out: list[Violation],
) -> None:
    _check_token_ids(para, idx, out)
    _check_reconstruction(para, idx, source, out)
    _check_ruby_counts(para, idx, out)
    _check_spans(para, idx, out)
    _check_word_and_sentence_tiling(para, idx, out)
    _check_grammar(para, idx, ruleset, out)
    _check_pos(para, idx, out)
    _check_no_reserved(para, idx, out)


def _check_token_ids(para: Paragraph, idx: int, out: list[Violation]) -> None:
    # Invariant 2
    for expected, tok in enumerate(para.tokens):
        if tok.id != expected:
            out.append(Violation(2, Severity.ERROR,
                f"token id {tok.id} at position {expected} (expected sequential from 0)",
                paragraph_index=idx))
            return


def _check_reconstruction(
    para: Paragraph, idx: int, source: str | None, out: list[Violation]
) -> None:
    # Invariant 3
    if source is None:
        return
    reconstructed = para.surface_text
    if reconstructed != source:
        out.append(Violation(3, Severity.ERROR,
            f"surface concatenation {reconstructed!r} != source {source!r}",
            paragraph_index=idx))


def _check_ruby_counts(para: Paragraph, idx: int, out: list[Violation]) -> None:
    # Invariant 4
    for tok in para.tokens:
        expected = _count_kanji_runs(tok.surface)
        if len(tok.ruby) != expected:
            out.append(Violation(4, Severity.ERROR,
                f"token {tok.id} ({tok.surface!r}): "
                f"{len(tok.ruby)} ruby entries but {expected} kanji run(s)",
                paragraph_index=idx))


def _check_spans(para: Paragraph, idx: int, out: list[Violation]) -> None:
    # Invariant 5 (well-formedness handled by Span.__post_init__; check bounds)
    n = len(para.tokens)
    for w in para.words:
        _check_span_bounds(w.span, n, idx, "word", out)
    for s in para.sentences:
        _check_span_bounds(s.span, n, idx, "sentence", out)
    for g in para.grammar:
        for name, target in g.roles:
            if isinstance(target, int):
                if not (0 <= target < n):
                    out.append(Violation(5, Severity.ERROR,
                        f"grammar {g.rule_id} role {name}={target} out of bounds "
                        f"(num_tokens={n})", paragraph_index=idx))
            else:
                _check_span_bounds(target, n, idx, f"grammar {g.rule_id}.{name}", out)


def _check_span_bounds(
    span: Span, n: int, idx: int, what: str, out: list[Violation]
) -> None:
    if span.end > n:
        out.append(Violation(5, Severity.ERROR,
            f"{what} span {span} out of bounds (num_tokens={n})",
            paragraph_index=idx))


def _check_word_and_sentence_tiling(
    para: Paragraph, idx: int, out: list[Violation]
) -> None:
    # Invariant 6
    n = len(para.tokens)
    sorted_words = sorted(para.words, key=lambda w: w.span.start)
    for a, b in zip(sorted_words, sorted_words[1:]):
        if a.span.end > b.span.start:
            out.append(Violation(6, Severity.ERROR,
                f"word spans overlap: {a.span} and {b.span}",
                paragraph_index=idx))

    if not para.sentences or n == 0:
        return
    sorted_sents = sorted(para.sentences, key=lambda s: s.span.start)
    cursor = 0
    tiling_broken = False
    for s in sorted_sents:
        if s.span.start != cursor:
            out.append(Violation(6, Severity.ERROR,
                f"sentence spans don't tile: expected start {cursor}, got {s.span.start}",
                paragraph_index=idx))
            tiling_broken = True
            break
        cursor = s.span.end
    if not tiling_broken and cursor != n:
        out.append(Violation(6, Severity.ERROR,
            f"sentence spans end at {cursor}, expected {n} (num_tokens)",
            paragraph_index=idx))


def _check_grammar(
    para: Paragraph, idx: int, ruleset: dict, out: list[Violation]
) -> None:
    # Invariant 7
    for g in para.grammar:
        rule = ruleset.get(g.rule_id)
        if rule is None:
            out.append(Violation(7, Severity.ERROR,
                f"unknown grammar rule {g.rule_id!r}", paragraph_index=idx))
            continue
        rule_roles = rule.get("roles", {})
        provided = {name for name, _ in g.roles}
        for name, spec in rule_roles.items():
            if isinstance(spec, dict) and spec.get("required") and name not in provided:
                out.append(Violation(7, Severity.ERROR,
                    f"rule {g.rule_id} missing required role {name!r}",
                    paragraph_index=idx))
        for name, _ in g.roles:
            if name not in rule_roles:
                out.append(Violation(7, Severity.ERROR,
                    f"rule {g.rule_id} has unknown role {name!r}",
                    paragraph_index=idx))


def _check_pos(para: Paragraph, idx: int, out: list[Violation]) -> None:
    # Invariant 8
    for tok in para.tokens:
        if tok.pos not in POS_TAGS:
            out.append(Violation(8, Severity.ERROR,
                f"token {tok.id}: pos {tok.pos!r} not in closed set",
                paragraph_index=idx))


def _check_no_reserved(para: Paragraph, idx: int, out: list[Violation]) -> None:
    # Invariant 9
    for tok in para.tokens:
        fields = (tok.surface, tok.dictionary_form, tok.pos, tok.gloss, *tok.ruby)
        for f in fields:
            if contains_reserved(f):
                out.append(Violation(9, Severity.ERROR,
                    f"token {tok.id}: field {f!r} contains reserved PUA codepoint",
                    paragraph_index=idx))
    for w in para.words:
        if contains_reserved(w.dictionary_form) or contains_reserved(w.gloss):
            out.append(Violation(9, Severity.ERROR,
                f"word {w.span}: contains reserved PUA codepoint",
                paragraph_index=idx))
    for s in para.sentences:
        if contains_reserved(s.translation):
            out.append(Violation(9, Severity.ERROR,
                f"sentence {s.span}: translation contains reserved PUA codepoint",
                paragraph_index=idx))
    for g in para.grammar:
        if contains_reserved(g.rule_id):
            out.append(Violation(9, Severity.ERROR,
                "grammar rule_id contains reserved PUA codepoint",
                paragraph_index=idx))


# --- Helpers ---------------------------------------------------------------

def _count_kanji_runs(text: str) -> int:
    count = 0
    in_run = False
    for ch in text:
        if _is_kanji(ch):
            if not in_run:
                count += 1
                in_run = True
        else:
            in_run = False
    return count


def _is_kanji(ch: str) -> bool:
    # CJK Unified Ideographs + common extensions. Kept broad on purpose:
    # anything the tokenizer might see in real Japanese text should count.
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF   # CJK Unified Ideographs
        or 0x3400 <= o <= 0x4DBF   # Extension A
        or 0xF900 <= o <= 0xFAFF   # Compatibility Ideographs
        or 0x20000 <= o <= 0x2A6DF  # Extension B
        or 0x2A700 <= o <= 0x2B73F  # Extension C
        or 0x2B740 <= o <= 0x2B81F  # Extension D
        or 0x2B820 <= o <= 0x2CEAF  # Extension E
    )


# Grammar loading is provided by .grammar — re-export the loader for callers
# who want to introspect the pinned inventory (e.g. the CLI --lint tool).
__all__ = [
    "KNOWN_FORMAT_VERSIONS", "KNOWN_RULESET_VERSIONS", "POS_TAGS",
    "Severity", "Violation", "LintError", "lint",
]
