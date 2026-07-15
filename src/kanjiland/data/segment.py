"""Paragraph and sentence segmentation policy.

The translation corpora (M3) arrive already sentence-aligned, so the pipeline
that builds train/valid.jsonl does *not* need this. It exists for the
**annotation track** (M7+): FORMAT_SPEC §3 says the system pre-splits input
into paragraphs, each annotated independently with its own ⟨P⟩ boundary and
paragraph-relative token IDs. This module is that pre-splitter's policy, kept
here so it's tested and versioned before M7 depends on it.

Two levels:

- **Paragraphs**: split on blank lines (one or more). This is the unit of
  independent annotation — bounded context, parallelizable (ADR-006).
- **Sentences** (within a paragraph): Japanese sentence-final punctuation is
  unambiguous enough (。！？ and their halfwidth forms) that a rule beats a
  model here. We keep the terminator attached to its sentence and treat a
  closing quote/bracket immediately after a terminator as part of the same
  sentence (。」 stays together).

Deliberately dependency-free and conservative: when unsure, *under*-split.
An over-eager sentence splitter creates misaligned fragments, which is exactly
the kind of silent data corruption M2 is trying to avoid.
"""

from __future__ import annotations

import re

# Blank-line paragraph break: a newline, optional whitespace, then newline(s).
_PARAGRAPH_BREAK = re.compile(r"\n[ \t　]*\n+")

# Japanese sentence terminators (full and half width).
_JA_TERMINATORS = "。！？!?"
# Matched quote/bracket pairs. We track nesting depth and only treat a
# terminator as sentence-final at depth 0, so a terminator *inside* a quote
# (「行こう！」と言った。) doesn't wrongly split the sentence.
_OPENERS = "「『（(〈《【〔"
_CLOSERS = "」』）)〉》】〕"


def split_paragraphs(text: str) -> list[str]:
    """Split ``text`` into paragraphs on blank lines. Empty paragraphs dropped;
    each returned paragraph has its own leading/trailing whitespace stripped and
    internal single newlines preserved (the caller decides what to do with
    intra-paragraph line breaks)."""
    parts = _PARAGRAPH_BREAK.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_sentences_ja(paragraph: str) -> list[str]:
    """Split a Japanese paragraph into sentences on 。！？ at quote depth 0.

    A terminator inside a quote or bracket (「…！」) does not split; a run of
    text with no depth-0 terminator is returned as a single sentence. The
    terminator, and any unmatched closer immediately after it, stay with the
    sentence they end."""
    sentences: list[str] = []
    start = 0
    depth = 0
    i = 0
    n = len(paragraph)
    while i < n:
        ch = paragraph[i]
        if ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS and depth > 0:
            depth -= 1
        elif ch in _JA_TERMINATORS and depth == 0:
            end = i + 1
            while end < n and paragraph[end] in _CLOSERS:  # trailing unmatched closer
                end += 1
            chunk = paragraph[start:end].strip()
            if chunk:
                sentences.append(chunk)
            start = end
            i = end
            continue
        i += 1
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
