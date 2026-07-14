"""Wire-format separators (FORMAT_SPEC.md §2).

Real separators are PUA codepoints; DEBUG_NAMES is the human-readable
rendering used by the pretty-printer and in documentation.
"""

HEADER = "\ue000"      # ⟨H⟩
TOKEN = "\ue001"       # ⟨T⟩
WORD = "\ue002"        # ⟨W⟩
SENTENCE = "\ue003"    # ⟨S⟩
GRAMMAR = "\ue004"     # ⟨G⟩
PARAGRAPH = "\ue005"   # ⟨P⟩
LIST_SEP = "\ue00e"    # ⟨L⟩
FIELD_SEP = "\ue00f"   # ⟨F⟩
RECORD_END = "\ue010"  # ⟨E⟩

RECORD_STARTS = {HEADER, TOKEN, WORD, SENTENCE, GRAMMAR, PARAGRAPH}

# Reserved range: content fields must not contain these (linter invariant 9).
RESERVED_LO, RESERVED_HI = 0xE000, 0xE01F

DEBUG_NAMES = {
    HEADER: "⟨H⟩",
    TOKEN: "⟨T⟩",
    WORD: "⟨W⟩",
    SENTENCE: "⟨S⟩",
    GRAMMAR: "⟨G⟩",
    PARAGRAPH: "⟨P⟩",
    LIST_SEP: "⟨L⟩",
    FIELD_SEP: "⟨F⟩",
    RECORD_END: "⟨E⟩",
}


def to_debug(wire: str) -> str:
    """Render wire text with human-readable separator names."""
    for cp, name in DEBUG_NAMES.items():
        wire = wire.replace(cp, name)
    return wire


def from_debug(debug: str) -> str:
    """Inverse of to_debug — handy for writing readable test fixtures."""
    for cp, name in DEBUG_NAMES.items():
        debug = debug.replace(name, cp)
    return debug


def contains_reserved(text: str) -> bool:
    """True if text contains any reserved PUA codepoint (invariant 9)."""
    return any(RESERVED_LO <= ord(ch) <= RESERVED_HI for ch in text)
