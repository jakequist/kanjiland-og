"""M0 acceptance tests. These define "done" for the format library.

Fixtures use the debug rendering (⟨T⟩ etc.) via from_debug() to stay
readable in source.
"""

import pytest

from kanjiland.format import (
    Document,
    GrammarAnnotation,
    Header,
    Paragraph,
    Sentence,
    Span,
    Token,
    Word,
    from_debug,
)

# --- Fixture: the worked example from FORMAT_SPEC.md §5 --------------------

WIRE_EXAMPLE = from_debug(
    "⟨H⟩0.2⟨F⟩grammar-0.1⟨E⟩\n"
    "⟨P⟩⟨E⟩\n"
    "⟨T⟩0⟨F⟩私⟨F⟩わたし⟨F⟩私⟨F⟩PRON⟨F⟩I⟨E⟩\n"
    "⟨T⟩1⟨F⟩は⟨F⟩⟨F⟩は⟨F⟩PART⟨F⟩(topic marker)⟨E⟩\n"
    "⟨T⟩2⟨F⟩学生⟨F⟩がくせい⟨F⟩学生⟨F⟩NOUN⟨F⟩student⟨E⟩\n"
    "⟨T⟩3⟨F⟩です⟨F⟩⟨F⟩です⟨F⟩COP⟨F⟩am (polite)⟨E⟩\n"
    "⟨T⟩4⟨F⟩。⟨F⟩⟨F⟩。⟨F⟩PUNCT⟨F⟩⟨E⟩\n"
    "⟨S⟩0:5⟨F⟩I am a student.⟨E⟩\n"
    "⟨G⟩TOPIC_WA⟨F⟩topic=0⟨F⟩marker=1⟨F⟩scope=2:5⟨E⟩\n"
    "⟨G⟩COPULA_POLITE⟨F⟩complement=2⟨F⟩copula=3⟨E⟩\n"
)

DOC_EXAMPLE = Document(
    header=Header("0.2", "grammar-0.1"),
    paragraphs=[
        Paragraph(
            tokens=[
                Token(0, "私", ("わたし",), "私", "PRON", "I"),
                Token(1, "は", (), "は", "PART", "(topic marker)"),
                Token(2, "学生", ("がくせい",), "学生", "NOUN", "student"),
                Token(3, "です", (), "です", "COP", "am (polite)"),
                Token(4, "。", (), "。", "PUNCT", ""),
            ],
            sentences=[Sentence(Span(0, 5), "I am a student.")],
            grammar=[
                GrammarAnnotation(
                    "TOPIC_WA",
                    (("topic", 0), ("marker", 1), ("scope", Span(2, 5))),
                ),
                GrammarAnnotation(
                    "COPULA_POLITE",
                    (("complement", 2), ("copula", 3)),
                ),
            ],
        )
    ],
)


# --- Parser / serializer ----------------------------------------------------

def test_parse_worked_example():
    from kanjiland.format.parser import parse

    assert parse(WIRE_EXAMPLE) == DOC_EXAMPLE


def test_serialize_worked_example():
    from kanjiland.format.serializer import serialize

    assert serialize(DOC_EXAMPLE) == WIRE_EXAMPLE


def test_round_trip():
    from kanjiland.format.parser import parse
    from kanjiland.format.serializer import serialize

    assert parse(serialize(DOC_EXAMPLE)) == DOC_EXAMPLE


def test_multi_kanji_run_ruby():
    """取り引き: two kanji runs -> two ruby entries joined by ⟨L⟩."""
    from kanjiland.format.parser import parse

    wire = from_debug(
        "⟨H⟩0.2⟨F⟩grammar-0.1⟨E⟩\n⟨P⟩⟨E⟩\n"
        "⟨T⟩0⟨F⟩取り引き⟨F⟩と⟨L⟩ひ⟨F⟩取り引き⟨F⟩NOUN⟨F⟩transaction⟨E⟩\n"
        "⟨S⟩0:1⟨F⟩A transaction.⟨E⟩\n"
    )
    doc = parse(wire)
    assert doc.paragraphs[0].tokens[0].ruby == ("と", "ひ")


def test_parse_error_on_garbage():
    from kanjiland.format.parser import ParseError, parse

    with pytest.raises(ParseError):
        parse("not the wire format at all")


def test_serializer_rejects_reserved_codepoints_in_content():
    from kanjiland.format.serializer import serialize

    bad = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(tokens=[Token(0, "x\ue00fy", (), "x", "NOUN", "g")])
        ],
    )
    with pytest.raises(ValueError):
        serialize(bad)


# --- Linter (one test per SPEC §7 invariant; add the rest as implemented) --

def test_lint_clean_document_has_no_violations():
    from kanjiland.format.linter import lint

    assert lint(DOC_EXAMPLE, source_paragraphs=["私は学生です。"]) == []


def test_lint_catches_reconstruction_mismatch():
    from kanjiland.format.linter import lint

    violations = lint(DOC_EXAMPLE, source_paragraphs=["私は先生です。"])
    assert any(v.invariant == 3 for v in violations)


def test_lint_catches_nonsequential_token_ids():
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=[
                    Token(0, "私", ("わたし",), "私", "PRON", "I"),
                    Token(2, "は", (), "は", "PART", ""),  # gap: 1 missing
                ]
            )
        ],
    )
    assert any(v.invariant == 2 for v in lint(doc))


def test_lint_catches_sentence_spans_not_tiling():
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=DOC_EXAMPLE.paragraphs[0].tokens,
                sentences=[Sentence(Span(0, 3), "partial")],  # doesn't cover 3:5
            )
        ],
    )
    assert any(v.invariant == 6 for v in lint(doc))


# --- Span basics (no implementation needed — should pass immediately) ------

def test_span_parse_and_str():
    s = Span.parse("2:5")
    assert (s.start, s.end) == (2, 5)
    assert str(s) == "2:5"


def test_span_rejects_invalid():
    with pytest.raises(ValueError):
        Span(3, 3)
    with pytest.raises(ValueError):
        Span(-1, 2)


def test_debug_rendering_round_trip():
    from kanjiland.format import from_debug, to_debug

    debug = "⟨T⟩0⟨F⟩私⟨E⟩"
    assert to_debug(from_debug(debug)) == debug


# --- Additional §7 invariant coverage --------------------------------------

def test_lint_catches_ruby_count_mismatch():
    """Invariant 4: ruby entries must equal the number of maximal kanji runs."""
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(tokens=[
                # 学生 is ONE kanji run but we give TWO ruby entries.
                Token(0, "学生", ("がく", "せい"), "学生", "NOUN", "student"),
            ])
        ],
    )
    assert any(v.invariant == 4 for v in lint(doc))


def test_lint_counts_split_kanji_runs():
    """Invariant 4: 取り引き is two runs (取, 引) so needs two ruby entries."""
    from kanjiland.format.linter import lint

    good = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(tokens=[
                Token(0, "取り引き", ("と", "ひ"), "取り引き", "NOUN", "transaction"),
            ])
        ],
    )
    assert not any(v.invariant == 4 for v in lint(good))

    bad = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(tokens=[
                Token(0, "取り引き", ("とりひき",), "取り引き", "NOUN", "transaction"),
            ])
        ],
    )
    assert any(v.invariant == 4 for v in lint(bad))


def test_lint_catches_out_of_bounds_span():
    """Invariant 5: sentence span extends past num_tokens."""
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=[Token(0, "私", ("わたし",), "私", "PRON", "I")],
                sentences=[Sentence(Span(0, 5), "out of range")],
            )
        ],
    )
    assert any(v.invariant == 5 for v in lint(doc))


def test_lint_catches_overlapping_word_spans():
    """Invariant 6: word spans must not overlap."""
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=DOC_EXAMPLE.paragraphs[0].tokens,
                words=[Word(Span(0, 3), "x", ""), Word(Span(2, 5), "y", "")],
                sentences=[Sentence(Span(0, 5), "I am a student.")],
            )
        ],
    )
    assert any(v.invariant == 6 for v in lint(doc))


def test_lint_catches_unknown_grammar_rule():
    """Invariant 7: rule_id must exist in the pinned ruleset."""
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=DOC_EXAMPLE.paragraphs[0].tokens,
                sentences=[Sentence(Span(0, 5), "I am a student.")],
                grammar=[GrammarAnnotation("NO_SUCH_RULE", (("x", 0),))],
            )
        ],
    )
    assert any(v.invariant == 7 for v in lint(doc))


def test_lint_catches_missing_required_role():
    """Invariant 7: required roles must be present."""
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=DOC_EXAMPLE.paragraphs[0].tokens,
                sentences=[Sentence(Span(0, 5), "I am a student.")],
                # TOPIC_WA requires 'topic' and 'marker'; we only give marker.
                grammar=[GrammarAnnotation("TOPIC_WA", (("marker", 1),))],
            )
        ],
    )
    assert any(v.invariant == 7 for v in lint(doc))


def test_lint_catches_bad_pos_tag():
    """Invariant 8: pos must be in the closed set."""
    from kanjiland.format.linter import lint

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(tokens=[
                Token(0, "私", ("わたし",), "私", "NONSENSE", "I"),
            ])
        ],
    )
    assert any(v.invariant == 8 for v in lint(doc))


def test_lint_catches_unknown_header_versions():
    """Invariant 1: format_version and ruleset_version must be known."""
    from kanjiland.format.linter import lint

    doc = Document(header=Header("99.9", "grammar-99"), paragraphs=[])
    violations = lint(doc)
    assert any(v.invariant == 1 for v in violations)


def test_lint_strict_mode_raises():
    from kanjiland.format.linter import LintError, lint

    with pytest.raises(LintError):
        lint(DOC_EXAMPLE, source_paragraphs=["私は先生です。"], mode="strict")


# --- Round-trip property tests ---------------------------------------------

def test_round_trip_serialize_then_parse_multi_paragraph():
    from kanjiland.format.parser import parse
    from kanjiland.format.serializer import serialize

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            DOC_EXAMPLE.paragraphs[0],
            Paragraph(
                tokens=[
                    Token(0, "はい", (), "はい", "INTERJ", "yes"),
                    Token(1, "。", (), "。", "PUNCT", ""),
                ],
                sentences=[Sentence(Span(0, 2), "Yes.")],
            ),
        ],
    )
    assert parse(serialize(doc)) == doc


def test_round_trip_with_word_and_multi_kanji():
    from kanjiland.format.parser import parse
    from kanjiland.format.serializer import serialize

    doc = Document(
        header=Header("0.2", "grammar-0.1"),
        paragraphs=[
            Paragraph(
                tokens=[
                    Token(0, "取り引き", ("と", "ひ"), "取り引き", "NOUN", "transaction"),
                    Token(1, "を", (), "を", "PART", "(object marker)"),
                    Token(2, "する", (), "する", "VERB", "do"),
                    Token(3, "。", (), "。", "PUNCT", ""),
                ],
                words=[Word(Span(1, 3), "する", "do a transaction")],
                sentences=[Sentence(Span(0, 4), "Do a transaction.")],
            )
        ],
    )
    assert parse(serialize(doc)) == doc


def test_parse_rejects_unknown_record_tag():
    from kanjiland.format.parser import ParseError, parse

    wire = from_debug("⟨H⟩0.2⟨F⟩grammar-0.1⟨E⟩\n") + "\ue00c" + "\ue010\n"
    with pytest.raises(ParseError):
        parse(wire)


def test_parse_rejects_missing_header():
    from kanjiland.format.parser import ParseError, parse

    wire = from_debug("⟨P⟩⟨E⟩\n")
    with pytest.raises(ParseError):
        parse(wire)


def test_parse_rejects_record_before_paragraph():
    from kanjiland.format.parser import ParseError, parse

    wire = from_debug(
        "⟨H⟩0.2⟨F⟩grammar-0.1⟨E⟩\n"
        "⟨T⟩0⟨F⟩私⟨F⟩わたし⟨F⟩私⟨F⟩PRON⟨F⟩I⟨E⟩\n"
    )
    with pytest.raises(ParseError):
        parse(wire)
