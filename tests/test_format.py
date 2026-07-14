"""M0 acceptance tests. These define "done" for the format library.

They are written against the debug rendering (⟨T⟩ etc.) via from_debug()
so fixtures stay human-readable. All are xfail until implemented — flipping
them green IS milestone M0.
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

M0 = pytest.mark.xfail(raises=NotImplementedError, reason="M0: implement me")

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

@M0
def test_parse_worked_example():
    from kanjiland.format.parser import parse

    assert parse(WIRE_EXAMPLE) == DOC_EXAMPLE


@M0
def test_serialize_worked_example():
    from kanjiland.format.serializer import serialize

    assert serialize(DOC_EXAMPLE) == WIRE_EXAMPLE


@M0
def test_round_trip():
    from kanjiland.format.parser import parse
    from kanjiland.format.serializer import serialize

    assert parse(serialize(DOC_EXAMPLE)) == DOC_EXAMPLE


@M0
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


@M0
def test_parse_error_on_garbage():
    from kanjiland.format.parser import ParseError, parse

    with pytest.raises(ParseError):
        parse("not the wire format at all")


@M0
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

@M0
def test_lint_clean_document_has_no_violations():
    from kanjiland.format.linter import lint

    assert lint(DOC_EXAMPLE, source_paragraphs=["私は学生です。"]) == []


@M0
def test_lint_catches_reconstruction_mismatch():
    from kanjiland.format.linter import lint

    violations = lint(DOC_EXAMPLE, source_paragraphs=["私は先生です。"])
    assert any(v.invariant == 3 for v in violations)


@M0
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


@M0
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
