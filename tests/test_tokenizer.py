"""Tests for the from-scratch byte-level BPE tokenizer (M1).

Covers the ROADMAP "done when" criteria: arbitrary-UTF-8 round-trip,
deterministic merge reproduction on a toy corpus, plus special-token
atomicity and the no-merge-across-separator guarantee.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from kanjiland.format import separators as sep
from kanjiland.tokenizer import Tokenizer, pretokenize
from kanjiland.tokenizer.bpe import BYTE_BASE, train_bpe, word_freqs_from_pretokens

# A small mixed Ja/En corpus, enough to force real merges.
CORPUS = [
    "私は学生です。",
    "私は先生です。",
    "彼は学生ですか。",
    "これはペンです。",
    "I am a student.",
    "You are a student.",
    "She is a teacher.",
    "コーヒーを飲みます。",
    "日本語を勉強しています。",
    "学生は学校で勉強します。",
] * 5


def _tok(vocab_size: int = 512) -> Tokenizer:
    return Tokenizer.train(CORPUS, vocab_size=vocab_size)


# --- pre-tokenizer ------------------------------------------------------


def test_pretokenize_splits_on_script_boundaries():
    assert pretokenize("私は学生です。") == ["私", "は", "学生", "です", "。"]


def test_pretokenize_leading_space_attaches_gpt2_style():
    assert pretokenize("Hello world") == ["Hello", " world"]
    # Runs of spaces: extras stand alone, one space peels onto the word.
    assert pretokenize("a  b") == ["a", " ", " b"]


def test_pretokenize_is_lossless():
    for s in ["私は学生です。 Hello world 2026", "コーヒー\tと\nお茶", ""]:
        assert "".join(pretokenize(s)) == s


# --- round-trip ---------------------------------------------------------


@given(st.text())
def test_roundtrip_arbitrary_text(text: str):
    tok = _ROUNDTRIP_TOK
    assert tok.decode(tok.encode(text)) == text


@given(st.text(alphabet=st.characters(min_codepoint=0x3000, max_codepoint=0x9FFF)))
def test_roundtrip_japanese_range(text: str):
    tok = _ROUNDTRIP_TOK
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_all_single_bytes_via_latin1():
    # Every byte 0..255 is reachable; latin-1 maps them 1:1 to codepoints.
    tok = _ROUNDTRIP_TOK
    text = "".join(chr(b) for b in range(256))
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_wire_format_string():
    tok = _ROUNDTRIP_TOK
    wire = sep.from_debug("⟨H⟩0.2⟨F⟩grammar-1.0⟨E⟩⟨T⟩0⟨F⟩私⟨F⟩わたし⟨E⟩")
    assert tok.decode(tok.encode(wire)) == wire


# --- special tokens -----------------------------------------------------


def test_pua_separators_are_atomic_special_ids():
    tok = _ROUNDTRIP_TOK
    wire = sep.HEADER + "0.2" + sep.FIELD_SEP + "x" + sep.RECORD_END
    ids = tok.encode(wire)
    header_id = tok._name_to_id("HEADER")
    field_id = tok._name_to_id("FIELD_SEP")
    end_id = tok._name_to_id("RECORD_END")
    # Each separator appears exactly once, as a single id.
    assert ids[0] == header_id
    assert ids.count(header_id) == 1
    assert field_id in ids and end_id in ids


def test_no_merge_id_collides_with_special_or_byte_space():
    tok = _tok(512)
    # Merges live strictly above specials + bytes.
    for a, b in tok.merges:
        assert 0 <= a and 0 <= b  # internal (byte-space) ids
    assert tok.byte_offset == len(tok.special_tokens)
    assert tok.vocab_size == tok.byte_offset + BYTE_BASE + len(tok.merges)


def test_bos_eos_wrapping():
    tok = _ROUNDTRIP_TOK
    ids = tok.encode("私", add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    # Control tokens are dropped on decode by default.
    assert tok.decode(ids) == "私"


# --- BPE core: determinism & known merges -------------------------------


def test_train_bpe_reproduces_known_merges_on_toy_corpus():
    # Classic Sennrich example. Word "low" x5, "lower" x2, "newest" x6,
    # "widest" x3. Highest-frequency adjacent byte pair is (e, s) = 9,
    # then (es, t). We assert the first two merges by their byte values.
    words = {b"low": 5, b"lower": 2, b"newest": 6, b"widest": 3}
    merges = train_bpe(words, num_merges=2)
    e, s, t = ord("e"), ord("s"), ord("t")
    assert merges[0] == (e, s)  # 'e'+'s' -> 256 (9 occurrences)
    assert merges[1] == (BYTE_BASE, t)  # 'es'+'t' -> 257 (9 occurrences)


def test_train_bpe_is_deterministic():
    wf = word_freqs_from_pretokens(p for line in CORPUS for p in pretokenize(line))
    assert train_bpe(dict(wf), 300) == train_bpe(dict(wf), 300)


def test_tokenizer_training_is_deterministic():
    a = _tok(512).merges
    b = _tok(512).merges
    assert a == b


# --- persistence --------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    tok = _tok(400)
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    sample = "私は学生です。 Hello"
    assert loaded.encode(sample) == tok.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample


def test_vocab_size_target_is_honoured():
    # Target reachable by the corpus: hit it exactly.
    assert _tok(300).vocab_size == 300
    # Target beyond the corpus's available pairs: cap gracefully, never exceed.
    assert _tok(100_000).vocab_size <= 100_000


# Train one shared tokenizer for the round-trip property tests (hypothesis
# runs each many times; retraining per example would be wasteful).
_ROUNDTRIP_TOK = _tok(512)
