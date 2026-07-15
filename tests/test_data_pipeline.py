"""Tests for the M2 data-cleaning engine (normalize/filters/pipeline/segment).

Model-based filters (fastText langid, LaBSE) are exercised separately/offline;
here we test the deterministic core that decides most of the corpus quality.
"""

from __future__ import annotations

from kanjiland.data.filters import (
    FilterConfig,
    dedup_key,
    english_latin_ratio,
    has_japanese,
    length_ok,
    ratio_ok,
    script_ok,
)
from kanjiland.data.corpus_io import read_jsonl, reservoir_split_to_file, write_jsonl
from kanjiland.data.normalize import count_pua, normalize_text
from kanjiland.data.pipeline import clean_stream, split_pairs
from kanjiland.data.segment import split_paragraphs, split_sentences_ja
from kanjiland.data.sources.jparacrawl import _parse_line
from kanjiland.data.stats import Funnel, length_stats


# --- normalize ----------------------------------------------------------


def test_normalize_nfkc_folds_width_variants():
    # Fullwidth ASCII and halfwidth katakana fold to canonical forms.
    assert normalize_text("ＡＩ　１２３") == "AI 123"
    assert normalize_text("ｶﾀｶﾅ") == "カタカナ"


def test_normalize_strips_pua_and_controls():
    dirty = "abcdef\x07\tghi"  # PUA separator + bell + tab
    assert count_pua(dirty) == 1
    assert normalize_text(dirty) == "abcdefghi"


def test_normalize_collapses_whitespace_and_is_idempotent():
    once = normalize_text("  hello   world \n\t")
    assert once == "hello world"
    assert normalize_text(once) == once


# --- filters ------------------------------------------------------------


def test_script_detectors():
    assert has_japanese("これは日本語")
    assert has_japanese("カタカナ")
    assert not has_japanese("just english 123")
    assert english_latin_ratio("Hello, world!") == 1.0
    assert english_latin_ratio("これは") == 0.0


def test_length_and_ratio_filters():
    cfg = FilterConfig()
    assert length_ok("日本語", "Japanese", cfg)
    assert not length_ok("", "x", cfg)  # empty ja
    # ratio: en/ja char ratio must be within band
    assert ratio_ok("私は学生です", "I am a student", cfg)
    assert not ratio_ok("あ", "this english side is way too long relative to ja", cfg)


def test_script_ok_rejects_swapped_or_romaji():
    cfg = FilterConfig()
    assert script_ok("これは本です", "This is a book", cfg)
    assert not script_ok("kore wa hon desu", "This is a book", cfg)  # romaji ja
    assert not script_ok("これは本です", "これは本です", cfg)  # en side not english


def test_dedup_key_canonicalizes_case_and_spacing():
    assert dedup_key("日本 語", "Hello World") == dedup_key("日本語", "hello world")


# --- pipeline -----------------------------------------------------------


def test_clean_stream_funnel_and_dedup():
    cfg = FilterConfig()
    pairs = [
        ("私は学生です。", "I am a student."),  # keep
        ("私は学生です。", "I am a student."),  # dup -> drop
        ("", "empty ja"),  # length -> drop
        ("これは本です。", "これは本です。"),  # en side not english -> script drop
        ("あ", "wildly overlong english translation that dwarfs the ja side"),  # ratio
        ("犬が好きです。", "I like dogs."),  # keep
    ]
    funnel = Funnel(source="test")
    seen: set[bytes] = set()
    kept = list(clean_stream(pairs, cfg, funnel, seen, langid_cfg=None, identifier=None))

    assert kept == [("私は学生です。", "I am a student."), ("犬が好きです。", "I like dogs.")]
    assert funnel.input_pairs == 6
    assert funnel.kept == 2
    assert funnel.dropped["dedup"] == 1
    assert funnel.dropped["length"] == 1
    assert funnel.dropped["script"] == 1
    assert funnel.dropped["ratio"] == 1


def test_clean_stream_dedup_is_global_across_calls():
    cfg = FilterConfig()
    seen: set[bytes] = set()
    f1, f2 = Funnel(source="a"), Funnel(source="b")
    p = [("犬が好きです。", "I like dogs.")]
    assert len(list(clean_stream(p, cfg, f1, seen))) == 1
    assert len(list(clean_stream(p, cfg, f2, seen))) == 0  # already seen
    assert f2.dropped["dedup"] == 1


# --- splitting ----------------------------------------------------------


def test_split_pairs_sizes_disjoint_deterministic():
    pairs = [(f"ja{i}", f"en{i}") for i in range(1000)]
    a = split_pairs(pairs, seed=1, valid_size=100, test_size=50)
    b = split_pairs(pairs, seed=1, valid_size=100, test_size=50)
    assert a == b  # reproducible
    assert len(a["valid"]) == 100 and len(a["test"]) == 50 and len(a["train"]) == 850
    allp = a["train"] + a["valid"] + a["test"]
    assert len(set(allp)) == 1000  # partition, no overlap/loss


def test_split_pairs_tiny_corpus_scales_holdout_down():
    pairs = [(f"ja{i}", f"en{i}") for i in range(10)]
    s = split_pairs(pairs, seed=1, valid_size=100, test_size=100)
    assert len(s["train"]) >= 8  # doesn't starve train
    assert len(s["train"]) + len(s["valid"]) + len(s["test"]) == 10


# --- segmentation -------------------------------------------------------


def test_split_paragraphs_on_blank_lines():
    text = "para one\nstill one\n\npara two\n\n\npara three"
    assert split_paragraphs(text) == ["para one\nstill one", "para two", "para three"]


def test_split_sentences_ja_keeps_terminator_and_closer():
    para = "今日はいい天気です。散歩しましょう。「行こう！」と言った。"
    assert split_sentences_ja(para) == [
        "今日はいい天気です。",
        "散歩しましょう。",
        "「行こう！」と言った。",
    ]


def test_split_sentences_ja_no_terminator_is_one_sentence():
    assert split_sentences_ja("終わりのない文") == ["終わりのない文"]


# --- streaming split + IO ----------------------------------------------


def test_reservoir_split_sizes_partition_deterministic(tmp_path):
    pairs = [(f"ja{i}", f"en{i}") for i in range(1000)]
    train_p = tmp_path / "train.jsonl"
    valid, test, total = reservoir_split_to_file(pairs, 1, 100, 50, train_p)
    assert total == 1000
    assert len(valid) == 100 and len(test) == 50
    train = list(read_jsonl(train_p))
    assert len(train) == 850
    # exact partition: no loss, no overlap
    allp = set(train) | set(valid) | set(test)
    assert allp == set(pairs) and len(train) + len(valid) + len(test) == 1000
    # reproducible
    valid2, test2, _ = reservoir_split_to_file(pairs, 1, 100, 50, tmp_path / "t2.jsonl")
    assert valid == valid2 and test == test2


def test_reservoir_holdout_is_uniform(tmp_path):
    # Every item should be equally likely to land in the holdout; over the whole
    # index range the holdout mean index should sit near the middle.
    pairs = [(f"ja{i}", f"en{i}") for i in range(5000)]
    valid, test, _ = reservoir_split_to_file(pairs, 7, 200, 200, tmp_path / "t.jsonl")
    idxs = [int(ja[2:]) for ja, _ in valid + test]
    assert 2000 < sum(idxs) / len(idxs) < 3000  # ~2500 expected


def test_write_read_jsonl_roundtrip(tmp_path):
    rows = [("日本語", "Japanese"), ("猫", "cat")]
    p = tmp_path / "x.jsonl"
    assert write_jsonl(p, rows) == 2
    assert list(read_jsonl(p)) == rows


# --- JParaCrawl line parsing -------------------------------------------


def test_jparacrawl_parse_locates_ja_by_script():
    # v3.0 format: url_en \t url_ja \t score \t en \t ja
    line = "http://a.com\thttp://b.jp\t0.82\tI am a student.\t私は学生です。"
    assert _parse_line(line, None) == ("私は学生です。", "I am a student.")


def test_jparacrawl_bicleaner_floor_drops_low_scores():
    line = "http://a.com\thttp://b.jp\t0.30\tI am a student.\t私は学生です。"
    assert _parse_line(line, min_bicleaner=0.5) is None
    assert _parse_line(line, min_bicleaner=0.2) is not None


# --- stats --------------------------------------------------------------


def test_length_stats_basic():
    s = length_stats([1, 2, 3, 4, 100])
    assert s["p50"] == 3 and s["max"] == 100
