"""Tests for the Tatoeba downloader. Network is mocked; we assemble a
tiny in-memory zip that matches the ManyThings format."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path


from kanjiland.data import tatoeba


SAMPLE_LINES = [
    "Hello.\tこんにちは。\tCC-BY 2.0 (France) Attribution: tatoeba.org #1 (test) & #2 (test)",
    "I am a student.\t私は学生です。\tCC-BY 2.0 (France) Attribution: tatoeba.org #3 (test) & #4 (test)",
    "Thank you.\tありがとう。",  # attribution missing — should still parse
    "",                            # blank line — should be skipped
    "orphan",                       # missing tab — should be skipped
]


def _fake_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(tatoeba.INNER_FILENAME, "\n".join(SAMPLE_LINES))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_download_writes_expected_outputs(tmp_path: Path, monkeypatch):
    fake = _fake_zip()
    monkeypatch.setattr(
        tatoeba, "__name__", tatoeba.__name__
    )  # keep module ref alive

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResponse(fake))

    stats = tatoeba.download(tmp_path)

    assert (tmp_path / "pairs.tsv").exists()
    assert (tmp_path / "ja.txt").exists()
    assert (tmp_path / "en.txt").exists()
    assert (tmp_path / "stats.json").exists()

    assert stats.pair_count == 3
    assert stats.source_url == tatoeba.DEFAULT_URL

    ja = (tmp_path / "ja.txt").read_text().splitlines()
    en = (tmp_path / "en.txt").read_text().splitlines()
    assert ja == ["こんにちは。", "私は学生です。", "ありがとう。"]
    assert en == ["Hello.", "I am a student.", "Thank you."]

    persisted = json.loads((tmp_path / "stats.json").read_text())
    assert persisted["pair_count"] == 3
    assert persisted["sha256"]  # non-empty


def test_download_is_idempotent(tmp_path: Path, monkeypatch):
    fake = _fake_zip()
    call_count = {"n": 0}

    def fake_get(*a, **kw):
        call_count["n"] += 1
        return _FakeResponse(fake)

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    tatoeba.download(tmp_path)
    tatoeba.download(tmp_path)  # should skip re-download
    assert call_count["n"] == 1

    tatoeba.download(tmp_path, force=True)
    assert call_count["n"] == 2


def test_stats_percentiles_stable_for_small_sample():
    pairs = [("a b c", "あ", ""), ("a b", "あい", ""), ("a", "あいう", "")]
    stats = tatoeba._compute_stats(pairs, url="x", sha256="y")
    assert stats.pair_count == 3
    assert stats.ja_char_len_p50 == 2
    assert stats.en_word_len_p50 == 2


def test_parse_pairs_skips_malformed():
    parsed = list(tatoeba._parse_pairs(SAMPLE_LINES))
    assert len(parsed) == 3
    assert parsed[2] == ("Thank you.", "ありがとう。", "")
