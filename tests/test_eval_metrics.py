"""Tests for the M4 eval harness: metrics + the seed-aware results store."""

from __future__ import annotations

import pytest

from kanjiland.eval import results


# --- results store (pure, no heavy deps) --------------------------------


def test_upsert_replaces_same_key_appends_new_seed():
    recs: list[dict] = []
    base = {"run": "a", "test_set": "t", "beam": 4}
    results.upsert(recs, {**base, "seed": 1, "metrics": {"chrf": 50.0}})
    results.upsert(recs, {**base, "seed": 1, "metrics": {"chrf": 60.0}})  # same key -> replace
    assert len(recs) == 1 and recs[0]["metrics"]["chrf"] == 60.0
    results.upsert(recs, {**base, "seed": 2, "metrics": {"chrf": 55.0}})  # new seed -> append
    assert len(recs) == 2


def test_render_single_seed_is_bare_number():
    recs = [
        {
            "run": "m",
            "test_set": "kftt",
            "seed": 1,
            "beam": 4,
            "metrics": {"chrf": 47.2, "bleu": 21.0, "comet": 0.82},
        }
    ]
    md = results.render_markdown(recs)
    assert "47.20" in md
    assert "47.20 ±" not in md  # one seed -> bare number, not mean±std
    assert "| 1 |" in md  # seed count column


def test_render_two_seeds_shows_mean_and_std():
    recs = [
        {"run": "m", "test_set": "kftt", "seed": 1, "beam": 4, "metrics": {"chrf": 47.0}},
        {"run": "m", "test_set": "kftt", "seed": 2, "beam": 4, "metrics": {"chrf": 47.4}},
    ]
    md = results.render_markdown(recs)
    assert "47.20 ± 0.20" in md  # mean 47.2, population std 0.2


def test_load_missing_returns_empty(tmp_path):
    assert results.load(tmp_path / "nope.json") == []


def test_save_load_roundtrip(tmp_path):
    recs = [{"run": "m", "test_set": "t", "seed": 1, "beam": 4, "metrics": {"chrf": 50.0}}]
    p = tmp_path / "r.json"
    results.save(p, recs)
    assert results.load(p) == recs


# --- metrics (need the `eval` extra: sacrebleu) -------------------------


def test_chrf_perfect_vs_mismatch():
    pytest.importorskip("sacrebleu")
    from kanjiland.eval import metrics

    assert metrics.chrf(["the cat sat"], ["the cat sat"]) > 99.0
    assert metrics.chrf(["totally unrelated words"], ["the cat sat"]) < 40.0


def test_bleu_perfect_and_signature():
    pytest.importorskip("sacrebleu")
    from kanjiland.eval import metrics

    score, sig = metrics.bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert score > 99.0
    assert isinstance(sig, str) and sig  # reproducibility signature present
