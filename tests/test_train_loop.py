"""Tests for the training pieces: label-smoothed loss + LR schedule (M3)."""

from __future__ import annotations

import torch

from kanjiland.train.loss import label_smoothed_ce
from kanjiland.train.schedule import lr_at_step


def test_loss_ignores_pad_tokens():
    logits = torch.randn(2, 4, 10)
    target = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # 0 = pad
    _, ntok = label_smoothed_ce(logits, target, pad_id=0)
    assert ntok.item() == 5  # 3 + 2 non-pad


def test_loss_lower_when_confidently_correct():
    target = torch.tensor([[1, 2]])
    good = torch.zeros(1, 2, 10)
    good[0, 0, 1] = 10.0
    good[0, 1, 2] = 10.0
    bad = torch.zeros(1, 2, 10)
    bad[0, 0, 5] = 10.0
    bad[0, 1, 7] = 10.0
    lg, _ = label_smoothed_ce(good, target, pad_id=0)
    lb, _ = label_smoothed_ce(bad, target, pad_id=0)
    assert lg.item() < lb.item()


def test_loss_smoothing_keeps_a_floor():
    # With smoothing, even a perfect prediction has non-zero loss (the model is
    # trained to keep a little mass on other tokens).
    target = torch.tensor([[1]])
    perfect = torch.zeros(1, 1, 10)
    perfect[0, 0, 1] = 100.0
    loss, ntok = label_smoothed_ce(perfect, target, pad_id=0, smoothing=0.1)
    assert loss.item() / ntok.item() > 0.0


def test_lr_schedule_shape():
    peak, warmup = 5e-4, 1000
    assert abs(lr_at_step(warmup, peak, warmup) - peak) < 1e-12  # peaks at warmup
    assert abs(lr_at_step(500, peak, warmup) - peak * 0.5) < 1e-12  # linear ramp
    # inverse-sqrt decay after warmup
    assert abs(lr_at_step(4000, peak, warmup) - peak * (warmup / 4000) ** 0.5) < 1e-12
    assert lr_at_step(4000, peak, warmup) < peak
