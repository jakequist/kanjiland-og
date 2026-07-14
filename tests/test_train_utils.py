"""Tests for train utilities (seeding + W&B init).

The W&B test runs in ``WANDB_MODE=disabled`` so it never touches the
network or writes to disk beyond a temp dir.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def test_seed_everything_is_reproducible():
    from kanjiland.train.seed import seed_everything

    seed_everything(1234)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(1234)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_wandb_init_disabled_mode(tmp_path: Path, monkeypatch):
    """Smoke test: init_run parses config, spins up wandb, and returns a
    Run object even with W&B disabled (no network, no files)."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")

    cfg = {
        "run_name": "unit-test-run",
        "seed": 42,
        "wandb": {"project": "kanjiland-tests", "tags": ["unit"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from kanjiland.train.wandb_init import init_run
    run = init_run(cfg_path)
    try:
        # config was uploaded
        assert run.config["seed"] == 42
        # our metadata namespace was attached
        assert "_run" in run.config
        meta = run.config["_run"]
        assert "git_sha" in meta
        assert "torch_version" in meta
    finally:
        run.finish()


def test_wandb_init_rejects_missing_config(tmp_path: Path):
    from kanjiland.train.wandb_init import init_run
    import pytest
    with pytest.raises(FileNotFoundError):
        init_run(tmp_path / "no_such_config.yaml")
