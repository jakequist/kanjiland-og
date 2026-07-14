"""Determinism helpers (CLAUDE.md rule 6).

Every training run seeds Python's ``random``, NumPy, and PyTorch (CPU +
CUDA) from a single integer. Deterministic cuDNN is enabled; keep in mind
that some ops (e.g. atomic reductions) remain nondeterministic on GPU and
the eventual reproducibility guarantee is "same seed on the same hardware,
same package versions".
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed all RNGs and return the seed for logging convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
