"""Device + mixed-precision helpers — portable across CUDA, Apple MPS, and CPU.

Keeps the "where does this run and in what precision" decision in one place so
the training code stays device-agnostic. Order of preference: an NVIDIA GPU
(CUDA) if present, else Apple's GPU (MPS/Metal), else the CPU.
"""

from __future__ import annotations

import contextlib

import torch


def pick_device() -> str:
    """Best available accelerator: CUDA > MPS (Apple GPU) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    # torch.backends.mps exists on all recent builds but is only "available" on
    # Apple silicon with a Metal-capable PyTorch; guard defensively either way.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def amp_context(device: str):
    """Mixed-precision context for a forward pass.

    bf16 autocast is a real speedup on CUDA tensor-core GPUs, so we enable it
    there. On MPS/CPU we return a no-op context and run plain fp32 — there's no
    tensor-core win to capture, and bf16 autocast on MPS is unreliable. Same
    math everywhere, just slower off-CUDA (fine for small runs).
    """
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()
