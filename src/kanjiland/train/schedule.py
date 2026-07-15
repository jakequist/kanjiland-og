"""Learning-rate schedule: linear warmup, then inverse-square-root decay.

Transformers are famously touchy about the early optimization steps — the
attention softmax and LayerNorms produce large, noisy gradients before anything
has organized itself, and a full learning rate then can diverge. So we **warm
up** linearly from 0 to the peak LR over the first ``warmup`` steps, then **decay
as 1/√step**, which keeps steps large early (fast progress) and small late
(fine-tuning). This is the original Transformer schedule, re-parameterized to
peak at a configured ``peak_lr`` instead of being implicitly set by d_model.
"""

from __future__ import annotations


def lr_at_step(step: int, peak_lr: float, warmup: int) -> float:
    """LR for a 1-based ``step``.

    step ≤ warmup:  peak_lr · step / warmup            (linear warmup)
    step >  warmup:  peak_lr · sqrt(warmup / step)      (inverse-sqrt decay)

    The two pieces meet exactly at ``peak_lr`` when step == warmup.
    """
    step = max(1, step)
    if step <= warmup:
        return peak_lr * step / warmup
    return peak_lr * (warmup / step) ** 0.5
