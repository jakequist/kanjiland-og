"""Label-smoothed cross-entropy — the standard NMT training loss.

Plain cross-entropy pushes the model to put *all* probability on the gold token,
which encourages over-confidence and hurts generalization. **Label smoothing**
(Szegedy et al.; standard since the original Transformer) softens the target:
the correct token gets probability ``1 − ε`` and the remaining ``ε`` is spread
uniformly over the vocabulary. The model is trained to be a little unsure, which
calibrates it better and consistently improves BLEU/chrF a point or so.

Padding positions are excluded from the loss entirely (they carry no signal).
We return the summed loss and the token count so the training loop can average
correctly across gradient-accumulation microbatches of differing size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def label_smoothed_ce(
    logits: Tensor,
    target: Tensor,
    pad_id: int,
    smoothing: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Return (summed_loss, n_tokens).

    ``logits``: (B, S, V); ``target``: (B, S). Loss is summed over non-pad
    positions; divide by ``n_tokens`` for the per-token mean.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)  # float32 for stable loss

    # nll = −log p(gold); smooth = −mean_v log p(v) (the uniform part).
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, S)
    smooth = -log_probs.mean(dim=-1)  # (B, S)
    loss = (1.0 - smoothing) * nll + smoothing * smooth

    mask = target != pad_id
    n_tokens = mask.sum()
    loss = torch.where(mask, loss, torch.zeros_like(loss)).sum()
    return loss, n_tokens.clamp(min=1)
