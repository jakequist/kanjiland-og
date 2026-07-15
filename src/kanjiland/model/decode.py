"""Autoregressive decoding: greedy and beam search, both with a KV cache.

Generation is inherently sequential — token t+1 depends on token t — so the
decoder is run one step at a time. The **KV cache** (learning target for M3) is
what makes that affordable: each layer's key/value projections for the tokens
already generated are stored, so step t only computes K/V for the *one* new
token instead of re-encoding the whole prefix. Cross-attention K/V come from the
fixed encoder memory and are computed once. Without the cache, generating an
N-token sentence is O(N²) forward passes' worth of work; with it, O(N).

Two strategies:

- **Greedy**: take the argmax token each step. Fast, deterministic, used for
  quick eval during training.
- **Beam search**: keep the ``beam`` highest-probability *sequences* so far,
  expanding and re-pruning each step. Finds higher-probability translations that
  greedy's local choices miss, at ``beam``× the compute. Length-normalized so it
  doesn't over-prefer short outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .transformer import Transformer


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = 128,
) -> Tensor:
    """Greedy-decode a batch. Returns token ids (B, L) including the leading BOS;
    positions after a sequence's EOS are filled with pad."""
    model.eval()
    device = src.device
    b = src.size(0)
    src_mask = model._src_key_mask(src)
    memory = model.encode(src, src_mask)
    cache = model.init_cache()

    ys = torch.full((b, 1), bos_id, dtype=torch.long, device=device)  # start with BOS
    finished = torch.zeros(b, dtype=torch.bool, device=device)
    for t in range(max_len):
        # Feed only the newest token; the cache supplies the rest of the prefix.
        logits = model.decode(ys[:, -1:], memory, None, src_mask, cache=cache, self_pos=t)
        nxt = logits[:, -1].argmax(dim=-1)  # (B,)
        nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        finished = finished | (nxt == eos_id)
        if bool(finished.all()):
            break
    return ys


def _reorder_cache(cache: list, index: Tensor) -> None:
    """Reindex the per-layer KV cache along the batch dim to follow beams."""
    for layer in cache:
        for sub in ("self", "cross"):
            c = layer[sub]
            if "k" in c:
                c["k"] = c["k"].index_select(0, index)
                c["v"] = c["v"].index_select(0, index)


@torch.no_grad()
def beam_search(
    model: Transformer,
    src: Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam: int = 4,
    max_len: int = 128,
    length_penalty: float = 0.6,
) -> Tensor:
    """Beam-decode a batch. Returns the best sequence per input (B, L) including
    the leading BOS. ``length_penalty`` follows GNMT: divide the sum-log-prob by
    ``((5+len)/6)**alpha`` so longer, equally-fluent hypotheses aren't penalized."""
    model.eval()
    device = src.device
    b, k = src.size(0), beam

    src_mask = model._src_key_mask(src)
    memory = model.encode(src, src_mask)
    # Replicate each sentence across its k beams along the batch dim.
    memory = memory.repeat_interleave(k, dim=0)
    src_mask = src_mask.repeat_interleave(k, dim=0)
    cache = model.init_cache()

    ys = torch.full((b * k, 1), bos_id, dtype=torch.long, device=device)
    # Only beam 0 is "real" at the first step (all beams share BOS); the others
    # start at -inf so the first expansion doesn't produce k identical beams.
    scores = torch.full((b, k), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)  # (b*k,)
    finished = torch.zeros(b * k, dtype=torch.bool, device=device)
    lengths = torch.ones(b * k, dtype=torch.long, device=device)

    for t in range(max_len):
        logits = model.decode(ys[:, -1:], memory, None, src_mask, cache=cache, self_pos=t)
        logp = F.log_softmax(logits[:, -1].float(), dim=-1)  # (b*k, V)
        vocab = logp.size(-1)

        # A finished beam may only extend with pad and gains no score, so it
        # stays put in the ranking instead of being forced to keep emitting.
        pad_row = torch.full_like(logp, float("-inf"))
        pad_row[:, pad_id] = 0.0
        logp = torch.where(finished[:, None], pad_row, logp)

        cand = (scores[:, None] + logp).view(b, k * vocab)  # (b, k*V)
        top_scores, top_idx = cand.topk(k, dim=-1)  # (b, k)
        beam_idx = top_idx // vocab  # which parent beam (0..k-1)
        tok_idx = top_idx % vocab  # which token

        # Flat indices into (b*k) for reordering parents.
        flat = (torch.arange(b, device=device)[:, None] * k + beam_idx).view(-1)
        _reorder_cache(cache, flat)
        ys = torch.cat([ys[flat], tok_idx.view(-1, 1)], dim=1)
        prev_finished = finished[flat]
        scores = top_scores.view(-1)
        new_finished = prev_finished | (tok_idx.view(-1) == eos_id)
        # Grow length only for beams that were still active this step.
        lengths = lengths[flat] + (~prev_finished).long()
        finished = new_finished
        if bool(finished.all()):
            break

    # Length-normalize and pick the best beam per sentence.
    norm = ((5.0 + lengths.float()) / 6.0) ** length_penalty
    final = (scores / norm).view(b, k)
    best = final.argmax(dim=-1)  # (b,)
    ys = ys.view(b, k, -1)
    return ys[torch.arange(b, device=device), best]
