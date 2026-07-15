"""Batch translation for evaluation: ja sentences -> en strings.

Wraps the model's greedy/beam decoders with the tokenizer boilerplate — encode
+ pad a batch of source sentences, decode, strip BOS and cut at EOS. Kept
separate from the metrics so the harness can translate once and score with all
three metrics.
"""

from __future__ import annotations

import torch

from kanjiland.model import beam_search, greedy_decode
from kanjiland.tokenizer import Tokenizer
from kanjiland.train.device import amp_context


def _ids_to_text(ids, tok: Tokenizer) -> str:
    lst = ids.tolist()
    if lst and lst[0] == tok.bos_id:
        lst = lst[1:]
    if tok.eos_id in lst:
        lst = lst[: lst.index(tok.eos_id)]
    return tok.decode(lst)


@torch.no_grad()
def translate(
    model,
    tok: Tokenizer,
    sources: list[str],
    device: str,
    *,
    beam: int = 1,
    max_src: int = 128,
    max_len: int = 128,
    batch_size: int = 64,
    on_progress=None,
) -> list[str]:
    """Translate ``sources`` (ja) to en strings. Greedy if ``beam <= 1``."""
    out: list[str] = []
    for i in range(0, len(sources), batch_size):
        chunk = sources[i : i + batch_size]
        enc = [tok.encode(s)[: max_src - 1] + [tok.eos_id] for s in chunk]
        width = max(len(e) for e in enc)
        src = torch.full((len(enc), width), tok.pad_id, dtype=torch.long)
        for j, e in enumerate(enc):
            src[j, : len(e)] = torch.tensor(e)
        src = src.to(device)
        with amp_context(device):
            if beam > 1:
                gen = beam_search(model, src, tok.bos_id, tok.eos_id, tok.pad_id, beam, max_len)
            else:
                gen = greedy_decode(model, src, tok.bos_id, tok.eos_id, tok.pad_id, max_len)
        out.extend(_ids_to_text(g, tok) for g in gen)
        if on_progress is not None:
            on_progress(min(i + batch_size, len(sources)), len(sources))
    return out
