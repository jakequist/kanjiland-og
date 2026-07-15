"""Streaming corpus IO: JSONL read/write and a scale-free train/valid/test split.

At JParaCrawl scale (~25M pairs) we can't hold the surviving corpus in memory
to shuffle it. ``reservoir_split_to_file`` does the split in a single streaming
pass using reservoir sampling: it keeps only the held-out sample (a few
thousand pairs) in memory and streams everything else straight to the train
file on disk.

Why reservoir sampling gives a *uniform* random holdout in one pass: keep the
first H items; for the i-th item (i >= H), keep it with probability H/(i+1),
and if kept, evict a uniformly-random current holdout item (which then belongs
to train). Every item ends up in the holdout with equal probability H/N, no
matter how many items there turn out to be — so we never need to know N in
advance or make two passes. Seeded RNG makes the split reproducible.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

Pair = tuple[str, str]


def write_jsonl(path: Path, rows: Iterable[Pair]) -> int:
    """Write (ja, en) rows as ``{"ja":..., "en":...}`` JSONL. Returns count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ja, en in rows:
            f.write(json.dumps({"ja": ja, "en": en}, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Pair]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["ja"], obj["en"]


def reservoir_split_to_file(
    rows: Iterable[Pair],
    seed: int,
    valid_size: int,
    test_size: int,
    train_path: Path,
) -> tuple[list[Pair], list[Pair], int]:
    """Stream ``rows`` into ``train_path`` (JSONL), reservoir-sampling a
    ``valid_size + test_size`` uniform holdout. Returns (valid, test, total).

    Memory is O(valid_size + test_size); the train split is written as it
    streams. The holdout is partitioned into valid then test at the end.
    """
    holdout_size = valid_size + test_size
    rng = random.Random(seed)
    reservoir: list[Pair] = []
    train_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with train_path.open("w", encoding="utf-8") as train_f:

        def _write_train(pair: Pair) -> None:
            train_f.write(json.dumps({"ja": pair[0], "en": pair[1]}, ensure_ascii=False))
            train_f.write("\n")

        for pair in rows:
            if total < holdout_size:
                reservoir.append(pair)
            else:
                # keep with prob holdout_size/(total+1); if kept, evict a random
                # reservoir slot to train.
                j = rng.randint(0, total)
                if j < holdout_size:
                    _write_train(reservoir[j])
                    reservoir[j] = pair
                else:
                    _write_train(pair)
            total += 1

    # Shuffle the holdout so the valid/test partition isn't order-biased, then
    # carve it up. (If the corpus was smaller than the holdout, scale down.)
    rng.shuffle(reservoir)
    if len(reservoir) < holdout_size:
        v = len(reservoir) * valid_size // holdout_size if holdout_size else 0
        valid, test = reservoir[:v], reservoir[v:]
    else:
        valid, test = reservoir[:valid_size], reservoir[valid_size:]
    return valid, test, total
