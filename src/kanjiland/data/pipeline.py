"""The cleaning pipeline: raw pairs in, clean parallel pairs out.

Streams (ja, en) pairs from a source through normalization and the
deterministic filters, tracking every drop in a ``Funnel``. Filters run
cheapest-first so we pay for the expensive ones as rarely as possible:

    normalize -> length -> ratio -> script -> dedup -> language-ID

Language ID (fastText) is the one costly per-pair step, so it runs last, only
on pairs that survived everything else. The even-costlier LaBSE semantic filter
is *not* here — it wants GPU batches, so the driver applies it to the survivors
(see ``similarity.py``).

Deduplication is global across sources (a pair in both Tatoeba and JParaCrawl
should appear once) via a shared ``seen`` set. To keep that set small at
25M-pair scale we store a 16-byte BLAKE2b digest of the dedup key rather than
the key strings themselves — ~1.6 GB instead of ~10 GB, with collision odds
that are, for practical purposes, zero.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Iterator

from .filters import FilterConfig, dedup_key, length_ok, ratio_ok, script_ok
from .langid import LangIDConfig, LanguageIdentifier, langid_ok
from .normalize import normalize_text
from .stats import Funnel

Pair = tuple[str, str]


def _digest(key: tuple[str, str]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(key[0].encode("utf-8"))
    h.update(b"\x00")  # unambiguous field separator so ("ab","c") != ("a","bc")
    h.update(key[1].encode("utf-8"))
    return h.digest()


def clean_stream(
    pairs: Iterable[Pair],
    cfg: FilterConfig,
    funnel: Funnel,
    seen: set[bytes],
    *,
    langid_cfg: LangIDConfig | None = None,
    identifier: LanguageIdentifier | None = None,
) -> Iterator[Pair]:
    """Yield normalized, filtered, deduplicated (ja, en) pairs from ``pairs``.

    Mutates ``funnel`` (counts) and ``seen`` (global dedup set) as it goes.
    """
    run_langid = bool(langid_cfg and langid_cfg.enabled and identifier is not None)
    for ja_raw, en_raw in pairs:
        funnel.input_pairs += 1
        ja = normalize_text(ja_raw)
        en = normalize_text(en_raw)

        if not length_ok(ja, en, cfg):
            funnel.drop("length")
            continue
        if not ratio_ok(ja, en, cfg):
            funnel.drop("ratio")
            continue
        if not script_ok(ja, en, cfg):
            funnel.drop("script")
            continue

        digest = _digest(dedup_key(ja, en))
        if digest in seen:
            funnel.drop("dedup")
            continue
        seen.add(digest)

        if run_langid and not langid_ok(ja, en, identifier, langid_cfg):
            funnel.drop("langid")
            continue

        funnel.kept += 1
        yield ja, en


def split_pairs(
    pairs: list[Pair],
    seed: int,
    valid_size: int,
    test_size: int,
) -> dict[str, list[Pair]]:
    """Deterministically shuffle and carve off fixed-size valid/test sets.

    Fixed-size (not percentage) held-out sets keep evaluation cost constant as
    the corpus grows. The shuffle is seeded so the split is reproducible; we
    shuffle a copy of indices rather than the (large) list of pairs to keep peak
    memory down. If the corpus is smaller than valid+test, we scale the held-out
    sets down proportionally rather than starving train.
    """
    n = len(pairs)
    want_holdout = valid_size + test_size
    if want_holdout >= n:
        # Degenerate/tiny corpus: give at most ~20% to holdout, split evenly.
        holdout = max(0, n // 5)
        valid_size = holdout // 2
        test_size = holdout - valid_size

    order = list(range(n))
    random.Random(seed).shuffle(order)
    valid_idx = order[:valid_size]
    test_idx = order[valid_size : valid_size + test_size]
    train_idx = order[valid_size + test_size :]
    return {
        "train": [pairs[i] for i in train_idx],
        "valid": [pairs[i] for i in valid_idx],
        "test": [pairs[i] for i in test_idx],
    }
