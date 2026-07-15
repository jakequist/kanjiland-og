"""KFTT — Kyoto Free Translation Task (Wikipedia articles on Kyoto/Buddhism).

~440k high-quality Ja-En training pairs, plus small official dev/tune/test
splits. Clean, formal, single-domain — a good backbone corpus.

Data hygiene: ROADMAP M3/M4 evaluate translation quality on the *KFTT test
set*. So ``iter_pairs`` yields only ``train`` + ``tune`` by default and never
``dev``/``test`` — training on your own eval set inflates every metric and is
the classic silent way to fool yourself.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ._util import download_file, extract_tar, read_lines

URL = "http://www.phontron.com/kftt/download/kftt-data-1.0.tar.gz"
DEFAULT_ROOT = Path("data/raw/kftt")
_ORIG_SUBDIR = "kftt-data-1.0/data/orig"


def download(root: Path = DEFAULT_ROOT, *, force: bool = False) -> Path:
    archive = download_file(URL, root / "kftt.tar.gz", force=force)
    extract_tar(archive, root, marker=root / _ORIG_SUBDIR)
    return root


def iter_pairs(
    root: Path = DEFAULT_ROOT,
    splits: tuple[str, ...] = ("train", "tune"),
) -> Iterator[tuple[str, str]]:
    """Yield (ja, en) pairs from the requested KFTT splits (train+tune only by
    default; dev/test are reserved for evaluation)."""
    base = root / _ORIG_SUBDIR
    for split in splits:
        ja_lines = read_lines(base / f"kyoto-{split}.ja")
        en_lines = read_lines(base / f"kyoto-{split}.en")
        for ja, en in zip(ja_lines, en_lines):
            yield ja, en
