"""JESC — Japanese-English Subtitle Corpus (~2.8M training pairs).

Crowd-aligned movie/TV subtitles: large, colloquial, and *noisy* — short lines,
heavy repetition, loose alignment. Exactly the corpus the dedup and filtering
stages earn their keep on. The tab-separated ``split/train`` file is
``english<TAB>japanese`` (English first).

As with KFTT, only the official ``train`` split is yielded by default; ``dev``
and ``test`` are held back for evaluation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ._util import download_file, extract_tar, read_lines

URL = "https://nlp.stanford.edu/projects/jesc/data/split.tar.gz"
DEFAULT_ROOT = Path("data/raw/jesc")


def download(root: Path = DEFAULT_ROOT, *, force: bool = False) -> Path:
    archive = download_file(URL, root / "split.tar.gz", force=force)
    extract_tar(archive, root, marker=root / "split" / "train")
    return root


def iter_pairs(
    root: Path = DEFAULT_ROOT,
    splits: tuple[str, ...] = ("train",),
) -> Iterator[tuple[str, str]]:
    """Yield (ja, en) from JESC splits. Lines are ``en<TAB>ja``; malformed lines
    (missing tab) are skipped rather than crashing the stream."""
    for split in splits:
        for line in read_lines(root / "split" / split):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            en, ja = parts[0], parts[1]
            if ja and en:
                yield ja, en
