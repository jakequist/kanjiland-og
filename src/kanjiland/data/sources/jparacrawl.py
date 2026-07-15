"""JParaCrawl v3.0 — web-mined Ja-En bitext (~25M pairs).

Huge and *noisy*: sentences were aligned automatically across crawled web
pages, so a real fraction are only loosely parallel. Two knobs help:

1. **bicleaner score** — JParaCrawl ships a per-pair alignment-quality score
   (higher = more confidently parallel). We can pre-filter on it cheaply,
   before spending GPU on LaBSE.
2. **LaBSE** — the semantic filter (``similarity.py``) is the real cleanup for
   this source (ADR-013/014).

Format: tab-separated, one pair per line. v3.0 is
``url_en \t url_ja \t bicleaner_score \t english \t japanese``. Rather than
trust the column order blindly, ``iter_pairs`` locates the Japanese side by
script and treats the other text field as English — robust to format drift.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..filters import has_japanese
from ._util import download_file, extract_tar

URL = "http://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-ja.tar.gz"
DEFAULT_ROOT = Path("data/raw/jparacrawl")


def download(root: Path = DEFAULT_ROOT, *, force: bool = False) -> Path:
    archive = download_file(URL, root / "en-ja.tar.gz", force=force)
    extract_tar(archive, root, marker=_find_bitext(root))
    return root


def _find_bitext(root: Path) -> Path | None:
    """Locate the extracted bitext text file (name varies by release)."""
    candidates = sorted(root.rglob("*.txt")) + sorted(root.rglob("en-ja*"))
    for c in candidates:
        if c.is_file() and c.suffix != ".gz":
            return c
    return None


def _parse_line(line: str, min_bicleaner: float | None) -> tuple[str, str] | None:
    """Return (ja, en) from one TSV line, or None to skip.

    Finds the Japanese field by script and takes the longest *other* text field
    as English (skips URL-looking fields). Applies the bicleaner floor if a
    numeric score column is present.
    """
    parts = line.split("\t")
    if len(parts) < 2:
        return None

    # Optional bicleaner pre-filter: the score is the numeric field in [0, 1].
    if min_bicleaner is not None:
        score = None
        for p in parts:
            try:
                v = float(p)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                score = v
                break
        if score is not None and score < min_bicleaner:
            return None

    ja_field = None
    for p in parts:
        if has_japanese(p):
            ja_field = p
            break
    if ja_field is None:
        return None

    # English = the longest remaining field that isn't the ja field or a URL.
    en_field = ""
    for p in parts:
        if p is ja_field or p.startswith(("http://", "https://")):
            continue
        if has_japanese(p):
            continue
        if len(p) > len(en_field):
            en_field = p
    if not en_field:
        return None
    return ja_field, en_field


def iter_pairs(
    root: Path = DEFAULT_ROOT,
    min_bicleaner: float | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream (ja, en) pairs from the JParaCrawl bitext file."""
    bitext = _find_bitext(root)
    if bitext is None:
        raise FileNotFoundError(f"no JParaCrawl bitext under {root}; run download() first")
    with bitext.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            pair = _parse_line(line.rstrip("\n"), min_bicleaner)
            if pair is not None:
                yield pair
