"""Download Tatoeba Japanese-English parallel sentences.

This is our smoke-test corpus (M2 uses the full pipeline; here we just want
something small and clean to poke the tokenizer and model at). Source is
the ManyThings mirror of Tatoeba, which serves a single zip of cleaned
tab-separated pairs — the same file Anki decks are built from.

Output layout (under ``data/raw/tatoeba/``):
    pairs.tsv    English<TAB>Japanese<TAB>attribution
    ja.txt       one Japanese sentence per line
    en.txt       one English sentence per line
    stats.json   { pair_count, ja_char_stats, en_word_stats, source_url, sha256 }

Usage (module or CLI):
    uv run python -m kanjiland.data.tatoeba
    uv run python -m kanjiland.data.tatoeba --dest data/raw/tatoeba --force
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_URL = "https://www.manythings.org/anki/jpn-eng.zip"
DEFAULT_DEST = Path("data/raw/tatoeba")
INNER_FILENAME = "jpn.txt"  # name inside the ManyThings zip


def iter_pairs(root: Path = DEFAULT_DEST) -> "Iterable[tuple[str, str]]":
    """Yield (ja, en) pairs from an already-downloaded Tatoeba dataset.

    Uniform with the other sources under ``sources/`` so the corpus driver can
    treat Tatoeba identically. Reads the ja.txt/en.txt written by ``download``.
    """
    ja_path, en_path = root / "ja.txt", root / "en.txt"
    with ja_path.open(encoding="utf-8") as jf, en_path.open(encoding="utf-8") as ef:
        for ja, en in zip(jf, ef):
            yield ja.rstrip("\n"), en.rstrip("\n")


@dataclass(frozen=True)
class Stats:
    pair_count: int
    ja_char_len_mean: float
    ja_char_len_p50: int
    ja_char_len_p95: int
    en_word_len_mean: float
    en_word_len_p50: int
    en_word_len_p95: int
    source_url: str
    sha256: str

    def to_dict(self) -> dict:
        return {
            "pair_count": self.pair_count,
            "ja_char_len_mean": round(self.ja_char_len_mean, 2),
            "ja_char_len_p50": self.ja_char_len_p50,
            "ja_char_len_p95": self.ja_char_len_p95,
            "en_word_len_mean": round(self.en_word_len_mean, 2),
            "en_word_len_p50": self.en_word_len_p50,
            "en_word_len_p95": self.en_word_len_p95,
            "source_url": self.source_url,
            "sha256": self.sha256,
        }


def download(
    dest: Path = DEFAULT_DEST,
    url: str = DEFAULT_URL,
    force: bool = False,
) -> Stats:
    """Download the Tatoeba Ja-En zip, extract, and write outputs.

    Idempotent: skips the download if ``pairs.tsv`` already exists and
    ``force`` is False. Returns computed stats either way.
    """
    dest.mkdir(parents=True, exist_ok=True)
    pairs_path = dest / "pairs.tsv"
    stats_path = dest / "stats.json"

    if pairs_path.exists() and stats_path.exists() and not force:
        return _load_stats(stats_path)

    print(f"downloading {url} ...", file=sys.stderr)
    import requests  # local: keeps this optional (belongs to data extra)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = requests.get(url, timeout=60, headers=headers)
    resp.raise_for_status()
    raw = resp.content
    sha256 = hashlib.sha256(raw).hexdigest()

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(INNER_FILENAME) as f:
            body = f.read().decode("utf-8")

    pairs = list(_parse_pairs(body.splitlines()))
    _write_outputs(dest, pairs)
    stats = _compute_stats(pairs, url, sha256)
    stats_path.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    print(f"wrote {stats.pair_count} pairs -> {dest}", file=sys.stderr)
    return stats


def _parse_pairs(lines: Iterable[str]) -> Iterable[tuple[str, str, str]]:
    """ManyThings format: ``en<TAB>ja<TAB>attribution``. Ignore malformed
    lines rather than failing — this is external data."""
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        en, ja = parts[0], parts[1]
        attr = parts[2] if len(parts) > 2 else ""
        if not en or not ja:
            continue
        yield en, ja, attr


def _write_outputs(dest: Path, pairs: list[tuple[str, str, str]]) -> None:
    (dest / "pairs.tsv").write_text(
        "".join(f"{en}\t{ja}\t{attr}\n" for en, ja, attr in pairs),
        encoding="utf-8",
    )
    (dest / "en.txt").write_text(
        "".join(f"{en}\n" for en, _, _ in pairs),
        encoding="utf-8",
    )
    (dest / "ja.txt").write_text(
        "".join(f"{ja}\n" for _, ja, _ in pairs),
        encoding="utf-8",
    )


def _compute_stats(
    pairs: list[tuple[str, str, str]],
    url: str,
    sha256: str,
) -> Stats:
    ja_lens = [len(ja) for _, ja, _ in pairs]
    en_word_lens = [len(en.split()) for en, _, _ in pairs]
    return Stats(
        pair_count=len(pairs),
        ja_char_len_mean=statistics.fmean(ja_lens) if ja_lens else 0.0,
        ja_char_len_p50=_percentile(ja_lens, 50),
        ja_char_len_p95=_percentile(ja_lens, 95),
        en_word_len_mean=statistics.fmean(en_word_lens) if en_word_lens else 0.0,
        en_word_len_p50=_percentile(en_word_lens, 50),
        en_word_len_p95=_percentile(en_word_lens, 95),
        source_url=url,
        sha256=sha256,
    )


def _percentile(values: list[int], p: int) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _load_stats(path: Path) -> Stats:
    d = json.loads(path.read_text())
    return Stats(
        pair_count=d["pair_count"],
        ja_char_len_mean=d["ja_char_len_mean"],
        ja_char_len_p50=d["ja_char_len_p50"],
        ja_char_len_p95=d["ja_char_len_p95"],
        en_word_len_mean=d["en_word_len_mean"],
        en_word_len_p50=d["en_word_len_p50"],
        en_word_len_p95=d["en_word_len_p95"],
        source_url=d["source_url"],
        sha256=d["sha256"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument(
        "--force", action="store_true", help="re-download even if the destination already exists"
    )
    args = ap.parse_args()
    stats = download(args.dest, args.url, args.force)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
