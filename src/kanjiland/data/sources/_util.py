"""Shared download/extract helpers for corpus sources."""

from __future__ import annotations

import tarfile
from pathlib import Path


def download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream ``url`` to ``dest`` (skipped if present and not ``force``)."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import requests  # `data` extra; local import keeps the module optional

    print(f"downloading {url} -> {dest} ...")
    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def extract_tar(archive: Path, dest: Path, *, marker: Path | None = None) -> Path:
    """Extract a .tar.gz to ``dest``. If ``marker`` exists, skip (idempotent)."""
    if marker is not None and marker.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive} -> {dest} ...")
    with tarfile.open(archive, "r:*") as tar:
        # filter="data" (Python 3.12+) refuses absolute paths / traversal — safe
        # extraction of untrusted archives.
        tar.extractall(dest, filter="data")
    return dest


def read_lines(path: Path):
    """Yield newline-stripped lines from a UTF-8 text file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
