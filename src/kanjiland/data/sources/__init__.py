"""Corpus source downloaders for the M2 pipeline.

Each source module exposes a uniform pair of callables:

    download(root: Path = ..., force: bool = False) -> Path   # fetch + extract
    iter_pairs(root: Path = ...) -> Iterator[tuple[str, str]]  # yield (ja, en)

so the driver can treat every source identically. ``iter_pairs`` *streams* —
JESC (~2.8M) and JParaCrawl (~25M) never fit comfortably in memory, and the
whole point of the pipeline is to filter them down before we materialize
anything.

Data hygiene: where a corpus ships an official train/dev/test split (KFTT,
JESC), ``iter_pairs`` yields only the training portion by default. The dev/test
portions are held back so M3/M4 can evaluate on them without the model having
been trained on the answers.
"""

from . import jesc, jparacrawl, kftt

__all__ = ["kftt", "jesc", "jparacrawl"]
