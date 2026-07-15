"""LaBSE cross-lingual similarity scoring (offline, GPU-batched).

JParaCrawl is web-mined: pairs are aligned by heuristics, so a chunk of them
are only *loosely* translations — partial, paraphrased, or plain wrong. Length
and language-ID filters can't see meaning, so they let these through. LaBSE
(Language-agnostic BERT Sentence Embedding) maps a sentence in any of 109
languages into a shared vector space where a sentence and its translation land
close together. The cosine similarity between the ja and en embeddings is then
a direct "are these actually the same sentence?" score, and we drop pairs below
a threshold.

This is the expensive filter — a ~470M-parameter transformer — so it runs on
the GPU in batches and only over pairs that already survived the cheap filters.

**Offline-only (ADR-014).** LaBSE is loaded via ``sentence-transformers`` (and
thus HuggingFace ``transformers``). That is fine here: ADR-010 / rule #2 forbid
HF ``transformers`` in the *from-scratch model and training loop*, not in an
offline corpus-cleaning tool. Nothing under ``model/``, ``train/`` or the
runtime path imports this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

Pair = tuple[str, str]


@dataclass
class LaBSEConfig:
    enabled: bool = False  # off by default; only JParaCrawl needs it
    model_name: str = "sentence-transformers/LaBSE"
    # Cosine-similarity floor. True translations typically score ~0.6–0.9;
    # misalignments fall below ~0.5. 0.6 trims the noisy tail without being
    # aggressive. (ADR-013.)
    threshold: float = 0.6
    batch_size: int = 256
    device: str | None = None  # None -> auto (cuda if available else cpu)
    # Half-precision encoding on the GPU. LaBSE's cosine scores are robust to
    # fp16's reduced precision (we only threshold at 0.6), and fp16 roughly
    # doubles encode throughput on the 4090 — the single biggest speedup for the
    # dominant filtering phase. Ignored on CPU.
    fp16: bool = True


class LaBSEScorer:
    """Lazily-loaded LaBSE encoder + cosine scorer."""

    def __init__(self, cfg: LaBSEConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self.cfg.device
            if device is None:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.cfg.model_name, device=device)
            if self.cfg.fp16 and device == "cuda":
                self._model = self._model.half()
        return self._model

    def score(self, pairs: list[Pair]) -> list[float]:
        """Cosine similarity for each (ja, en) pair.

        Both sides are L2-normalized on encode, so the cosine similarity is just
        the row-wise dot product of the two embedding matrices.
        """
        if not pairs:
            return []
        model = self._load()
        ja = [p[0] for p in pairs]
        en = [p[1] for p in pairs]
        emb_ja = model.encode(
            ja,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb_en = model.encode(
            en,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Row-wise dot product = cosine similarity (unit vectors).
        return (emb_ja * emb_en).sum(axis=1).tolist()


def filter_by_similarity(
    pairs: list[Pair],
    scorer: LaBSEScorer,
    threshold: float,
    *,
    chunk_size: int = 50_000,
    on_progress=None,
) -> tuple[list[Pair], int]:
    """Return (kept_pairs, dropped_count), keeping pairs scoring >= threshold.

    Scores in chunks so a 25M-pair corpus doesn't materialize all its
    embeddings at once. ``on_progress(done, total)`` is called per chunk.
    """
    kept: list[Pair] = []
    dropped = 0
    total = len(pairs)
    for start in range(0, total, chunk_size):
        chunk = pairs[start : start + chunk_size]
        scores = scorer.score(chunk)
        for pair, s in zip(chunk, scores):
            if s >= threshold:
                kept.append(pair)
            else:
                dropped += 1
        if on_progress is not None:
            on_progress(min(start + chunk_size, total), total)
    return kept, dropped


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
