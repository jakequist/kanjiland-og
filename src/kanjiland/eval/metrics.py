"""Translation metrics: chrF, SacreBLEU, and COMET (ADR-008).

Three metrics, three jobs (ROADMAP M4):

- **chrF** — character n-gram F-score. Our *iteration* metric: fast, no external
  model, and robust for Japanese→English where word tokenization is fuzzy. What
  we watch during development.
- **SacreBLEU (BLEU)** — the field's *comparability* metric. n-gram precision
  with a brevity penalty. Not the best-correlated with human judgment, but
  everyone reports it with a standard, versioned tokenization, so it's how you
  compare against published numbers.
- **COMET** — a *learned* metric (a fine-tuned multilingual encoder) that scores
  (source, hypothesis, reference) triples and correlates with human judgment far
  better than surface-overlap metrics. Our *headline* number. It needs a ~2.3GB
  model and a GPU, so it's the expensive one.

chrF/BLEU come from ``sacrebleu`` (which also carries a version signature so
scores are reproducible); COMET from ``unbabel-comet``. Both are standard eval
tooling, imported lazily so importing this module stays cheap. They live under
eval/ and never touch the from-scratch model path (ADR-010).
"""

from __future__ import annotations


def chrf(hypotheses: list[str], references: list[str]) -> float:
    """chrF score (0–100). One reference per hypothesis."""
    import sacrebleu

    return sacrebleu.corpus_chrf(hypotheses, [references]).score


def bleu(hypotheses: list[str], references: list[str]) -> tuple[float, str]:
    """(BLEU score 0–100, sacrebleu signature). The signature records tokenizer
    + version so the number is reproducible and comparable to published work."""
    import sacrebleu

    # The class-based API exposes the reproducibility signature on the metric.
    metric = sacrebleu.BLEU()
    score = metric.corpus_score(hypotheses, [references]).score
    return score, metric.get_signature().format()


class CometScorer:
    """Lazily-loaded COMET model (reference-based, GPU). Load once, score many."""

    def __init__(
        self,
        model_name: str = "Unbabel/wmt22-comet-da",
        batch_size: int = 64,
        gpus: int = 1,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpus = gpus
        self._model = None

    def _load(self):
        if self._model is None:
            from comet import download_model, load_from_checkpoint

            # Downloads to the HF cache on first use (~2.3GB), then reuses it.
            self._model = load_from_checkpoint(download_model(self.model_name))
        return self._model

    def score(self, sources: list[str], hypotheses: list[str], references: list[str]) -> float:
        """System-level COMET score (typically ~0–1, higher better)."""
        model = self._load()
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
        out = model.predict(data, batch_size=self.batch_size, gpus=self.gpus, progress_bar=False)
        return float(out.system_score)
