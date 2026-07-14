# Kanjiland 

A from-scratch Japanese reading-comprehension engine.

**Input:** raw Japanese text.
**Output:** word segmentation, furigana, dictionary forms, contextual English
glosses, sentence translations, and grammar-pattern annotations — in a compact
tagged format designed for small-model generation (see
[docs/FORMAT_SPEC.md](docs/FORMAT_SPEC.md)).

**Constraints that make it interesting:**
- Model, tokenizer, and training loop implemented from scratch (raw PyTorch —
  no HuggingFace transformers in the modeling path).
- Trained end-to-end on a single RTX 4090.
- Inference runs on-device (CPU-friendly, quantized) — no hosted GPU.
- No classical-NLP runtime dependencies.

**Method highlights (planned):** sequence-level knowledge distillation,
data-quality filtering pipeline for JParaCrawl, systematic ablations
(vocab size, embedding tying, RoPE vs sinusoidal) with multi-seed reporting,
cross-attention alignment visualization.

## Status

Pre-v1. Current milestone: **M0 — format library**.
See [docs/ROADMAP.md](docs/ROADMAP.md) and
[docs/DECISIONS.md](docs/DECISIONS.md).

## Development

```bash
uv sync
uv run pytest
```

## Docs

- [Format spec](docs/FORMAT_SPEC.md) — the annotation wire format
- [Roadmap](docs/ROADMAP.md) — milestones M0–M10
- [Decision log](docs/DECISIONS.md) — ADRs, including open questions
- [Grammar rules](docs/GRAMMAR_RULES.md) — the ⟨G⟩ rule inventory
- Reports land in `docs/reports/` as milestones complete
