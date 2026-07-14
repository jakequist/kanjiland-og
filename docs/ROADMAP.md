# Roadmap

Each milestone ends with: passing tests, a W&B report (if training was
involved), and a short writeup in `docs/reports/`. Milestones are sized to be
individually completable; don't start Mn+1 with Mn broken.

## M0 — Format library  ✅ DONE
Record dataclasses, parser, serializer, linter (strict/report/repair modes).
Round-trip and fuzz tests. Debug pretty-printer (PUA → ⟨T⟩ rendering).
**Learning targets:** none ML-specific; this is engineering foundation.
**Done when:** `parse(serialize(x)) == x` property tests pass; linter catches
every invariant in SPEC §7 with a test per invariant.

## M1 — Tokenizer from scratch  ✅ DONE
Comparison table: docs/reports/m1-tokenizer.md. Artifacts:
data/processed/tokenizer-{8,16,32}k.json. See ADR-012 for the vocab-strategy
evidence (final call deferred to M5).

Byte-level BPE trainer + encoder/decoder implemented from scratch (no
sentencepiece at runtime). Special-token support for PUA separators. Train
joint Ja+En vocab candidates (8k/16k/32k) on a corpus sample; measure
tokens/sentence per language.
**Learning targets:** BPE algorithm, vocab/sequence-length tradeoffs, why
byte-level, special-token handling.
**Done when:** encode/decode round-trips arbitrary UTF-8; trainer reproduces
expected merges on a toy corpus; vocab-size comparison table produced.

## M2 — Data pipeline  ← CURRENT
Download + normalize KFTT, JESC, Tatoeba, JParaCrawl (streamed). Cleaning:
language ID, length-ratio filters, dedup, similarity scoring (LaBSE) for
JParaCrawl. Paragraph/sentence segmentation policy. Corpus stats notebook.
**Learning targets:** data quality's outsized role; filtering ablation design.
**Done when:** filtered parallel corpus on disk with stats report; filtering
choices logged as ADRs.

## M3 — Translation model v1 (the "start small" goal)
Encoder-decoder transformer from scratch in raw PyTorch: attention, RoPE and
sinusoidal variants, pre-LN, label smoothing, warmup+inverse-sqrt schedule,
mixed precision, torch.compile, checkpointing. Greedy + beam decoding.
Train transformer-base (~60M) Ja→En.
**Learning targets:** the whole transformer, KV caching, decoding.
**Done when:** trained model beats a word-substitution baseline by a mile and
produces recognizably fluent English on KFTT test; W&B report with curves.

## M4 — Evaluation harness
SacreBLEU + chrF + COMET runners; standard test sets (KFTT test, WMT);
seed-variance protocol; results tables auto-generated from W&B.
**Done when:** one command evaluates any checkpoint and updates a results doc.

## M5 — Ablations round 1 + writeup
Vocab size; tied vs untied embeddings; RoPE vs sinusoidal (incl. length
extrapolation eval). ≥2 seeds each.
**Done when:** `docs/reports/ablations-1.md` with tables, error bars,
interpretation — the flagship portfolio artifact.

## M6 — Distillation
Sequence-level KD: teacher translates monolingual Ja corpus → train student;
compare student vs same-size model on original data. (Also the dry run for
annotation distillation in M7.)
**Learning targets:** KD theory, teacher choice, synthetic-data hygiene.

## M7 — Annotation supervision + grammar inventory
Finalize GRAMMAR_RULES.md v1. Generate silver training data in FORMAT_SPEC
format (supervision source per ADR-007). Linter as data gate; measure
gate pass-rate. Human-audit a sample (the human reads Japanese — use it).

## M8 — Annotation model (the full product format)
Single model or multi-task head that emits the full ⟨T⟩/⟨W⟩/⟨S⟩/⟨G⟩ format.
Constrained decoding experiments (grammar-of-the-format enforcement vs free
generation + repair). Annotation-specific eval metrics (segmentation F1,
ruby accuracy, rule F1).

## M9 — On-device inference + demo
int8/int4 quantization, CPU benchmarks (tokens/sec, quality deltas),
minimal web UI with hover glosses, ruby rendering, attention-alignment
visualization, JMdict lookup integration.

## M10 — Polish for publication
README as research log, model cards, reproduce-from-scratch instructions,
W&B reports linked, blog-style writeup.

## Stretch
Quality-estimation head; scaling-law weekend on a rented A100; idiom
detection via literal-gloss vs translation divergence.
