# Kanjiland

A **from-scratch Japanese reading-comprehension engine**. Raw Japanese text in →
word segmentation, furigana, dictionary forms, contextual English glosses, sentence
translations, and grammar-pattern annotations out — in a compact tagged format
designed for a small model to generate on-device.

> **[▶ Interactive demo](https://claude.ai/code/artifact/ae3994f6-f7eb-4a99-a63a-bd4bd84e46c3)** — annotated Japanese with hover glosses, furigana, and grammar highlighting.

This repo is also a **research log**: every milestone ends with tests, a report in
[`docs/reports/`](docs/reports/), and an ADR for each design decision
([`docs/DECISIONS.md`](docs/DECISIONS.md)). The interesting part isn't just the
result — it's the from-scratch constraints and what they taught.

## The constraints (what makes it interesting)

- **From-scratch core** — the transformer, tokenizer, and training loop are raw
  PyTorch. No HuggingFace `transformers` in the modeling path (`datasets` only for
  corpus download).
- **Single RTX 4090** for training; cloud only to *parallelize* ablations.
- **On-device inference** — the shipped path runs on a CPU and imports **zero NLP
  dependencies**. No MeCab at runtime: the model learned segmentation itself.
- **The format spec is law** — a linter enforces every invariant; all training data
  must pass it.

## Results at a glance

| stage | what | headline |
|:--|:--|:--|
| [tokenizer](docs/reports/m1-tokenizer.md) | from-scratch byte-level BPE, joint Ja+En | 16k chosen at M5 |
| [corpus](docs/reports/m2-corpus.md) | Tatoeba+KFTT+JESC+JParaCrawl, filtered | 29.1M → **22.1M** pairs |
| [translation](docs/reports/m3-model.md) | 52.3M from-scratch transformer | KFTT chrF **47.2** vs 11.9 baseline |
| [ablations](docs/reports/ablations-1.md) | 15 runs × 3 seeds, cloud-parallel | **RoPE · 16k · tied** (COMET) |
| [distillation](docs/reports/m6-distillation.md) | sequence-level KD dry run | COMET parity in-domain, **wins out-of-domain** |
| [annotation data](docs/reports/m7-annotation.md) | MeCab + LLM-teacher hybrid, linter-gated | **9,388 silver @ 93.9%** gate |
| [annotation model](docs/reports/m8-annotation-model.md) | from-scratch, emits the full format | parse **77%** / valid 38% (e2e proven) |
| [on-device](docs/reports/m9-ondevice-demo.md) | CPU inference + demo | **521 tok/s on CPU**, int8 2.4× smaller |

The **system** is complete end-to-end; the **annotation quality** is an honest
first-pass baseline (trained on 6.8k examples — see "Where it stands").

## Models & datasets (🤗 Hugging Face)

| | | license |
|:--|:--|:--|
| **Model** | [kanjiland-translation](https://huggingface.co/jakequist/kanjiland-translation) — from-scratch Ja→En (chrF 47.2) | MIT |
| **Model** | [kanjiland-annotation](https://huggingface.co/jakequist/kanjiland-annotation) — Ja→full annotation format | MIT |
| **Dataset** | [kanjiland-silver-annotations](https://huggingface.co/datasets/jakequist/kanjiland-silver-annotations) — 9,388 KFTT annotations | CC-BY-SA 3.0 |
| **Dataset** | [kanjiland-kd-translations](https://huggingface.co/datasets/jakequist/kanjiland-kd-translations) — 185k KFTT human+teacher pairs | CC-BY-SA 3.0 |

The 22.1M training corpus is **not published** (JParaCrawl-heavy) — rebuild it from
the pipeline in `src/kanjiland/data/`. See [`NOTICE.md`](NOTICE.md) for all terms.

## The research log

**M0-M1 — foundations.** A tagged wire format ([spec](docs/FORMAT_SPEC.md)) using
Unicode Private-Use codepoints as separators, so they become single special tokens
in the vocab. Round-trip + fuzz-tested parser/serializer/linter. A byte-level BPE
tokenizer from scratch (lazy max-heap merge loop, ~10× faster than the naive scan)
so any UTF-8 — kanji, kana, emoji — round-trips with zero unknowns.

**M2 — data.** A streaming clean-up funnel (NFKC, length/ratio/script filters,
fastText language ID, LaBSE semantic filter on JParaCrawl, BLAKE2b dedup) taking
29.1M raw pairs to 22.1M. Lesson logged as ADRs: JParaCrawl is ~86% of the corpus,
so everything skews web-domain — a recurring confound downstream.

**M3-M4 — translation + eval.** A 52.3M encoder-decoder from scratch (RoPE,
pre-LN, label smoothing, Noam schedule, KV-cached greedy/beam). KFTT-test chrF
47.2 vs an 11.9 word-substitution baseline. A chrF/BLEU/COMET harness with a
seed-variance protocol (≥2 seeds, mean±std).

**M5 — ablations.** Full-scale (100k-step) sweep, 15 runs, parallelized across two
8×4090 cloud boxes in ~3.5h for ~$34. **RoPE beats sinusoidal** (clean,
non-overlapping seeds); **tying embeddings is free**; **16k vocab** is the
footprint/quality sweet spot. A vivid metric lesson: on the vocab axis, chrF and
COMET could diverge — COMET is the trusted headline.

**M6 — distillation.** Sequence-level KD, matched arms (same Japanese, teacher-En
vs human-En targets). Result was *domain-dependent*: in-domain the KD student lost
5 BLEU but was **COMET-tied** (a reference-style artifact); out-of-domain it **won
all three metrics**. KD buys generalization + semantic parity, not an in-domain
surface win.

**M7 — annotation supervision.** The hybrid per [ADR-007](docs/DECISIONS.md):
**MeCab/UniDic** (offline, `tools/` only) for the deterministic layer — segmentation,
POS, and a furigana aligner that recovers kanji-run-only ruby at 99.97% — and a
**gpt-5.6-luna teacher** for the judgment layer (glosses, translation, grammar). A
frozen [120-rule grammar inventory](docs/GRAMMAR_RULES.md) tuned for
intermediate-advanced readers, with a design principle: ⟨G⟩ marks only *non-obvious*
grammar (register/copula rules dropped as noise). 10k sentences → **9,388 silver
annotations at a 93.9% linter-gate pass-rate**; auto-repair in the assembler lifted
that from 83% for free.

**M8 — annotation model.** A from-scratch model that emits the *full* format
(Ja → ~800-token wire), trained on the free M7 silver set. It learns the structure
(**77% parse, 38% fully valid**) with plausible grammar labels — the e2e product
loop, proven. Content is rough (small-data repetition), as expected.

**M9 — on-device + demo.** The [demo](https://claude.ai/code/artifact/ae3994f6-f7eb-4a99-a63a-bd4bd84e46c3),
a CPU runtime path with no NLP deps, and a fragility finding worth its own note:
this small model **diverges into garbage in fp32** — generation must run under the
bf16 autocast it trained with, on CPU and GPU alike.

## Architecture

```
raw Japanese ─▶ from-scratch BPE ─▶ from-scratch transformer ─▶ tagged format ─▶ linter/UI
                (M1)                 (M3 translate / M8 annotate) (M0)            (M9)
```

```
docs/     FORMAT_SPEC · ROADMAP · DECISIONS · GRAMMAR_RULES · reports/
src/kanjiland/  format · tokenizer · data · model · train · eval   (the shipped core)
tools/    offline-only: teacher distillation, MeCab annotation, the demo
configs/  one YAML per experiment      scripts/  train · evaluate · annotate
```

See [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md) for the models and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the ADRs.

## Reproduce from scratch

```bash
uv sync                                   # core + torch (CUDA 12.6 on Linux)
uv run pytest                             # 100 tests: format round-trip, linter, tokenizer, model

# translation model (M3) — from the pretokenized corpus
uv run python scripts/pretokenize.py --config configs/m3_transformer_base.yaml
uv run python scripts/train.py          --config configs/m3_transformer_base.yaml
uv run python scripts/evaluate.py --config configs/m3_transformer_base.yaml \
    --checkpoint checkpoints/<run>/final.pt --test-sets kftt-test,m2-test

# annotation model (M8) — the full product format
uv run python scripts/train.py    --config configs/m8_annotate.yaml
uv run python scripts/annotate.py --config configs/m8_annotate.yaml \
    --checkpoint checkpoints/m8-annotate/seed1/final.pt --text "彼は古い寺を訪れた。" --device cpu
```

Offline data-gen (silver annotations, ablation sweeps) lives under `tools/` and
`scripts/ablate.py`; the cloud runbook is [`docs/CLOUD.md`](docs/CLOUD.md).

## Where it stands (honest)

- **Done:** the whole build (M0-M9) — a complete, from-scratch, on-device Japanese
  reading engine with an interactive demo.
- **Rough:** annotation *quality* is the 6.8k-data baseline (38% valid). The demo
  shows the teacher-supervised *target* quality; the from-scratch model reaches for
  it. The two highest-leverage improvements are **more silver data** and
  **constrained decoding** (enforce the format grammar during generation).
- **Not built:** JMdict dictionary-sense lookup and attention-alignment viz (both
  designed, deferred); grammar-1.1 classical/register rules.

## Development

```bash
uv sync                     # install
uv run pytest               # tests
uv run ruff check --fix .   # lint + format
```

Python ≥3.11, `uv`, `ruff`, `pytest`. Type hints everywhere; dataclasses for
records; conventional commits; one branch per milestone.

## License

Code: **MIT** (`LICENSE`). Data & third-party components have separate terms —
see [`NOTICE.md`](NOTICE.md). The published silver-annotation dataset is
**CC-BY-SA 3.0** (KFTT-derived); the 22.1M training corpus is **not redistributed**
(JParaCrawl-heavy) — rebuild it from the pipeline in `src/kanjiland/data/`.
