# CLAUDE.md — Kanjiland

Japanese reading-comprehension engine: raw Japanese text in → segmentation,
furigana, glosses, translations, and grammar annotations out. Built from
scratch. Runs on a single RTX 4090 for training; inference must eventually 
run on-device (CPU / small GPU).

## Non-negotiable engineering rules

1. **No runtime NLP dependencies.** The shipped inference path may not import
   MeCab/fugashi/SudachiPy/spaCy/etc. Classical tools MAY be used offline for
   silver-data generation, but only under `tools/` and clearly labeled.
   (See docs/DECISIONS.md ADR-007 — ACCEPTED: MeCab/UniDic offline for the
   deterministic labels, LLM teacher for glosses/translations/grammar.)
2. **From-scratch model code.** `src/kanjiland/model/` uses raw PyTorch only —
   no HuggingFace `transformers` in the model or training loop. HF `datasets`
   is acceptable for corpus downloading in `src/kanjiland/data/`.
3. **The format spec is law.** docs/FORMAT_SPEC.md defines the annotation
   format. The linter (`src/kanjiland/format/linter.py`) enforces it. All
   generated training data must pass the linter. Update the spec (and bump
   its version) before changing behavior.
4. **Log every training run** to W&B (project `kanjiland`) from run #1: config,
   git commit, seed, loss curves, eval metrics, GPU utilization. No
   unlogged runs, including debug runs.
5. **Tests first for the format layer.** Parser/serializer/linter are
   round-trip tested (`parse(serialize(x)) == x`) and fuzz-tested against
   malformed input.
6. **Determinism.** Seed everything; record seeds in configs. Ablations run
   ≥2 seeds before we claim a difference is real.

## Repo layout

```
docs/            FORMAT_SPEC.md, ROADMAP.md, DECISIONS.md, GRAMMAR_RULES.md
src/kanjiland/
  format/        record types, parser, serializer, linter  (M0)
  tokenizer/     BPE implemented from scratch               (M1)
  data/          corpus download, cleaning, filtering       (M2)
  model/         transformer from scratch (raw PyTorch)     (M3)
  train/         training loop, schedules, checkpointing    (M3)
  eval/          chrF / SacreBLEU / COMET harness           (M4)
tools/           offline-only helpers (data generation, teacher distillation)
configs/         YAML configs; one file per experiment
tests/           pytest; mirrors src layout
scripts/         entry points (train.py, translate.py, annotate.py)
```

## Conventions

- Python ≥3.11, `uv` for env management, `ruff` for lint+format,
  `pytest` for tests, type hints everywhere, dataclasses for records.
- Commits: conventional-commit style, small and topical.
- Every experiment gets a config file in `configs/` — no hyperparameters
  hardcoded in scripts.
- Docs are living: when we make a design decision, append an ADR to
  docs/DECISIONS.md in the same PR/commit.
- Comment generously and teach through the code. This is a learning project;
  the human wants to understand *why* a piece of logic exists, not just what
  it does. Explain the reasoning, the background concept, and any non-obvious
  tradeoff or gotcha — especially for ML/NLP algorithms (e.g. why byte-level
  BPE, why an incremental pair index, what an invariant protects). Prefer a
  short "why" comment over a restatement of the code. Err on the side of more
  context, not less.

## Commands

```bash
uv sync                        # install deps
uv run pytest                  # run tests
uv run ruff check --fix .      # lint
uv run python scripts/train.py --config configs/<name>.yaml
```

## Current status

Milestone: **M3 — translation model v1**. M0 (format library), M1 (from-scratch
byte-level BPE tokenizer), and M2 (22.1M-pair filtered corpus, 4 sources) are
done. See docs/ROADMAP.md for the full plan and docs/DECISIONS.md for open
questions (grammar-rule inventory scope; tokenizer vocab strategy — ADR-012,
M1 evidence in, final call at M5). Corpus skews long/web-domain (JParaCrawl is
~86%); tokenizer was bootstrapped on raw Tatoeba and should be retrained on the
M2 corpus before M3 training settles.

## Key context from design discussions

- Format uses Unicode Private Use Area codepoints as separators (spec §2);
  they become special tokens in the model vocab.
- Punctuation IS tokenized. Token IDs reset per paragraph. Spans half-open.
- Readings are ruby-only (kanji runs only, no okurigana), one ruby entry
  per maximal kanji run.
- Two segmentation levels: morpheme tokens (`⟨T⟩`) + word groupings (`⟨W⟩`).
- Glosses are contextual (this-sentence meaning); dictionary meanings come
  from JMdict lookup at display time via the dictionary_form field.
- Eval: chrF for iteration, COMET for headline results, SacreBLEU for
  comparability. Distillation (sequence-level KD) is the planned path for
  both translation quality and annotation supervision.
