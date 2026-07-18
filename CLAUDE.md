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
- **One branch per milestone.** Each milestone Mn is developed on its own git
  branch named `mn` (`m4`, `m5`, …), branched from the previous milestone's
  branch (or `main`). Start a new branch when beginning a new milestone; never
  develop milestone work directly on `main`. Merge to `main` when the milestone
  is done (human-reviewed).
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

**Restoring data/models on a fresh machine** (corpus + weights are gitignored;
private S3 backup — needs this project's AWS creds). See **docs/DATA.md**:
```bash
aws s3 sync s3://kanjiland/data/raw data/raw
aws s3 sync s3://kanjiland/data/processed data/processed
aws s3 sync s3://kanjiland/checkpoints checkpoints
```

## Current status

Milestone: **M10 — polish for publication — DONE. Project M0–M10 complete.**
(branch `m10`, off `m9`.) README.md rewritten as the research log; docs/
MODEL_CARDS.md added; ROADMAP marked complete. The from-scratch on-device Japanese
reading engine is done end-to-end (rough annotation quality — the 6.8k-data
baseline; improve via more silver data + constrained decoding). All milestone
branches `m5`→`m10` pushed, ready for human review + merge to main.
Previously: **M9 — on-device inference + demo — DONE** (branch `m9`, off `m8`).
docs/reports/m9-ondevice-demo.md: interactive reading-engine demo (tools/demo/,
published Artifact — mincho + real ruby furigana, hover glosses, grammar
highlight); runtime entry scripts/annotate.py (Ja → full annotation, ZERO NLP
deps, rule #1); on-device CPU inference 521 tok/s @ 40% lint-pass (= GPU) — bf16
autocast MANDATORY on cpu+cuda (fp32 diverges to 0% on this fragile small model);
int8 2.4× smaller (209→85MB). Previously: **M8 — annotation model (e2e de-risk) —
DONE** (branch `m8`, off `m7`).
Previously: **M7 — annotation supervision + grammar-1.0 — DONE** (branch `m7`),
**M6 — distillation dry run — DONE** (`m6`), **M5 — ablations — DONE** (`m5`). All
per-milestone branches, ready for human review + merge to main.

M7 (docs/reports/m7-annotation.md, ADR-011): grammar-1.0 = 120 rules (register/
copula rules dropped — ⟨G⟩ marks only non-obvious grammar). Hybrid pipeline under
tools/annotate/ (MeCab/UniDic deterministic ⟨T⟩ + luna teacher for gloss/⟨W⟩/⟨S⟩/⟨G⟩,
linter gate). Stage-1: 10k KFTT sentences → **9,388 silver annotations at 93.9% gate**
(~$30 luna Batch; auto-repair in assemble.py lifted 83%→93.9% free). M8 (docs/reports/
m8-annotation-model.md): from-scratch model (Ja → full wire format), trained FREE on
6.8k silver (local 4090, $0). **e2e proven** — parse 77% / lint-pass 38% / reconstruct
40%; valid structure, rough content (minimum-spend, per Jake). Improve later: expand
silver + constrained decoding + annotation F1 metrics. Next: M9 (on-device) or the
quality-improvement phase. M0 (format), M1 (BPE tokenizer), M2 (22.1M-pair
corpus), M3 (52.3M from-scratch transformer, KFTT-test chrF 47.2 vs 11.9
baseline), M4 (chrF/BLEU/COMET eval harness + seed-variance protocol,
docs/reports/m4-results.md), and M5 (three-axis ablation sweep,
docs/reports/ablations-1.md), and M6 (KD distillation dry run,
docs/reports/m6-distillation.md) are done. Next: **M7** (start a new `m7` branch
off `m6`) — annotation supervision + grammar inventory. Work happens on
per-milestone branches.

M6 outcome (docs/reports/m6-distillation.md, ADR-017): teacher = gpt-5.6-luna
(bake-off winner, Batch mode ~$30 for 185k KFTT sentences). Sequence-level KD,
matched arms (same JA, teacher-En vs human-En), 2 seeds: in-domain kftt-test the
KD student loses on surface (BLEU −5.06) but is **COMET-tied** — a reference-style
artifact; on mixed m2-test KD **wins all three** (COMET +0.0215). KD buys
generalization + semantic parity, not an in-domain surface win. Distillation is
viable → proceed to M7 with luna. Teacher stack (tools/teacher/: Batch client,
chunking, hygiene, matched-arm harness) is proven. Data (teacher_en.jsonl, pairs)
on local disk; S3 backup pending an aws re-auth.

M5 outcome (docs/reports/ablations-1.md, ADR-016; 15 runs = 5 configs × 3 seeds,
100k steps, run on 2×8-GPU vast.ai boxes via scripts/vast_up.sh, ~$34):
config into M6 is **RoPE · 16k · three-way-tied** (the M3 base, now empirically
justified). RoPE > sinusoidal (clean, non-overlapping seeds); tying is free;
vocab 16k is the footprint/quality sweet spot (resolves ADR-012). Open follow-up:
the RoPE length-extrapolation eval (needs a >128-token test set; checkpoints
preserved locally). Corpus skews long/web-domain (JParaCrawl ~86%), so KFTT
formal-domain eval trails the mixed-test number by ~8 chrF — a KFTT-weighted
training mix remains a candidate run. Cloud sweeps: docs/CLOUD.md (one-command
scripts/vast_up.sh + gotchas learned on the first live run).

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
