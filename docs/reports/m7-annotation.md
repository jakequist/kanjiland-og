# M7 — Annotation supervision (Stage 1)

Generating silver training data in the full FORMAT_SPEC format (⟨T⟩/⟨W⟩/⟨S⟩/⟨G⟩)
via the ADR-007 hybrid: **MeCab/UniDic** for the deterministic mechanical layer,
the **gpt-5.6-luna** teacher for the judgment layer, gated by the **linter** that
will police all training data.

## Pipeline (all offline, under tools/annotate/)

1. **Deterministic ⟨T⟩** (`deterministic.py`) — UniDic morpheme segmentation, POS
   (mapped to the §6 closed tagset), lemma (orthBase), and **ruby**. The furigana
   aligner recovers kanji-run-only readings (ADR-003) by using surface kana as
   anchors in the reading: **99.97%** success on 3,823 real tokens (取り引き→と/ひ,
   訪れ→おとず, okurigana stripped).
2. **Judgment layer** (`teacher.py`) — luna fills contextual glosses, ⟨W⟩ word
   groupings, the ⟨S⟩ translation, and ⟨G⟩ grammar labels against grammar-1.0, as
   JSON, with the inventory in a cacheable system prompt and token ids as targets.
3. **Assemble + gate** (`assemble.py`) — fuse into a real `format.Document`, gated
   by the **same linter** as training data. Structural slips are auto-repaired (see
   below); semantic correctness is the human audit's job.

## Grammar inventory refinement (this stage)

The frozen inventory dropped to **120 rules**: `COPULA`, `TEINEI_DESUMASU`, and
`DE_ARU` were removed on the principle that **⟨G⟩ marks only non-obvious grammar**.
です/ます/だ/である are the highest-frequency constructions; as ⟨G⟩ rules they'd
dominate the annotations and dilute the signal, while their register is inferable
from the surface token. Kept: keigo (non-trivial) and tense/negation
(meaning-bearing). (ADR-011.)

## Gate results — 10,000 KFTT sentences, luna Batch (~$30)

| | pass-rate | inv6 (tiling/overlap) | assemble_err | inv7 (grammar) | inv4 (ruby) | inv3 (reconstruct) |
|:--|--:|--:|--:|--:|--:|--:|
| first pass | 83.0% | 952 | 315 | 334 | 292 | 140 |
| **+ fixes** | **93.9%** | **0** | 33 | 259 | 308 | 141 |

The fixes were applied by **re-assembling the same teacher responses** (no
re-spend): auto-repair in `assemble.py` — force a single ⟨S⟩ tiling all tokens
(each input is one line), drop overlapping/zero-width ⟨W⟩, skip ⟨G⟩ roles with
invalid span targets, and skip dropped/hallucinated rule_ids. That erased the
entire inv6 category and most assemble errors.

**Result: 9,388 silver annotations** (`data/processed/m7_annot/silver.jsonl`).

### What still drops (~6%, inherent)
- **inv4 (3.1%)** — ruby-count mismatch, mostly tokens UniDic has *no reading* for
  (rare proper nouns); we can't invent a reading, so the sentence drops.
- **inv7 (2.6%)** — grammar roles the teacher got structurally wrong.
- **inv3 (1.4%)** — MeCab surface doesn't reconstruct the source (normalization).

These are acceptable hygiene: the gate keeps only clean data.

## Quality — the linter is semantically blind, so a human audits

Spot-checks show excellent translations, sensible ⟨W⟩ groupings, and mostly-correct
grammar. But the linter cannot catch a *plausible-but-wrong* label (e.g. before the
drop, である was mis-tagged as the polite register). **The ROADMAP's human audit —
Jake reads Japanese — is the real quality check** and is the remaining Stage-1 step
before scaling.

## Cost + status
~$30 (one luna Batch, reused via re-assembly). Deterministic layer + gate are free.
Next: human audit of a sample → decide production scale (with the per-call
sentence-batching optimization to ~halve the annotation cost at scale).
