# M8 — Annotation model (de-risk)

Can a from-scratch model emit the full product format (⟨T⟩/⟨W⟩/⟨S⟩/⟨G⟩) end-to-end?
Minimum-spend de-risk (Jake): reuse the FREE M7 silver set, train locally, ask only
"does it produce VALID format at all?" Quality expected to be rough at this scale.

## Setup
- **Data:** the 9.4k M7 silver annotations, length-filtered to src≤128 / tgt≤1024
  tokens → **6,445 train + 339 valid** `(ja → annotation wire)` pairs.
- **Model:** the M5-winner transformer (52.3M, RoPE / 16k / three-way tied),
  decoder length 1024 for the long wire target. PUA separators are native special
  tokens (M1), so the wire tokenizes directly.
- **Cost:** $0 — free silver data, local 4090 (ADR-009), no teacher/cloud spend.
- Trained 8k steps; loss 10.2 → 1.38, valid_loss → 1.79.

## Result — the e2e system exists

Generate an annotation per held-out sentence, push through the SAME parser + linter
that gate training data (`scripts/evaluate_annotate.py`):

| metric | 339 held-out | reading |
|:--|--:|:--|
| parse-rate | **77.0%** | learns the wire STRUCTURE (tags, token layout) |
| lint-pass-rate | **37.8%** | valid + every SPEC §7 invariant |
| reconstruct-ok | 40.4% | ⟨T⟩ surfaces reproduce the input |

**Verdict: de-risked.** On only 6.8k examples of an ~800-token structured task, the
model already emits well-formed format 77% of the time and fully-valid annotations
~38% of the time, with plausible grammar labels drawn from grammar-1.0. Content is
rough — glosses/translations fall into repetition loops (a classic small-data +
long-sequence failure), e.g. "destroyed the fire to the fire". Exactly the expected
trade of the minimum-spend path.

## Where the losses are (the improvement roadmap)
- **Reconstruction is the dominant invariant failure** (lint-pass 38% ≈ reconstruct
  40%): the model must COPY every source morpheme verbatim into ⟨T⟩ surfaces, and on
  this little data it drifts. Two fixes, in order of leverage:
  1. **More silver data** — the primary lever (expand the M7 set; teacher cost is
     incremental, per ADR-011 / the M7 report).
  2. **Constrained decoding** (ROADMAP M8) — enforce the format grammar during
     generation and copy source tokens into ⟨T⟩ slots. This can *guarantee* parse +
     reconstruction structurally, lifting the valid-rate sharply even at low data.
- Content quality (glosses/translation) improves with data + the above.

## Status
The core product loop is proven end-to-end: **raw Japanese → from-scratch annotation
model → segmentation + ruby + glosses + translation + grammar**, gated by the linter.
Rough quality by design. Next phase (post-e2e, "improve later"): expand silver data
+ constrained decoding + the annotation-specific metric suite (segmentation F1, ruby
accuracy, rule F1).
