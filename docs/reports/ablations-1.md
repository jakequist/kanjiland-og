# M5 — Ablations round 1

Full-scale ablations (ADR-015): three architecture/tokenizer axes, **100k steps**
each (matching the M3 base), **3 seeds** per config, evaluated on **kftt-test**
(1160 pairs, beam=4) with chrF / BLEU / COMET. Fifteen training runs total, run
in one wave across two 8×RTX-4090 vast.ai boxes (docs/CLOUD.md).

Raw numbers live in `docs/reports/m5-results.json` (seed-aggregated to mean ±
population std, per the ADR-008 / rule-#6 seed-variance protocol). We only treat
a difference as real when the seed *distributions* separate — not on a single
lucky run.

## Results

| config (rope/sinusoidal · vocab · tied/untied) | seeds | chrF | BLEU | COMET |
|:--|--:|--:|--:|--:|
| rope · 8k · tied            | 3 | 46.24 ± 0.23 | 20.04 ± 0.17 | 0.76 ± 0.00 |
| rope · 16k · tied *(base)*  | 3 | 46.93 ± 0.04 | 20.67 ± 0.19 | 0.77 ± 0.00 |
| rope · 16k · untied         | 3 | 46.99 ± 0.04 | 20.78 ± 0.18 | 0.77 ± 0.00 |
| rope · 32k · tied           | 3 | **47.31 ± 0.30** | **21.20 ± 0.39** | 0.77 ± 0.00 |
| sinusoidal · 16k · tied     | 3 | 46.46 ± 0.23 | 20.44 ± 0.26 | 0.76 ± 0.00 |

The five configs share a single baseline (rope · 16k · tied), so each axis is
read against it:

## Axis 1 — Positional encoding: RoPE vs sinusoidal

**RoPE wins, and the effect is clean.** +0.47 chrF / +0.23 BLEU in the mean, but
more convincingly the *distributions don't overlap*: RoPE's worst seed (46.89)
beats sinusoidal's best seed (46.77). Per-seed chrF:

- rope:       46.99, 46.91, 46.89   (tight, σ≈0.04)
- sinusoidal: 46.77, 46.22, 46.39

So this is a real difference, not seed noise. **Decision: keep RoPE.**

> **Caveat / follow-up:** ADR-015 also called for a *length-extrapolation* eval
> (train @128, decode longer) — RoPE's headline claim is better extrapolation
> beyond the training context, which an in-distribution kftt-test cannot show.
> That eval is **not yet run** (needs a >128-token test set we don't have
> prepared). One checkpoint each of rope·16k and sinusoidal·16k is pulled to
> `checkpoints/` so this can be done locally without re-renting. The in-domain
> win above is sufficient to keep RoPE regardless.

## Axis 2 — Embeddings: tied (three-way) vs untied

**Tying is free.** tied 46.93 ± 0.04 vs untied 46.99 ± 0.04 — a 0.06 chrF gap
with overlapping seeds (tied 46.89–46.99, untied 46.93–47.03). Untying adds a
separate output projection (vocab × d_model extra parameters) and buys nothing
measurable. **Decision: keep three-way tied embeddings** — same quality, fewer
parameters, smaller on-device footprint (a hard constraint per CLAUDE.md).

## Axis 3 — Vocab size: 8k / 16k / 32k

chrF rises monotonically with vocab, **with diminishing returns**:

    8k  46.24  ──(+0.69)──▶  16k 46.93  ──(+0.38)──▶  32k 47.31

This matches the M1 sequence-length finding (ADR-012): the 8k→16k step buys more
than 16k→32k. But two things temper the 32k win:

1. **32k is the noisiest config** (σ=0.30 vs 0.04 at 16k). Its per-seed chrF is
   46.88, 47.50, 47.54 — seed 1 (46.88) is *no better than 16k*; the mean is
   carried by seeds 2–3. The gain is real but soft.
2. **32k is the most expensive** — a 2× embedding/softmax matrix. On the boxes it
   trained at ~5.7 steps/s vs the faster small-vocab runs, and it enlarges the
   on-device model most. For a CPU/small-GPU inference target that cost is paid
   at every token.

**Decision (resolves ADR-012's size question): 16k is the working default.** It
captures ~65% of the 8k→32k chrF gain at half the 32k vocab cost, with the
tightest seed variance of any config. 32k is retained as a "quality-max" option
if a later milestone shows the +0.38 chrF matters more than the footprint. 8k is
dropped — clearly the worst, no compensating benefit at this scale.

## Net configuration going into M6

**RoPE · 16k · three-way-tied** — unchanged from the M3 base, now *empirically
justified* rather than assumed. The ablations show the base was already the right
call on two of three axes, and the third (16k vs 32k) is a deliberate
footprint/quality trade rather than leaving quality on the table.

## Methodology notes (honest record)

- **Compute (actual):** 15 runs × 100k steps in one wave, ~3.5 h training
  wall-clock (gated by the 32k runs at ~5.7 steps/s under 8-way per-box
  contention; smaller-vocab runs finished sooner). ~$34 total on two 8×4090
  boxes — in line with ADR-015's estimate, well inside the $75 budget.
- **Eval recovery:** the per-run eval step (not training) failed on 4 runs
  mid-sweep — the `eval` extra (sacrebleu/comet) was transiently pruned from one
  box's venv, and concurrent writes to the shared results JSON raced. All 15
  *training* runs succeeded and checkpointed; the affected evals were simply
  re-run from the saved checkpoints, each to its own results file (no shared-file
  race), then merged. No run was retrained. Root causes are fixed in
  `scripts/cloud_bootstrap.sh` and documented as gotchas #7 in docs/CLOUD.md.
- **What would strengthen these:** the length-extrapolation eval (above); a
  KFTT-weighted training mix (the corpus skews web-domain ~86% JParaCrawl, so all
  these numbers sit ~8 chrF below the mixed-domain test — a shared confound that
  cancels in the *relative* comparisons here but caps the absolute ceiling).
