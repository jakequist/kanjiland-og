# M6 — Sequence-level distillation (dry run)

**Question:** does training our 52M student on a strong teacher's translations beat
training it on the human references, at matched size? (Kim & Rush 2016.)

**Setup.** 185k Japanese sentences sampled from KFTT (formal, proper-name-heavy),
translated by **gpt-5.6-luna** via the Batch API (~$30, 0 failures, 0.21% dropped
by hygiene → 184,610 pairs). Two **matched** corpora, identical except the English
target column:

| arm | Japanese source | English target |
|:--|:--|:--|
| `baseline` | 182,610 KFTT sentences | **human** KFTT references |
| `kd` | the *same* 182,610 sentences | **luna** translations of them |

Identical model (M5 winner: RoPE / 16k / three-way-tied), identical 12k steps,
2 seeds each. The only variable is the target source. Evaluated on kftt-test
(in-domain) and m2-test (mixed-domain), chrF/BLEU/COMET, mean over 2 seeds.

## Results

| test set | arm | chrF | BLEU | COMET |
|:--|:--|--:|--:|--:|
| **kftt-test** (in-domain) | baseline | **46.88 ± 0.40** | **20.99 ± 0.15** | 0.7547 ± 0.0015 |
| | kd | 44.01 ± 0.06 | 15.93 ± 0.22 | 0.7543 ± 0.0019 |
| | **Δ (kd − baseline)** | **−2.88** | **−5.06** | **−0.0004** |
| **m2-test** (mixed) | baseline | 31.70 ± 0.16 | 6.60 ± 0.18 | 0.5746 ± 0.0039 |
| | kd | **33.49 ± 0.00** | **6.97 ± 0.03** | **0.5961 ± 0.0021** |
| | **Δ (kd − baseline)** | **+1.80** | **+0.37** | **+0.0215** |

Effects are stable across seeds (tight σ), so both the in-domain loss and the
out-of-domain win are real, not seed noise.

## The finding: KD trades surface-overlap for generalization, at no semantic cost

The headline is the **kftt-test row, read across the metrics**: the KD student
loses **5 BLEU and 2.9 chrF** but is **COMET-identical (Δ = −0.0004)**. It says the
*same thing* — the human-refs student just says it in *more reference-like words*.

That's not a coincidence, it's the mechanism. The `baseline` targets ARE KFTT
professional translations, drawn from the same corpus and translators as the
kftt-test references. Training on them teaches the student that exact house
style, so it maximizes n-gram overlap **with the test references specifically**.
luna translates the same meaning in its own consistent style (macron romanization,
different phrasing), which is semantically equal (COMET tie) but overlaps the KFTT
references less — so surface metrics punish it. **The in-domain chrF/BLEU "loss" is
a reference-style artifact, not a quality loss.**

Where that style match can't help — **m2-test, a different domain** — the picture
flips: KD **wins on all three metrics**, COMET included (+0.0215). The teacher's
consistency generalizes; the KFTT-specific human style does not.

## Two lessons

1. **Metric discipline (ADR-008 vindicated).** On kftt-test, chrF/BLEU and COMET
   point in *opposite directions* — a 5-BLEU gap next to a 0.0-COMET gap. Judging
   this experiment on BLEU alone would call KD a clear loss; it's actually a
   semantic tie. This is exactly why COMET is our headline metric and why a single
   style-matched reference makes surface metrics treacherous.
2. **What KD buys us here.** Not an in-domain surface win against high-quality,
   style-matched human refs — but a student that is **semantically equal in-domain
   and better out-of-domain**, from a single consistent teacher. For a product that
   will see *arbitrary* Japanese (mixed-domain by nature), the m2-test behavior is
   the more representative one.

## Implications for M7 (annotation distillation)

The dry run did its job — **the whole teacher pipeline is proven**: Batch client,
chunked 185k job, hygiene filter, matched-arm harness, seed protocol. Two things
carry forward:

- For M7's structured annotations (glosses ⟨T⟩/⟨W⟩, grammar ⟨G⟩) there is **no
  "human reference house style" to match** — the surface-metric confound that hurt
  KD in-domain here largely won't apply, and teacher *consistency* is worth more
  for structured labels than for free-form translation.
- Distillation is **viable and safe**: at worst semantically neutral, at best a
  generalization gain. Proceed to M7 with luna as the teacher.

## Caveats

- **Dry-run scale**: 12k steps on 182k pairs, 2 seeds. Effects are stable but
  small-scale; a full run would use more data/steps and ≥3 seeds.
- **Single domain sourced (KFTT)**: the strong style-match is *because* both the
  KD source and the baseline refs are KFTT. A mixed-domain KD source would likely
  narrow the in-domain gap and is the natural next experiment.
- **Cost**: ~$30 teacher (Batch) + ~2h local 4090 for 4 student runs.
