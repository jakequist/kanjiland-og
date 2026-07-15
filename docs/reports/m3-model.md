# M3 — Translation model v1 (from-scratch transformer)

Ja→En encoder-decoder transformer, raw PyTorch (ADR-010). This report covers the
architecture and what was validated; **training curves + KFTT chrF land when the
run completes** (see "Status").

## Architecture

`configs/m3_transformer_base.yaml` — a ~52M "transformer-base":

| | |
|---|---|
| d_model / heads / d_ff | 512 / 8 / 2048 |
| encoder / decoder layers | 6 / 6 |
| positional | RoPE (sinusoidal variant also built, for the M5 ablation) |
| norm | pre-LN (+ final LN) |
| embeddings | 3-way tied over the joint 16k vocab |
| params | 52.3M |

Attention has two numerically-equivalent implementations — a legible `manual`
softmax(QKᵀ/√d)·V and `F.scaled_dot_product_attention` (Flash) for speed —
selectable by config. Decoding (greedy + beam) runs incrementally with a
**KV cache** (O(N) instead of O(N²) generation).

## Data & training

- Corpus: the 22.1M-pair M2 corpus, pre-tokenized once to a uint16 memmap
  (`scripts/pretokenize.py`) so training never re-tokenizes.
- Batching: token-budget (12k tok/batch × 2 grad-accum) with megabatch length
  bucketing to minimize padding.
- Optimization: label-smoothed CE (ε=0.1), AdamW (β=0.9/0.98), linear warmup
  (4k steps) → inverse-sqrt decay peaking at 5e-4, grad-clip 1.0, bf16 autocast,
  `torch.compile(dynamic=True)`. Checkpoints every 5k steps; `--resume` supported.

## What was validated (tests + on-GPU)

- **93 unit tests** across model/data/train/decode/eval.
- **manual ≡ sdpa** attention (caught a real mask-convention bug).
- **KV-cache ≡ full-decode** for generation.
- **Overfit-a-batch** converges to the label-smoothing floor (~0 with smoothing
  off) at ~300k tok/s — the whole loop is correctly wired.
- **Full-scale pilot**: 52.3M model on the real corpus, ~118k tok/s (compile
  off), GPU 98% / 18GB of 24GB, loss decreasing through warmup, no OOM.

## Training run

100k steps on the 22.1M-pair corpus, ~172k tok/s (bf16 + compile) on the RTX
4090 — finished in a few hours. Loss curve (label-smoothed CE per token):

| step | 1 | 5k | 10k | 20k | 40k | 60k | 80k | 100k |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| train | 10.35 | 3.53 | 3.01 | 2.98 | 2.81 | 2.77 | 2.60 | 2.62 |
| valid | — | 3.32 | 2.99 | — | — | — | — | **2.62** |

Smooth convergence, train≈valid (no overfitting), plateauing ~2.6 in the second
half. (W&B logged offline; `wandb sync` to upload the curves.)

## Results — beats the baseline by a mile ✅

`scripts/evaluate.py`, beam=4, chrF vs the word-substitution baseline:

| test set | model chrF | baseline chrF | delta |
|---|--:|--:|--:|
| KFTT test (formal; roadmap benchmark) | **47.17** | 11.94 | **+35.23** |
| M2 test (mixed, web-heavy) | **55.29** | 18.57 | +36.72 |

Beam≈greedy (55.29 vs 55.22 on M2 test). Both sets beat the baseline ~4×, and
the output is recognizably fluent:

- 道元は、鎌倉時代初期の禅僧。 → *Dogen was a Zen priest in the early Kamakura period.*
- 曹洞宗の開祖。 → *The founder of the Soto sect.*
- ...スケジュールを効果的に配布し... → *...effectively distributing schedules and making them accessible to all parties involved.*

**Domain gap, as predicted:** KFTT (formal Wikipedia) scores ~8 chrF below the
M2 test because training is ~86% JParaCrawl web text. Failure modes are mostly
rare-entity hallucinations (place names). A KFTT-weighted training mix is the
natural M3.1/M5 experiment. Occasional over-generation on ambiguous inputs too.

## Status: DONE

All three "done when" criteria met — beats the baseline by a mile, fluent
English on KFTT test, loss curve logged. Reproduce the eval:

    uv run python scripts/evaluate.py --config configs/m3_transformer_base.yaml \
        --checkpoint checkpoints/m3-transformer-base/final.pt --split kftt-test --beam 4
