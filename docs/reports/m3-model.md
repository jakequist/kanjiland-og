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

## Status

Training **launched** (100k steps, ~4-6 h on the RTX 4090, W&B offline). The
ROADMAP "done when" — beat the word-substitution baseline by a mile + fluent
English on KFTT test + W&B report with curves — is evaluated once the run
finishes:

    uv run python scripts/evaluate.py --config configs/m3_transformer_base.yaml \
        --checkpoint checkpoints/m3-transformer-base/final.pt --split test --beam 4

**Note on domain:** the corpus is ~86% JParaCrawl web text, while KFTT test is
formal Wikipedia — expect some domain gap on the headline eval; a KFTT-weighted
mix is a candidate for a later run.

_(Results table + loss curve to be filled in on completion.)_
