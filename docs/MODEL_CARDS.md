# Model cards

All models are trained from scratch in raw PyTorch (no HuggingFace `transformers`
in the modeling path — ADR-010). Weights are reproducible from the configs in
`configs/`; seeds are recorded per run.

---

## Tokenizer — joint Ja+En byte-level BPE (M1)

- **Type:** byte-level BPE, implemented from scratch (`src/kanjiland/tokenizer/`).
- **Vocab:** 16k joint Japanese+English (8k/32k variants also on disk). Base
  alphabet is the 256 byte values, so any UTF-8 round-trips with zero unknowns.
- **Special tokens:** the format's Private-Use-Area separators (⟨T⟩⟨W⟩⟨S⟩⟨G⟩ …),
  baked in so the annotation wire tokenizes natively.
- **Training:** BPE merges learned on a 3M-pair random sample of the M2 corpus
  (retrained from the M1 Tatoeba bootstrap so merges match the training
  distribution). Lazy max-heap merge loop (~10× the naive scan).
- **Choice:** 16k selected at M5 — the footprint/quality sweet spot (ADR-012).
  Report: [`docs/reports/m1-tokenizer.md`](reports/m1-tokenizer.md).

---

## Translation model — Ja→En transformer (M3)

- **Architecture:** encoder-decoder Transformer, from scratch. 6+6 layers,
  d_model 512, 8 heads, d_ff 2048, RoPE positional encoding, pre-LN, three-way
  tied embeddings. **52.3M parameters.**
- **Training:** 100k steps on the 22.1M-pair M2 corpus (single RTX 4090). bf16
  autocast, `torch.compile`, label smoothing 0.1, warmup+inverse-sqrt (Noam) LR,
  token-budget batching. Config: `configs/m3_transformer_base.yaml`.
- **Decoding:** KV-cached greedy + beam search (from scratch).
- **Eval** (beam 4, chrF/BLEU/COMET — [ADR-008](DECISIONS.md)):
  - KFTT-test (formal): **47.2 / 20.5 / 0.77**
  - m2-test (mixed): 55.3 / 30.4 / 0.81 · WMT22: 42.2 / 16.3 / 0.76
  - vs word-substitution baseline: KFTT chrF 11.9.
- **Ablation-confirmed config** (M5, [ablations-1.md](reports/ablations-1.md)):
  RoPE > sinusoidal; tied ≈ untied (keep tied); 16k the vocab sweet spot.
- **Intended use:** Ja→En translation; the source side + M6 distillation feed the
  annotation pipeline. **Limitations:** corpus skews web-domain (~86% JParaCrawl),
  so formal-domain (KFTT) scores trail mixed-domain by ~8 chrF.

---

## Annotation model — Ja→full-format transformer (M8)

- **Architecture:** same 52.3M transformer (M5-winner config: RoPE / 16k / tied),
  but the **target is the full annotation wire** (⟨T⟩/⟨W⟩/⟨S⟩/⟨G⟩), not English —
  so the decoder is long (targets average ~800 tokens, cap 1024).
- **Training data:** 6,445 `(Japanese → annotation wire)` pairs, from the M7
  teacher-supervised silver set (length-filtered). Config:
  `configs/m8_annotate.yaml`. Trained free on the local 4090.
- **What it produces:** raw Japanese → segmentation, ruby, per-token glosses, word
  groupings, sentence translation, and grammar labels from the closed
  [120-rule inventory](GRAMMAR_RULES.md) — the whole product format, no MeCab at
  inference (the model learned segmentation).
- **Eval** (format validity, held-out — [m8 report](reports/m8-annotation-model.md)):
  parse-rate **77%**, lint-pass (fully valid) **38%**, reconstruct-ok 40%.
- **On-device:** runs on CPU at **521 tok/s, 40% lint-pass** (= GPU). int8 dynamic
  quant is 2.4× smaller (209→85 MB).
- **⚠ Numeric fragility:** generation MUST run under bf16 autocast (cpu + cuda). In
  fp32 the long autoregressive decode diverges to 0% parseable output — the model
  is trained-precision-sensitive at this data scale.
- **Intended use:** de-risk / e2e proof of the annotation product. **Limitations:**
  trained on only 6.8k examples — valid *structure*, rough *content* (glosses and
  translations loop on repetition). Improve via more silver data + constrained
  decoding, not architecture.
