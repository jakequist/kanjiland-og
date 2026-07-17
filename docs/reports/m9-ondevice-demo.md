# M9 — On-device inference + demo

Turn the working e2e system into something you can see, click, and run on a CPU.

## Interactive demo
Self-contained web reader (`tools/demo/index.html`, published as an Artifact):
raw Japanese rendered with the full annotation format — mincho type with real
`<ruby>` furigana, hover-glossable tokens (contextual gloss + POS + dictionary
form), grammar entries that light up the tokens they span, and furigana/grammar
toggles. Traditional-pigment identity (ai-indigo + shu-vermilion on washi),
light + dark. Populated with 12 gold annotations (the quality target).

## Shipped inference path (rule #1 clean)
`scripts/annotate.py`: raw Japanese → full ⟨T⟩/⟨W⟩/⟨S⟩/⟨G⟩ annotation, produced
**entirely by the from-scratch model** — it imports **zero NLP dependency**. The
M8 model learned segmentation/ruby itself, so there is no MeCab at inference; only
the from-scratch tokenizer + transformer. `--device cpu` runs the whole thing
on-device; `--json` emits the UI-ready structure the demo renders.

## On-device (52.3M model, 24-thread CPU, no GPU)

**The model runs on a CPU at the same quality as the GPU.** CPU inference:
**521 tok/s (~1.3 s/sentence), 40% lint-pass** on held-out — matching the GPU's
42% and the M8 eval's 38%.

**Numeric-fragility finding (fixed):** generation MUST run under bf16 autocast on
both cuda AND cpu. Run in fp32 and this small, data-starved model's long
autoregressive generation **diverges into 0% parseable output** — the same
weights, just outside the bf16 numerics they trained under. Autocast is now
mandatory in the runtime path (`scripts/annotate.py`); it costs ~2× CPU speed vs
raw fp32 but is the difference between working and garbage.

**int8 quantization:** dynamic-quantizing the Linear layers is **2.4× smaller
(209 → 85 MB)** — a clear on-device size win. But PyTorch's dynamic-quantized ops
don't compose with the bf16 autocast this model requires (they raise under
`torch.autocast`), so a *quality-preserving* int8 speedup isn't cleanly
demonstrated on this fragile model. The size reduction stands; int8 speed is
future work, and likely moot once a better-trained model tolerates fp32/int8
without diverging.

## Deferred (optional M9 extensions)
- **JMdict lookup** — dictionary *senses* on the `dictionary_form` field at display
  time (ADR-005). The demo already shows *contextual* glosses (the model's output);
  full dictionary senses are an additive data-layer (JMdict/EDICT2 download + parse).
- **Attention-alignment viz** — needs the model to surface cross-attention weights;
  a nice UI enhancement for showing token↔translation alignment.

## Status
The core product loop is now **demoable and on-device**: raw Japanese → from-scratch
annotation model → segmentation + ruby + glosses + translation + grammar, rendered
interactively and running on a CPU. Quality is the rough M8 baseline (expand silver
data + constrained decoding to improve).
