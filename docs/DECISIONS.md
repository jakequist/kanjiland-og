# Decision Log (ADRs)

Append-only. Status: ACCEPTED / OPEN / SUPERSEDED-BY-n.

## ADR-001 — Flat tagged wire format (ACCEPTED)
Custom flat format with PUA separators as the model's generation target;
JSON only at the API/UI boundary via deterministic converter.
Why: token-cheap, streamable, salvageable on malformed output; separators
become single special tokens.

## ADR-002 — Punctuation is tokenized (ACCEPTED)
Every character of the input belongs to exactly one token; reconstruction
invariant (SPEC §7.3) depends on this.

## ADR-003 — Ruby-only readings (ACCEPTED)
Reading field carries ruby per maximal kanji run, okurigana excluded.
Why: display-ready; token-level full readings lose kanji↔kana alignment.

## ADR-004 — Two segmentation levels (ACCEPTED)
Morpheme tokens ⟨T⟩ (grammar-rule targets) + word groupings ⟨W⟩
(learner-facing units, conjugation bundles).

## ADR-005 — Contextual glosses (ACCEPTED)
Token/word gloss = meaning in this sentence. Dictionary senses come from
JMdict at display time keyed on dictionary_form. Model doesn't memorize
dictionaries.

## ADR-006 — Paragraph-relative IDs (ACCEPTED)
System splits documents into paragraphs; token IDs reset per paragraph;
paragraphs annotated independently (bounded context, parallelizable).

## ADR-007 — Supervision source for annotations (OPEN — decide before M7)
The model must learn segmentation/ruby/gloss/grammar, so training labels
must come from somewhere. "No external NLP dependencies" is interpreted as a
RUNTIME constraint. Options for offline label generation:
  a) LLM-teacher distillation (quality varies; grammar roles especially)
  b) Existing annotated corpora (UD-Japanese GSD/BCCWJ, furigana corpora)
  c) MeCab/UniDic offline in tools/ (deterministic, high-quality seg+ruby)
  d) Hybrid: (c) for seg/ruby/lemma/pos, (a) for glosses/translations/grammar
Leaning: (d) — but the human wants to keep the project organic; confirm
explicitly before adding any classical-NLP tool even under tools/.

## ADR-008 — Eval stack (ACCEPTED)
chrF for iteration, COMET (headline), SacreBLEU (comparability). Report all
three; claim differences only with ≥2 seeds.

## ADR-009 — Single-GPU local training (ACCEPTED)
All training on the local RTX 4090; cloud only for optional scaling
experiments. Efficiency work (bf16, torch.compile, data loading) is a
first-class learning goal.

## ADR-010 — From-scratch core (ACCEPTED)
Model + training loop + tokenizer implemented from scratch (raw PyTorch).
No HF transformers in the modeling path. HF datasets OK for downloads.

## ADR-011 — Grammar rule inventory is closed + versioned (ACCEPTED,
inventory itself in progress)
⟨G⟩ rule_ids come from docs/GRAMMAR_RULES.md; version pinned in every
file header. Inventory must be frozen at some version before M7 data gen.

## ADR-012 — Tokenizer vocab strategy (OPEN — decide during M1)
Joint Ja+En vocab vs separate; size 8k/16k/32k. Resolve empirically via the
M1 comparison + M5 ablation.
