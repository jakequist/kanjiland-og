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

## ADR-007 — Supervision source for annotations (ACCEPTED — hybrid)
The model must learn segmentation/ruby/gloss/grammar, so training labels
must come from somewhere. "No external NLP dependencies" is interpreted as a
RUNTIME constraint (see ADR-010, rule #1): it governs the shipped inference
path only. Offline silver-label generation is unconstrained by it, so all
four options below preserve the from-scratch model — the choice is purely a
label-quality / determinism / license tradeoff.

Options considered:
  a) LLM-teacher distillation (quality varies; grammar roles especially)
  b) Existing annotated corpora (UD-Japanese GSD/BCCWJ, furigana corpora)
  c) MeCab/UniDic offline in tools/ (deterministic, high-quality seg+ruby)
  d) Hybrid: (c) for seg/ruby/lemma/pos, (a) for glosses/translations/grammar

DECISION: (d) Hybrid.
  - Deterministic layer → MeCab + UniDic, offline, under tools/ only:
    ⟨T⟩ segmentation boundaries, ruby (per maximal kanji run, aligned via
    UniDic kana readings — ADR-003), dictionary_form (lemma), pos (mapped
    from UniDic tags to the closed §6 tagset via a versioned mapping table
    committed under tools/).
  - Judgment layer → LLM teacher, offline: contextual glosses (⟨T⟩/⟨W⟩),
    sentence translations (⟨S⟩), grammar-role labels (⟨G⟩).
  - The linter (strict mode) is the data gate; M7 measures gate pass-rate
    and human-audits a sample (SPEC §7, ROADMAP M7).

Rationale:
  - Determinism (rule #6) favors UniDic for the mechanical labels; LLMs
    hallucinate readings and drift on morpheme boundaries.
  - Ruby-only-per-kanji-run (ADR-003) needs exact kanji↔reading alignment,
    which UniDic provides natively and LLMs do not.
  - Glosses/translations/grammar roles are LLM strengths and beyond MeCab.
  - License: MeCab (BSD) + UniDic (tri-license) is cleanly redistributable
    for M10 publication; BCCWJ is not freely redistributable and some
    UD-Japanese-GSD releases are non-commercial — so (b) is not a base.

Runtime cleanliness (rule #1) is unaffected: every tool here lives under
tools/ and is never importable by src/kanjiland (enforced by the tools/
import-guard test). The LLM-teacher prompt/model choice is validated during
M6 (the KD dry run) but does not reopen this ADR.

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

## ADR-012 — Tokenizer vocab strategy (OPEN — M1 evidence in; final call at M5)
Joint Ja+En vocab vs separate; size 8k/16k/32k. Resolve empirically via the
M1 comparison + M5 ablation.

M1 evidence (docs/reports/m1-tokenizer.md, joint vocab, script-aware
pre-tokenizer, Tatoeba): tokens/sentence falls monotonically with vocab size
for both languages but with clear diminishing returns — the 8k→16k gain is
larger than 16k→32k, and Japanese (denser script) benefits more than English
at every size. A joint vocabulary is used as the working default: it lets the
encoder/decoder share embeddings (ADR-010 tie-embeddings ablation) and keeps
one artifact. All three sizes are kept on disk so the M5 ablation can sweep
vocab size (and joint-vs-separate) against downstream chrF/COMET before this
ADR is closed. Not resolving now — sequence length is only a proxy; the M5
translation-quality numbers decide.
