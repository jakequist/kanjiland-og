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

NOTE (post-M2): the M1 tokenizers were trained on raw (un-normalized) Tatoeba
(117k pairs) as a bootstrap. The M2 corpus is 22.1M NFKC-normalized pairs that
skew long and web-domain (JParaCrawl ~86%). Before M3 training settles, retrain
the tokenizer on the M2 corpus so vocab/merges reflect the real training
distribution; the current tokenizers are fine for wiring up M3 but not for the
final numbers.

## ADR-013 — M2 corpus filtering thresholds (ACCEPTED, tunable)
The parallel corpus (KFTT, JESC, Tatoeba, JParaCrawl) is cleaned by a funnel
of filters; quality here dominates downstream translation quality. Thresholds
live in `configs/m2_corpus.yaml` (`FilterConfig`/`LangIDConfig`/`LaBSEConfig`)
and are logged per-stage in docs/reports/m2-corpus.md. Chosen defaults:

  - Length: ja 1–250 chars, en 1–500. Upper bounds drop run-on/boilerplate that
    overflows the 128-token model context and is usually misaligned.
  - Length ratio: en_chars/ja_chars in [0.5, 6.0]. Deliberately asymmetric —
    Japanese is compact, so real Ja→En pairs have MORE English characters.
    Measured on KFTT: median ratio ~3.5, p90 ~5.8; a 4.0 cap dropped 33% of
    good formal pairs, 6.0 drops 8.5% (the genuine misalignment tail, p99 ~12).
  - Script: ja side must contain kana/kanji; en side must be ≥50% Latin letters.
    Catches swapped columns and romaji before paying for language ID.
  - Language ID (fastText lid.176, ADR-014): require ja→`ja`, en→`en` at
    confidence ≥0.5, but SKIP the check when either side <10 chars — fastText
    confuses short kanji-heavy Japanese ("何て？") with Chinese and would drop
    good pairs. On a random Tatoeba sample this over-drop fell from ~22% (short-
    biased head) to ~1%.
  - Dedup: exact match after casefolding + removing spaces, global across all
    sources; stored as 16-byte BLAKE2b digests to fit 25M pairs in ~1.6 GB.
  - LaBSE semantic filter (ADR-014): JParaCrawl only. Cosine ≥0.6 (validated:
    aligned pairs score 0.9+, misaligned <0.35). A bicleaner ≥0.5 pre-filter
    trims the worst before the GPU pass.

These are starting points, not settled science — the ratio band, langid
threshold, and LaBSE cutoff are the prime candidates for the M2 filtering
ablation (ROADMAP M2 learning target). Changing a threshold re-runs the build;
the funnel report quantifies the effect.

M2 result (docs/reports/m2-corpus.md): 29.1M input → 22.1M kept (76.1%).
JParaCrawl dominates (18.97M of 22.1M ≈ 86%). LaBSE dropped 1.57M of its 20.5M
post-cheap-filter survivors (7.7%) — modest, because bicleaner ≥0.5 already
removed most gross noise upstream; the LaBSE cutoff is a natural ablation lever
(temp files kept under data/processed/_m2_tmp via --keep-tmp so LaBSE can be
re-thresholded without redoing the ~28M-pair cheap pass). Two consequences the
funnel makes visible: (1) JParaCrawl's cheap-filter drops are reported as one
lumped `cheap_filters` column, not per-stage — an artifact of the fp16 LaBSE
hot-swap resuming from temp files rather than re-running phase 1 (a fresh full
run restores the breakdown; parallel langid will make that cheap); (2) the
bicleaner pre-filter happens inside the JParaCrawl reader, so its drops are not
counted in the funnel at all — worth surfacing if we tune bicleaner.
Perf note: LaBSE runs fp16 (`LaBSEConfig.fp16`, batch 1024) — ~6.7k pairs/s on
the 4090, ~2x fp32, with no measurable effect on the 0.6-threshold decisions.

## ADR-014 — Offline neural data-filtering tools allowed (ACCEPTED)
Corpus cleaning uses two learned models offline: fastText `lid.176` (language
ID) and LaBSE via `sentence-transformers` (cross-lingual similarity). This is
consistent with the project's constraints:
  - Rule #1 / ADR-007 (no runtime NLP deps) targets the shipped INFERENCE path
    and classical Japanese analyzers; these are neither. They run offline to
    prepare data and are never imported by the model/train/eval runtime.
  - Rule #2 / ADR-010 (no HF `transformers` in the model) scopes to the
    from-scratch model + training loop. LaBSE pulls in `transformers`, but only
    inside `src/kanjiland/data` tooling that never touches the modeling path.
Both live under `src/kanjiland/data`, are lazily imported, and run on the GPU
offline. If the modeling path ever imports `transformers`, that is a bug ADR-010
forbids — worth extending the runtime import-guard test to cover it.
