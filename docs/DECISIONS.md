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

## ADR-011 — Grammar rule inventory is closed + versioned (ACCEPTED;
grammar-1.0 Tier-1 defined at M7)
⟨G⟩ rule_ids come from docs/GRAMMAR_RULES.md; version pinned in every
file header. Inventory must be frozen at a version before M7 data gen.

M7 scope decision (GRAMMAR_INVENTORY_PROPOSAL.md, accepted by Jake): target
audience is intermediate-to-advanced literature/newspaper readers, so the
inventory tilts N2-N1 and adds two categories beginner sets omit — classical
文語 and formal newspaper register. Rolled out in two tiers to keep the long
tail honest (rare rules the teacher can't label consistently and the student
can't learn):
  - **grammar-1.0 = Tier 1**: 120 high-frequency rules (N5=19/N4=38/N3=38/N2=20/
    N1=5) across 9 functional categories (case & info-structure particles;
    compound particles; tense/aspect/voice; connectives; nominalization & formal
    nouns; modality/evidentiality; keigo; set patterns; focus particles).
    Approved granularity splits: ようだ (infer/simile), そうだ (hearsay/appear).
    Exclusion principle (M7 refinement, Jake): ⟨G⟩ marks only grammar that is
    NON-OBVIOUS to an intermediate-advanced reader. Pure register / trivial
    predication is therefore NOT annotated — `COPULA` (だ/です), `TEINEI_DESUMASU`
    (です/ます politeness), and `DE_ARU` (である) are dropped: they are the highest-
    frequency constructions and would dilute the signal, while their register is
    inferable from the surface token. Kept: keigo (non-trivial, socially meaningful)
    and tense/negation (meaning-bearing, not mere register).
  - **grammar-1.1 = Tier 2** (~30, planned): classical 文語 (curated ~14),
    newspaper register (~8), advanced A-I tail. Frozen only AFTER the M7 gate +
    human audit show Tier-1 labels are clean, so tail-noise doesn't poison the
    first dataset.
Freeze: grammar-1.0 is the inventory M7 data-gen labels against. Adding rules
within a minor version is append-only; renames/removals bump the major version.

## ADR-012 — Tokenizer vocab strategy (ACCEPTED at M5 — 16k joint tied)
Joint Ja+En vocab vs separate; size 8k/16k/32k. Resolve empirically via the
M1 comparison + M5 ablation.

RESOLUTION (M5, docs/reports/ablations-1.md): **joint vocab, size 16k, three-way
tied.** The M5 vocab sweep (3 seeds, 100k steps, kftt-test chrF) gives 8k 46.24 →
16k 46.93 → 32k 47.31 — monotonic but with diminishing returns (8k→16k +0.69,
16k→32k +0.38) and 32k the noisiest config (σ=0.30; its seed-1 = 46.88 is no
better than 16k). 16k captures ~65% of the 8k→32k gain at half the 32k vocab
cost, with the tightest seed variance, and keeps the on-device footprint smaller
(rule: inference must run on CPU/small GPU). 32k retained as a documented
"quality-max" fallback if a later milestone shows +0.38 chrF outweighs the 2×
embedding/softmax cost; 8k dropped. Joint (not separate) confirmed — it lets the
embeddings tie three-way (Axis 2: tying is free, 46.93 vs 46.99) and keeps one
artifact. This closes the size + joint-vs-separate questions ADR left open.

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

NOTE (M3): the 16k tokenizer was RETRAINED on a 3M-pair random sample of the M2
corpus (data/processed/tokenizer-16k.json, docs/reports/m3-tokenizer.md) so
vocab/merges reflect the real training distribution — this is what M3 trains on.
The 8k/32k tokenizers are still the M1 raw-Tatoeba bootstrap; retrain them on M2
as part of M5-ablation prep. (Aside: BPE training on 6M sentences took ~30 min
in the pure-Python trainer — parallelizing it is the same opportunity as the M2
langid stage, worth doing before the M5 vocab sweep.)

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

## ADR-015 — Ablation methodology (M5) (ACCEPTED)
Ablations (ROADMAP M5) run at FULL SCALE — 100k steps, matching the M3 base.
The original plan was 20k-step reduced runs (relative effects only), but cheap
cloud parallelism (one RTX 4090 per run at ~$0.30/hr, ~$17 for the whole sweep,
docs/CLOUD.md) makes full-quality ablations affordable, so we keep credibility
and comparability to the M3 headline. ≥2 seeds/variant to claim a difference is
real (rule #6, ADR-008). `scripts/ablate.py` sweeps one axis × seeds, trains
each via train.py, evals via the M4 harness into docs/reports/m5-results.json,
and the store auto-aggregates seeds to mean±std.

Axes (each ≥2 seeds), eval chrF/BLEU/COMET on kftt-test:
  - RoPE vs sinusoidal — plus a length-extrapolation eval (train @128, decode
    longer) since that's RoPE's claimed advantage. No tokenizer dependency.
  - Tied (three_way) vs untied (none) embeddings — parameter/quality trade.
  - Vocab size 8k/16k/32k — DEPENDS on retraining the 8k/32k tokenizers on the
    M2 corpus (currently raw-Tatoeba bootstrap, ADR-012); the pure-Python BPE
    trainer is slow (~30 min each) so parallelizing it (or accepting the cost)
    is a prerequisite.

Compute: the full sweep is ~14 training runs at ~4 h/run (100k steps). On the
local single 4090 that is ~2 days; on rented GPUs (one 4090 per run via
--devices/--shard, docs/CLOUD.md) it is ~$17 and ~4 h wall-clock. Parallelism
ceiling is ~14 (one run per GPU) — beyond that nothing is left to parallelize,
so more than ~14 GPUs buys no speedup.

## ADR-016 — M5 ablation outcomes (ACCEPTED)
Results of the ADR-015 sweep (15 runs = 5 configs × 3 seeds, 100k steps,
kftt-test; docs/reports/ablations-1.md). Config going into M6 is **RoPE · 16k ·
three-way-tied** — the M3 base, now empirically justified:
  - **RoPE > sinusoidal** (Axis 1): +0.47 chrF with non-overlapping seed
    distributions (RoPE's worst seed > sinusoidal's best). Keep RoPE. Its
    length-extrapolation claim is NOT yet tested in-domain (needs a >128-token
    eval set; checkpoints preserved to run it locally — tracked in the report).
  - **Tied embeddings are free** (Axis 2): tied 46.93 vs untied 46.99 chrF,
    within seed noise. Keep three-way tied (fewer params, smaller on-device
    footprint, no quality cost).
  - **Vocab 16k** (Axis 3, resolves ADR-012): diminishing returns 8k→16k→32k;
    16k is the footprint/quality sweet spot, 32k the quality-max fallback.
Actuals: ~3.5 h wall-clock, ~$34 on two 8×4090 boxes (ADR-015 estimate held).
Process note: 4 eval-stage failures (not training) from a venv extra-prune + a
results-file write race were recovered by re-running eval from saved checkpoints
per-file, no retraining; fixes in cloud_bootstrap.sh + docs/CLOUD.md gotcha #7.

## ADR-017 — M6 distillation dry-run outcome + teacher choice (ACCEPTED)
Teacher = **gpt-5.6-luna** (bake-off winner: docs/reports/m6-teacher-bakeoff.md —
COMET-best of the affordable tier, correct JA proper-name readings; Batch mode at
½ rate, ~$30 for 185k). Sequence-level KD dry run (docs/reports/m6-distillation.md,
matched arms, 2 seeds): KD student vs human-reference student on the SAME 182k KFTT
sentences.

Result is domain-dependent and metric-splitting:
  - kftt-test (in-domain): baseline > kd on surface (chrF −2.88, BLEU −5.06) but
    **COMET tied (−0.0004)**. The gap is a reference-STYLE artifact — the human
    targets share KFTT's house style with the test refs; luna is semantically
    equal, just phrased differently.
  - m2-test (mixed): **kd wins all three** (chrF +1.80, COMET +0.0215).
Reading: KD buys generalization + semantic parity, not an in-domain surface win
against style-matched human refs. Vindicates COMET-as-headline (ADR-008): BLEU and
COMET point opposite ways here.

DECISION: distillation is viable — proceed to M7 (annotation distillation) with
luna. The in-domain surface confound largely won't apply to structured annotation
labels (no human house-style to match), where teacher consistency matters more.
The teacher stack (Batch client, chunking, hygiene, matched-arm harness) is proven.
