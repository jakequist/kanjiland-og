# kanjiland Annotation Format — SPEC v0.2

Status: DRAFT. The linter is the executable form of this spec; where they
disagree, fix one and bump the version.

## 1. Overview

A flat, line-oriented tagged format for Japanese text annotation. Chosen over
JSON/XML/YAML because it is (a) token-cheap for a small model to generate,
(b) streamable, (c) partially recoverable on malformed generation.

The model generates this format. A deterministic converter maps it to JSON at
the API/UI boundary. **This format is the wire format, not the public API.**

## 2. Separators and record tags

Real separators are Unicode Private Use Area codepoints (single characters →
single special tokens in the model vocab). Docs and debug output render them
as bracketed names.

| Codepoint | Debug | Meaning                       |
|-----------|-------|-------------------------------|
| U+E000    | ⟨H⟩   | header record                 |
| U+E001    | ⟨T⟩   | token (morpheme) record       |
| U+E002    | ⟨W⟩   | word-grouping record          |
| U+E003    | ⟨S⟩   | sentence record               |
| U+E004    | ⟨G⟩   | grammar annotation record     |
| U+E005    | ⟨P⟩   | paragraph boundary            |
| U+E00E    | ⟨L⟩   | intra-field list separator    |
| U+E00F    | ⟨F⟩   | field separator               |
| U+E010    | ⟨E⟩   | record terminator             |

Content fields MUST NOT contain any codepoint in U+E000–U+E01F. The data
pipeline strips/escapes PUA codepoints from source text before annotation
(they are vanishingly rare in real text; log any occurrence).

One record per line (`\n` after each ⟨E⟩) for human readability; the parser
treats newlines outside records as insignificant.

## 3. Document structure

Input is pre-split into **paragraphs** by the system (blank-line and layout
heuristics; see M2). Each paragraph is annotated independently:

```
⟨H⟩ ...header fields... ⟨E⟩
⟨P⟩⟨E⟩
  ...records for paragraph 0...
⟨P⟩⟨E⟩
  ...records for paragraph 1...
```

**Token IDs are zero-based and reset at each ⟨P⟩.** All spans are half-open
`start:end` (`2:5` = tokens 2,3,4) and are relative to the current paragraph.

## 4. Record types

### 4.1 Header ⟨H⟩

```
⟨H⟩format_version⟨F⟩ruleset_version⟨E⟩
```
Example: `⟨H⟩0.2⟨F⟩grammar-0.1⟨E⟩`

### 4.2 Token ⟨T⟩ — morpheme-level

```
⟨T⟩id⟨F⟩surface⟨F⟩ruby⟨F⟩dictionary_form⟨F⟩pos⟨F⟩gloss⟨E⟩
```

- `surface`: exact substring of the input. Concatenating all surfaces in a
  paragraph MUST reproduce the paragraph text exactly (whitespace included —
  whitespace, if present, belongs to a token).
- `ruby`: readings for **kanji runs only** (okurigana excluded). Zero or more
  entries, one per maximal kanji run in surface order, joined by ⟨L⟩.
  - 学生 → `がくせい` (one run)
  - 食べる → `た` (ruby for 食 only)
  - 取り引き → `と⟨L⟩ひ` (two runs: 取, 引)
  - は / です / 。 → empty field (no kanji)
- `dictionary_form`: lemma (e.g. 食べ → 食べる). Punctuation: same as surface.
- `pos`: coarse tag from the closed set in §6.
- `gloss`: **contextual** English gloss — what it means *here*. Dictionary
  senses are NOT enumerated here; the UI derives them from dictionary_form +
  JMdict at display time. Punctuation gloss: empty.

### 4.3 Word grouping ⟨W⟩

Groups morpheme tokens into learner-facing words/conjugation units.

```
⟨W⟩span⟨F⟩dictionary_form⟨F⟩gloss⟨E⟩
```

Example: 食べています tokenized as 食べ|て|い|ます (tokens 3–6):
`⟨W⟩3:7⟨F⟩食べる⟨F⟩is eating (polite)⟨E⟩`

Rules: ⟨W⟩ spans MUST NOT overlap each other. Single-token words that add no
information beyond their ⟨T⟩ record MAY be omitted.

### 4.4 Sentence ⟨S⟩

```
⟨S⟩span⟨F⟩translation⟨E⟩
```

⟨S⟩ spans MUST tile the paragraph exactly: non-overlapping, in order, union =
`0:num_tokens`. (Sentence segmentation is thus implicit in ⟨S⟩ records.)

### 4.5 Grammar ⟨G⟩

```
⟨G⟩rule_id⟨F⟩role=target⟨F⟩role=target...⟨E⟩
```

- `rule_id`: identifier from docs/GRAMMAR_RULES.md (closed, versioned set;
  version pinned in the header).
- `target`: a token id (`3`) or span (`2:5`).
- Required/optional roles per rule are defined in the rule inventory.

Example: `⟨G⟩TOPIC_WA⟨F⟩topic=0⟨F⟩marker=1⟨F⟩scope=2:5⟨E⟩`

## 5. Worked example

Input paragraph: `私は学生です。`

```
⟨H⟩0.2⟨F⟩grammar-0.1⟨E⟩
⟨P⟩⟨E⟩
⟨T⟩0⟨F⟩私⟨F⟩わたし⟨F⟩私⟨F⟩PRON⟨F⟩I⟨E⟩
⟨T⟩1⟨F⟩は⟨F⟩⟨F⟩は⟨F⟩PART⟨F⟩(topic marker)⟨E⟩
⟨T⟩2⟨F⟩学生⟨F⟩がくせい⟨F⟩学生⟨F⟩NOUN⟨F⟩student⟨E⟩
⟨T⟩3⟨F⟩です⟨F⟩⟨F⟩です⟨F⟩COP⟨F⟩am (polite)⟨E⟩
⟨T⟩4⟨F⟩。⟨F⟩⟨F⟩。⟨F⟩PUNCT⟨F⟩⟨E⟩
⟨S⟩0:5⟨F⟩I am a student.⟨E⟩
⟨G⟩TOPIC_WA⟨F⟩topic=0⟨F⟩marker=1⟨F⟩scope=2:5⟨E⟩
⟨G⟩COPULA_POLITE⟨F⟩complement=2⟨F⟩copula=3⟨E⟩
```

(Punctuation is a token — the fix from spec v0.1. Note ⟨S⟩ span is 0:5.)

## 6. POS tagset (closed, v0.2)

`NOUN VERB ADJ_I ADJ_NA ADV PRON PART COP AUX CONJ DET NUM INTERJ PREFIX
SUFFIX PUNCT SYM OTHER`

Deliberately coarse. Fine-grained distinctions live in grammar rules, not POS.

## 7. Linter invariants (M0 deliverable)

The linter checks, per paragraph:

1. Header present, versions known.
2. Token IDs sequential from 0, no gaps/duplicates.
3. **Reconstruction:** concatenated surfaces == source paragraph text.
4. Ruby entry count == number of maximal kanji runs in surface.
5. All spans in bounds, well-formed (start < end).
6. ⟨W⟩ spans non-overlapping; ⟨S⟩ spans tile 0:num_tokens.
7. `rule_id` exists in the pinned ruleset version; required roles present;
   role targets valid.
8. `pos` in the closed set.
9. No PUA codepoints inside field content.

Linter modes: `strict` (any violation fails — training data),
`report` (list violations — eval/debug), `repair` (best-effort fixes for a
defined subset, e.g. re-deriving ⟨S⟩ tiling — inference-time salvage).

## 8. Versioning

`format_version` bumps on any change to record syntax or invariants.
`ruleset_version` bumps on grammar-inventory changes. Every stored corpus
file begins with its header; the data pipeline refuses mixed versions.

## 9. Open questions

- Whitespace/newline tokens inside paragraphs: exact policy (M2).
- Should ⟨W⟩ carry a pos too? (Decide when building the annotation model.)
- Confidence scores per record (quality-estimation feature) — reserved
  field or separate record type? Deferred.
