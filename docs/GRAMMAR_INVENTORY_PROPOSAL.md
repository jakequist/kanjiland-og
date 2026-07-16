# Grammar Inventory — scope proposal (for review, pre-freeze)

Proposal for the `grammar-1.0` inventory the M7 teacher will label against. Builds
on the `grammar-0.1` schema in GRAMMAR_RULES.md (rule_id → {name, level, roles,
description}); this doc decides **which rules and how many**, not the schema.

**Audience (per Jake):** intermediate-to-advanced readers of Japanese **literature
and newspapers** — *not* beginners. That decision drives everything below: we
weight toward N2-N1, and we include two whole categories a beginner set skips —
**classical/literary grammar (文語)** and **formal written/newspaper register**.

---

## The design tension (what "bigger" costs and buys)

Every rule we add is a label the **teacher must apply consistently**, the **linter
must validate**, and the **student needs examples of to learn**. So inventory size
trades along four axes:

| bigger inventory → | upside | downside |
|:--|:--|:--|
| **coverage** | annotates more of what advanced readers actually hit | — |
| **teacher consistency** | — | rare/subtle rules get labeled inconsistently → noisy data |
| **data density** | — | long tail: rare rules get few examples → student can't learn them |
| **granularity** | finer distinctions (e.g. hearsay vs appearance そうだ) | more boundary calls the teacher can get wrong |

The failure mode of "too big" isn't the common rules — it's the **long tail**: a
180-rule set where 60 rules appear <500 times in the corpus produces labels the
student never learns and the teacher applies unreliably. So the real question isn't
"how many rules" but **"how many rules can we populate densely enough to be real."**

**My recommendation: ~140 rules, rolled out in two tiers** (details below) — the
large end you're leaning toward, but structured so the tail stays honest.

---

## What's available: the category map

Eleven functional categories span the whole difficulty range. Counts are the
*proposed* grammar-1.0 rule count per category; ★ marks categories that matter
most for the literature/newspaper audience.

### A. Case & information-structure particles — ~14 rules · N5-N2
は が を に で へ と から まで より の, plus advanced usage splits (は-contrast vs
は-topic, が-nominative vs が-but, をもって, にて). *Scaffolding — needed even for
advanced text.*

### B. Compound & complex particles ★ — ~22 rules · N3-N1
について・に対して・によって・として・における・に関して・をめぐって・に伴って・に基
づいて・をはじめ・にあたって・に際して・をもとに… **The workhorse of newspaper and
academic prose.** High value, and relatively easy for the teacher to spot (fixed
strings).

### C. Verb morphology: tense / aspect / voice — ~16 rules · N5-N3
〜ている・てある・ておく・てしまう・ていく/くる, passive, causative, causative-passive,
potential, 〜たことがある. Core conjugation machinery.

### D. Clause linking & connectives ★ — ~18 rules · N5-N1
て-linking, 〜し, たり, ながら, つつ, conditionals (ば/たら/なら/と), のに, ので, が/
けれど, **ものの, にもかかわらず, ところで, とはいえ** (the last four are the
intermediate-advanced literary connectives). Central to parsing long literary
sentences.

### E. Nominalization & formal nouns (形式名詞) ★ — ~15 rules · N4-N1
の・こと nominalizers, relative-clause modification, 〜という N, and the **formal
nouns**: わけ, はず, つもり, ため, まま, うち, かぎり, ところ, もの, ばかり. Formal
nouns are *the* advanced-reading bottleneck — high value, but some (わけ/もの) are
genuinely ambiguous and need careful teacher guidance.

### F. Modality & evidentiality ★ — ~16 rules · N4-N1
だろう/でしょう, かもしれない, はず, べき, ようだ/みたいだ, らしい, に違いない, まい,
and the **newspaper evidentials**: 〜とみられる, 〜とされる, 〜という(伝聞), 〜もよう
だ, 〜ものと思われる. Splitting 〜そうだ into **hearsay vs appearance** is the marquee
granularity call (recommend: split — it's high-value and the reading differs).

### G. Honorifics / keigo — ~12 rules · N4-N2
尊敬語 (お〜になる, 〜れる/られる-honorific, irregular いらっしゃる/おっしゃる), 謙譲語
(お〜する, irregular 申す/伺う), 丁寧語 (です/ます), 〜ていただく/くださる. Pervasive in
literature dialogue and formal articles.

### H. Auxiliary constructions & set patterns ★ — ~16 rules · N4-N1
〜なければならない, 〜てはいけない, 〜ほうがいい, 〜ことにする/なる, 〜ようにする/なる,
〜わけではない, 〜わけにはいかない, **〜ざるを得ない, 〜を余儀なくされる, 〜てならない,
〜に越したことはない**. The advanced set patterns are dense in editorials.

### I. Focus & discourse particles — ~12 rules · N5-N2
も, だけ, しか, ばかり, こそ, さえ, まで, でも, なんて, くらい, ほど, など. Subtle
scope/nuance; frequent.

### J. Classical / literary grammar (文語) ★ — ~14 rules · advanced
**The literature differentiator, absent from any JLPT/beginner set.** Curated to
what actually surfaces in modern literary prose and commonly-quoted classical text:
negation ず/ぬ/ざる, classical copula なり/たり, べし/べからず, past・perfect き/けり/
つ/ぬ/たり, conjectural む/らむ/けむ, ごとし, 〜んとす, causative しむ, rhetorical
〜んや. *Not* a full 文語 grammar (that's 40+ rules) — the high-frequency subset.

### K. Written-register & newspaper-specific ★ — ~8 rules · N2-N1
である copula, nominal-predicate / 体言止め, 〜的/〜化/〜性 derivation, heavy-passive
newspaper style, 〜による(受動的原因), 〜をめぐる (headline register). Register markers
that flag "this is formal written Japanese."

**Category totals:** A14 B22 C16 D18 E15 F16 G12 H16 I12 J14 K8 ≈ **163 candidate
rules**; the recommendation trims the sparsest to land ~140.

---

## Recommended shape: ~140 rules, two tiers

Rather than freeze 163 at once and discover half the tail is unlearnable, **tier
the rollout**:

- **Tier 1 — `grammar-1.0` core (~95 rules):** categories A-I minus the rarest,
  the full high-frequency N5-N2 backbone + the common N1 patterns. Dense in the
  corpus → teacher labels them reliably, student actually learns them. We validate
  labeling quality here first (M7 gate pass-rate + human audit — you read Japanese).
- **Tier 2 — `grammar-1.1` advanced/literary (~45 rules):** the classical (J), the
  newspaper register (K), and the advanced tail of F/H. Added *after* Tier-1
  labeling is shown clean, so we don't bake tail-noise into the first dataset.

This gives you the large intermediate-advanced inventory you want (~140 total) while
keeping the first frozen dataset trustworthy. If you'd rather freeze all ~140 at
once, we can — the risk is purely tail data-density, which we'd measure and could
backfill by oversampling rare rules in generation.

---

## Decisions I need from you

1. **Target size / tiering:** ~140 two-tier (recommended) · all ~163 at once ·
   tighter ~100 · larger (push J/K to full 文語 + register, ~200)?
2. **Classical depth (category J):** curated high-frequency subset (~14, recommended)
   · fuller 文語 (~30-40) · skip classical for v1?
3. **Granularity splits:** split the ambiguous-but-high-value ones (そうだ
   hearsay/appearance, ようだ, passive-vs-potential られる)? Recommend yes for those
   three, merge the rest.
4. **Level metadata:** keep JLPT N5-N1 tags for UI filtering, and add a `classical`
   level for J? (Recommend yes — lets the product filter by learner level.)

Once you pick, I'll enumerate the full rule list (ids, roles, descriptions,
≥3 examples each) into GRAMMAR_RULES.md and bump it to `grammar-1.0`.
