# Grammar Rule Inventory — grammar-1.0

Closed, versioned set of `rule_id`s for ⟨G⟩ records (FORMAT_SPEC §4.5). The linter
validates every ⟨G⟩ against the version pinned in a file's header (`⟨F⟩grammar-1.0`).

**Scope** (ADR-011, GRAMMAR_INVENTORY_PROPOSAL.md — accepted): intermediate-to-
advanced readers of literature + newspapers. grammar-1.0 is **Tier 1**: **122**
high-frequency rules (N5=21, N4=38, N3=38, N2=20, N1=5) — dense enough in the
corpus that the teacher labels them reliably and the student can learn them.
**Tier 2** (classical 文語, newspaper register, the advanced N1 tail — ~30 rules)
is enumerated at the bottom as *Planned grammar-1.1* and frozen only after Tier-1
labeling quality is validated (M7 gate pass-rate + human audit). Total planned
inventory ≈ 150.

## Schema

```yaml
RULE_ID:
  name: human-readable name
  level: N5 | N4 | N3 | N2 | N1 | classical   # for product UI filtering
  roles:
    role_name: {target: token|span, required: true|false}
  description: one-line learner-facing summary
```

Role vocabulary is deliberately small and reused: `marker` (the particle/morpheme
token that signals the rule), and content roles `topic subject object predicate
verb complement clause condition result agent head scope focus`. Name/description
strings are single-quoted (they contain `/ , : " ( )`); `roles` uses inline flow
maps whose values are simple tokens, which is safe. Positive examples (≥3 each)
live in `tests/fixtures/grammar/` and are linter-tested — NOT inlined here.
Renames/removals bump the major version; additions within a minor version are
append-only.

---

## A. Case & information-structure particles (N5-N2)

```yaml
TOPIC_WA:
  name: 'Topic marking with は'
  level: N5
  roles: {topic: {target: span, required: true}, marker: {target: token, required: true}, scope: {target: span, required: false}}
  description: 'は marks what the sentence is about.'
CONTRAST_WA:
  name: 'Contrastive は'
  level: N3
  roles: {contrasted: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'は setting up an explicit or implied contrast (as-for-X-at-least).'
SUBJECT_GA:
  name: 'Subject marking with が'
  level: N5
  roles: {subject: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'が marks the grammatical subject.'
OBJECT_GA:
  name: 'Object of stative predicate with が'
  level: N4
  roles: {object: {target: span, required: true}, marker: {target: token, required: true}, predicate: {target: span, required: false}}
  description: 'が marking the object of 好き / 上手 / ほしい / できる etc.'
OBJECT_WO:
  name: 'Direct object with を'
  level: N5
  roles: {object: {target: span, required: true}, marker: {target: token, required: true}, verb: {target: span, required: false}}
  description: 'を marks the direct object of a transitive verb.'
TRAVERSAL_WO:
  name: 'を of path / traversal / departure'
  level: N4
  roles: {path: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'を with motion verbs (道を歩く / 家を出る) — path traversed or point left.'
LOCATION_NI:
  name: 'に of location / existence'
  level: N5
  roles: {location: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'に marking where something exists (ある / いる / 住む).'
TARGET_NI:
  name: 'に of goal / recipient / time'
  level: N5
  roles: {target_np: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'に marking a destination, recipient, or point in time.'
AGENT_NI:
  name: 'に of agent / source'
  level: N3
  roles: {agent: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'に marking the agent in passives or the source in もらう / 借りる.'
LOCATION_DE:
  name: 'で of action location'
  level: N5
  roles: {location: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'で marking where an action takes place.'
MEANS_DE:
  name: 'で of means / cause / material'
  level: N4
  roles: {means: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'で marking instrument, method, cause, or material.'
DIRECTION_E:
  name: 'へ of direction'
  level: N5
  roles: {direction: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'へ marking direction of movement.'
COMITATIVE_TO:
  name: 'と of accompaniment / reciprocal'
  level: N5
  roles: {partner: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'と marking with / together.'
QUOTATIVE_TO:
  name: 'Quotative と'
  level: N4
  roles: {quote: {target: span, required: true}, marker: {target: token, required: true}, verb: {target: span, required: false}}
  description: 'と marking a quote / thought with 言う / 思う / 考える etc.'
SOURCE_KARA:
  name: 'から of source / origin'
  level: N5
  roles: {source: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'から marking a spatial / temporal starting point.'
LIMIT_MADE:
  name: 'まで of extent / limit'
  level: N5
  roles: {limit: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'まで marking the end point of a range.'
COMPARISON_YORI:
  name: 'より of comparison'
  level: N4
  roles: {standard: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'より marking the standard in a comparison (than-X).'
GENITIVE_NO:
  name: 'Genitive / modifier の'
  level: N5
  roles: {modifier: {target: span, required: true}, marker: {target: token, required: true}, head: {target: span, required: false}}
  description: 'の linking a noun modifier to its head (possession / attribute / apposition).'
COPULA:
  name: 'Copula だ / です (nominal predication)'
  level: N5
  roles: {complement: {target: span, required: true}, copula: {target: token, required: true}}
  description: 'X だ / です predicates "is X" (register captured separately by 丁寧).'
```

## B. Compound & complex particles (N3-N1) — newspaper/academic workhorse

```yaml
NI_TSUITE:
  name: 'について (about / concerning)'
  level: N3
  roles: {theme: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Marks the topic under discussion.'
NI_TAISHITE:
  name: 'に対して (toward / against / in contrast)'
  level: N3
  roles: {target_np: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Directed attitude or action toward, or contrast with, X.'
NI_YOTTE:
  name: 'によって (by means of / depending on / by agent)'
  level: N3
  roles: {cause: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Agent, means, or basis of variation.'
TOSHITE:
  name: 'として (as / in the capacity of)'
  level: N3
  roles: {role_np: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'In the role or capacity of X.'
NI_OKERU:
  name: 'における (in / at, formal locative)'
  level: N2
  roles: {domain: {target: span, required: true}, marker: {target: span, required: true}, head: {target: span, required: false}}
  description: 'Formal written in / at, modifying a noun.'
NI_KANSHITE:
  name: 'に関して (regarding)'
  level: N2
  roles: {theme: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Formal concerning-X.'
NI_TOMONATTE:
  name: 'に伴って (accompanying / as X changes)'
  level: N2
  roles: {trigger: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Concurrent change — as X, correspondingly Y.'
NI_MOTOZUITE:
  name: 'に基づいて (based on)'
  level: N2
  roles: {basis: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Grounded on X.'
WO_HAJIME:
  name: 'をはじめ (starting with / including)'
  level: N2
  roles: {representative: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'X as the foremost example of a set.'
NI_ATATTE:
  name: 'にあたって (on the occasion of)'
  level: N2
  roles: {occasion: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'At the significant time of doing X.'
WO_MOTONI:
  name: 'をもとに (on the basis of)'
  level: N2
  roles: {basis: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Using X as material or foundation.'
NI_TOTTE:
  name: 'にとって (for / to, from the standpoint of)'
  level: N3
  roles: {perspective: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'From X point of view.'
NI_OITE:
  name: 'において (in / at, formal)'
  level: N2
  roles: {domain: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Formal locative or temporal in.'
WO_TSUJITE:
  name: 'を通じて / を通して (through / throughout)'
  level: N2
  roles: {medium: {target: span, required: true}, marker: {target: span, required: true}}
  description: 'Via X, or throughout a span.'
NI_YORU:
  name: 'による (due to / by, attributive)'
  level: N3
  roles: {cause: {target: span, required: true}, marker: {target: span, required: true}, head: {target: span, required: false}}
  description: 'Attributive form modifying a noun (caused-by-X).'
```

## C. Verb morphology: tense / aspect / voice (N5-N3)

```yaml
TE_IRU_PROG:
  name: '〜ている progressive'
  level: N5
  roles: {verb: {target: span, required: true}}
  description: 'Ongoing action.'
TE_IRU_RESULT:
  name: '〜ている resultative / state'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Continuing state resulting from a change (結婚している).'
TE_ARU:
  name: '〜てある resultative-purposive'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'State left in place intentionally.'
TE_OKU:
  name: '〜ておく preparatory'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Do X in advance, or leave as is.'
TE_SHIMAU:
  name: '〜てしまう completion / regret'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Completed action, often with regret.'
TE_IKU:
  name: '〜ていく (away / henceforth)'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Change moving away or continuing into the future.'
TE_KURU:
  name: '〜てくる (toward / up to now)'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Change approaching or up to the present.'
PASSIVE:
  name: 'Passive 〜れる / られる'
  level: N4
  roles: {verb: {target: span, required: true}, agent: {target: span, required: false}}
  description: 'Passive voice; agent marked に if present.'
CAUSATIVE:
  name: 'Causative 〜せる / させる'
  level: N4
  roles: {verb: {target: span, required: true}, causee: {target: span, required: false}}
  description: 'Make or let someone do X.'
CAUS_PASSIVE:
  name: 'Causative-passive 〜させられる'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Be made to do X (unwillingly).'
POTENTIAL:
  name: 'Potential 〜られる / える'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Can, or be able to do X.'
TA_KOTO_GA_ARU:
  name: '〜たことがある experience'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Have had the experience of X.'
PAST_TA:
  name: '〜た past / perfect'
  level: N5
  roles: {predicate: {target: span, required: true}}
  description: 'Plain past form.'
NEGATIVE_NAI:
  name: '〜ない negation'
  level: N5
  roles: {predicate: {target: span, required: true}}
  description: 'Plain negative form.'
```

## D. Clause linking & connectives (N5-N1)

```yaml
TE_LINK:
  name: 'て-form clause linking'
  level: N5
  roles: {clause: {target: span, required: true}}
  description: 'Sequential, means, or manner clause chaining.'
SHI_LIST:
  name: '〜し listing reasons'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'And besides — accumulating reasons.'
TARI:
  name: '〜たり〜たり'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'Representative or alternating actions.'
NAGARA:
  name: '〜ながら simultaneous'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'While doing X (same subject).'
TSUTSU:
  name: '〜つつ simultaneous / concessive (literary)'
  level: N2
  roles: {clause: {target: span, required: true}}
  description: 'While or although — written register.'
COND_BA:
  name: '〜ば conditional'
  level: N4
  roles: {condition: {target: span, required: true}, result: {target: span, required: false}}
  description: 'Hypothetical or general if.'
COND_TARA:
  name: '〜たら conditional'
  level: N4
  roles: {condition: {target: span, required: true}, result: {target: span, required: false}}
  description: 'If or when X, then Y.'
COND_NARA:
  name: '〜なら conditional / contextual'
  level: N4
  roles: {condition: {target: span, required: true}, result: {target: span, required: false}}
  description: 'If it is the case that X.'
COND_TO:
  name: '〜と natural-consequence conditional'
  level: N4
  roles: {condition: {target: span, required: true}, result: {target: span, required: false}}
  description: 'Whenever X, inevitably Y.'
NONI:
  name: '〜のに adversative'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'Even though, or despite.'
NODE:
  name: '〜ので reason'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'Because (objective / soft).'
KARA_REASON:
  name: '〜から reason'
  level: N5
  roles: {clause: {target: span, required: true}}
  description: 'Because or since (subjective).'
GA_KEREDO:
  name: '〜が / けれど adversative link'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'But or however, clause link.'
MONONO:
  name: '〜ものの concessive'
  level: N2
  roles: {clause: {target: span, required: true}}
  description: 'Although X is true — written.'
TOWA_IE:
  name: '〜とはいえ concessive'
  level: N1
  roles: {clause: {target: span, required: true}}
  description: 'That said, or even so.'
```

## E. Nominalization & formal nouns 形式名詞 (N4-N1)

```yaml
NOMINALIZER_NO:
  name: 'の nominalizer'
  level: N4
  roles: {clause: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Turns a clause into a noun phrase.'
NOMINALIZER_KOTO:
  name: 'こと nominalizer'
  level: N4
  roles: {clause: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Nominalizes a clause (abstract fact / action).'
REL_CLAUSE:
  name: 'Relative-clause modification'
  level: N4
  roles: {clause: {target: span, required: true}, head: {target: span, required: true}}
  description: 'A clause modifying a following noun.'
TO_IU_N:
  name: '〜という N apposition'
  level: N3
  roles: {content: {target: span, required: true}, head: {target: span, required: false}}
  description: 'The N that is called or defined as X.'
FN_WAKE:
  name: '形式名詞 わけ'
  level: N2
  roles: {clause: {target: span, required: true}}
  description: 'Reason or conclusion that naturally follows (no wonder / it means).'
FN_HAZU:
  name: '形式名詞 はず expectation'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'Should be, or expected to be (logical expectation).'
FN_TSUMORI:
  name: '形式名詞 つもり intention'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'Intention or conviction of doing / being.'
FN_TAME_PURPOSE:
  name: '〜ため(に) purpose'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'In order to X.'
FN_TAME_CAUSE:
  name: '〜ため cause'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'Because of X (formal).'
FN_MAMA:
  name: '形式名詞 まま unchanged state'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'As is, or while remaining in the state of X.'
FN_UCHI:
  name: '〜うちに (within / before)'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'While or before the state X still holds.'
FN_KAGIRI:
  name: '〜かぎり (extent / as long as)'
  level: N2
  roles: {clause: {target: span, required: true}}
  description: 'As far as, or as long as X.'
FN_TOKORO:
  name: '形式名詞 ところ juncture'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'Point in time — just about to, or just did.'
```

## F. Modality & evidentiality (N4-N1) — incl. approved granularity splits

```yaml
DAROU:
  name: '〜だろう / でしょう conjecture'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'Probably, or conjecture.'
KAMOSHIRENAI:
  name: '〜かもしれない possibility'
  level: N4
  roles: {clause: {target: span, required: true}}
  description: 'Might, or maybe.'
BEKI:
  name: '〜べき obligation / propriety'
  level: N2
  roles: {verb: {target: span, required: true}}
  description: 'Should or ought to (moral / logical).'
YOUDA_INFER:
  name: '〜ようだ inference from evidence'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'It seems, from evidence. [split from YOUDA_SIMILE]'
YOUDA_SIMILE:
  name: '〜ようだ simile / likeness'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'Like, or as if X (comparison). [split from YOUDA_INFER]'
RASHII:
  name: '〜らしい hearsay / typical'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'Apparently, or seems by reputation.'
NI_CHIGAINAI:
  name: '〜に違いない certainty'
  level: N2
  roles: {clause: {target: span, required: true}}
  description: 'Must be, or no doubt.'
SOUDA_HEARSAY:
  name: '〜そうだ hearsay'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'I hear that X (reported). [split from SOUDA_APPEAR]'
SOUDA_APPEAR:
  name: '〜そうだ appearance / imminence'
  level: N3
  roles: {predicate: {target: span, required: true}}
  description: 'Looks like X, or about to X (from appearance). [split from SOUDA_HEARSAY]'
TO_MIRARERU:
  name: '〜とみられる (is seen as)'
  level: N1
  roles: {clause: {target: span, required: true}}
  description: 'Newspaper evidential — is regarded or estimated as.'
TO_SARERU:
  name: '〜とされる (is held to be)'
  level: N1
  roles: {clause: {target: span, required: true}}
  description: 'Newspaper / academic — is considered or said to be.'
TO_IU_HEARSAY:
  name: '〜という hearsay / definition'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'Reported content, or the fact that.'
MAI:
  name: '〜まい negative conjecture / volition'
  level: N1
  roles: {verb: {target: span, required: true}}
  description: 'Probably not, or will not (literary).'
```

## G. Honorifics / keigo (N4-N2)

```yaml
SONKEI_NINARU:
  name: '尊敬 お〜になる'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Respectful form (subject is honored).'
SONKEI_RERU:
  name: '尊敬 〜れる / られる'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Honorific use of the passive form.'
SONKEI_IRREG:
  name: '尊敬 irregular (いらっしゃる / おっしゃる / なさる / 召し上がる)'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Suppletive respectful verbs.'
KENJOU_OSURU:
  name: '謙譲 お〜する'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Humble form (speaker lowers self).'
KENJOU_IRREG:
  name: '謙譲 irregular (申す / 伺う / いたす / 拝見する)'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Suppletive humble verbs.'
TEINEI_DESUMASU:
  name: '丁寧 です / ます'
  level: N5
  roles: {predicate: {target: span, required: true}}
  description: 'Polite (addressee-oriented) register.'
TE_ITADAKU:
  name: '〜ていただく humble receiving'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Humbly receive the favor of X being done.'
TE_KUDASARU:
  name: '〜てくださる respectful giving'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Someone honored does X for me.'
TE_ORU:
  name: '〜ておる humble / written continuous'
  level: N2
  roles: {verb: {target: span, required: true}}
  description: 'Humble or formal equivalent of ている.'
```

## H. Auxiliary constructions & set patterns (N4-N1)

```yaml
NAKEREBA_NARANAI:
  name: '〜なければならない obligation'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Must, or have to do X.'
TEWA_IKENAI:
  name: '〜てはいけない prohibition'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Must not do X.'
HOU_GA_II:
  name: '〜ほうがいい advice'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Had better do X.'
KOTO_NI_SURU:
  name: '〜ことにする decision'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Decide to do X.'
KOTO_NI_NARU:
  name: '〜ことになる outcome / arrangement'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'It is arranged, or turns out, that X.'
YOU_NI_SURU:
  name: '〜ようにする effort'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Make an effort to habitually X.'
YOU_NI_NARU:
  name: '〜ようになる change of state'
  level: N3
  roles: {verb: {target: span, required: true}}
  description: 'Reach the point where X.'
WAKE_DEWA_NAI:
  name: '〜わけではない partial negation'
  level: N2
  roles: {clause: {target: span, required: true}}
  description: 'It is not necessarily the case that X.'
WAKE_NIWA_IKANAI:
  name: '〜わけにはいかない social inability'
  level: N2
  roles: {verb: {target: span, required: true}}
  description: 'Cannot bring myself, or socially, to do X.'
ZARU_WO_ENAI:
  name: '〜ざるを得ない unavoidable'
  level: N1
  roles: {verb: {target: span, required: true}}
  description: 'Have no choice but to X.'
KOTO_GA_DEKIRU:
  name: '〜ことができる potential (analytic)'
  level: N5
  roles: {verb: {target: span, required: true}}
  description: 'Be able to do X.'
HAZU_GA_NAI:
  name: '〜はずがない impossibility'
  level: N3
  roles: {clause: {target: span, required: true}}
  description: 'There is no way that X.'
NAKUTEWA_IKENAI:
  name: '〜なくてはいけない obligation (variant)'
  level: N4
  roles: {verb: {target: span, required: true}}
  description: 'Must do X (colloquial variant).'
```

## I. Focus & discourse particles (N5-N2)

```yaml
MO_ALSO:
  name: 'も (also / even)'
  level: N5
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Addition (too) or scalar (even).'
DAKE:
  name: 'だけ (only)'
  level: N5
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Limit — only X.'
SHIKA_NAI:
  name: 'しか〜ない (nothing but)'
  level: N4
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Exclusive, with a negative predicate.'
BAKARI:
  name: 'ばかり (just / only / nothing but)'
  level: N3
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Approximation, exclusivity, or just-did.'
KOSO:
  name: 'こそ emphatic'
  level: N2
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Emphatic focus — X precisely.'
SAE:
  name: 'さえ (even, extreme)'
  level: N2
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Scalar extreme; with ば means if-only.'
MADE_EVEN:
  name: 'まで (even / to the point of)'
  level: N3
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Surprising extent — even X.'
DEMO_EG:
  name: 'でも (or something / even)'
  level: N4
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Softening example, or scalar even.'
KURAI:
  name: 'くらい / ぐらい (about / to the extent)'
  level: N3
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Approximation or degree.'
HODO:
  name: 'ほど (extent / the more)'
  level: N3
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Degree or comparison — to the extent of.'
NADO:
  name: 'など (et cetera / such as)'
  level: N4
  roles: {focus: {target: span, required: true}, marker: {target: token, required: true}}
  description: 'Exemplification or downplaying.'
```

---

## Planned grammar-1.1 (Tier 2 — frozen after Tier-1 labeling validated)

Enumerated for planning; NOT valid `rule_id`s until grammar-1.1 is cut. Added once
the M7 gate + human audit confirm Tier-1 labels are clean, so the sparse tail
doesn't poison the first dataset.

**J. Classical / literary 文語 (level: classical)** — curated high-frequency subset:
`NEG_ZU` (ず/ぬ/ざる), `COPULA_NARI`, `COPULA_TARI`, `BESHI` (べし/べからず),
`PERF_TSU_NU` (つ/ぬ), `PERF_TARI_RI` (たり/り), `PAST_KI_KERI` (き/けり),
`CONJ_MU_RAMU` (む/らむ/けむ), `GOTOSHI` (ごとし), `N_TO_SU` (〜んとす),
`CAUS_SHIMU` (しむ), `RHETORICAL_NYA` (〜んや), `NEG_INTENT_MAJI` (まじ),
`EMPH_NAMU` (なむ).

**K. Written-register & newspaper-specific (N2-N1):** `DE_ARU` (である copula),
`TAIGEN_DOME` (体言止め nominal ending), `DERIV_TEKI` (〜的), `DERIV_KA` (〜化),
`DERIV_SEI` (〜性), `WO_MEGURU` (をめぐる), `MONO_TO_OMOWARERU` (〜ものと思われる),
`WO_YOGINAKU` (〜を余儀なくされる).

**Advanced tail of A-I promoted to 1.1:** finer particle senses (がの-conversion,
をもって, にて), rarer connectives (にもかかわらず, ところで), rarer set patterns
(〜てならない, 〜に越したことはない, 〜ないではいられない), rarer evidential/modal
(〜もようだ, 〜きらいがある), classical-derived formal nouns (〜次第, 〜あまり).
