# Grammar Rule Inventory — grammar-0.1 (DRAFT)

Closed, versioned set of rule_ids for ⟨G⟩ records. The linter validates
against the version pinned in each file's header.

Sources to mine when expanding: Dictionary of Basic/Intermediate/Advanced
Japanese Grammar pattern lists; Bunpro grammar-point inventory (as a
checklist, not content); UD-Japanese relations for the dependency-ish rules.

## Schema

```yaml
RULE_ID:
  name: human-readable name
  level: N5..N1            # JLPT-ish difficulty, for UI filtering
  roles:
    role_name: {target: token|span, required: true|false}
  description: one-line learner-facing summary
```

## Seed rules (v0.1 — expand before M7, then freeze)

```yaml
TOPIC_WA:
  name: Topic marking with は
  level: N5
  roles:
    topic:  {target: span,  required: true}
    marker: {target: token, required: true}
    scope:  {target: span,  required: false}
  description: The は particle marks what the sentence is about.

SUBJECT_GA:
  name: Subject marking with が
  level: N5
  roles:
    subject: {target: span,  required: true}
    marker:  {target: token, required: true}
  description: が marks the grammatical subject.

OBJECT_WO:
  name: Direct object with を
  level: N5
  roles:
    object: {target: span,  required: true}
    marker: {target: token, required: true}
    verb:   {target: span,  required: false}
  description: を marks the direct object of a verb.

COPULA_POLITE:
  name: Polite copula です
  level: N5
  roles:
    complement: {target: span,  required: true}
    copula:     {target: token, required: true}
  description: X です — "is X" (polite).

TE_FORM_PROGRESSIVE:
  name: 〜ている progressive/resultative
  level: N5
  roles:
    verb: {target: span, required: true}   # the whole 〜ている unit
  description: Ongoing action or resulting state.

TE_FORM_REQUEST:
  name: 〜てください request
  level: N5
  roles:
    verb: {target: span, required: true}
  description: Polite request "please do X".

NEGATIVE_NAI:
  name: 〜ない negation
  level: N5
  roles:
    predicate: {target: span, required: true}
  description: Plain negative form.

PAST_TA:
  name: 〜た past tense
  level: N5
  roles:
    predicate: {target: span, required: true}
  description: Plain past form.

CONDITIONAL_TARA:
  name: 〜たら conditional
  level: N4
  roles:
    condition: {target: span, required: true}
    result:    {target: span, required: false}
  description: "If/when X, then Y."

NOMINALIZER_NO:
  name: の nominalizer
  level: N4
  roles:
    clause:      {target: span,  required: true}
    nominalizer: {target: token, required: true}
  description: Turns a clause into a noun phrase.

POTENTIAL:
  name: Potential form (〜られる/〜える)
  level: N4
  roles:
    verb: {target: span, required: true}
  description: "Can do X."

PASSIVE:
  name: Passive (〜られる/〜れる)
  level: N4
  roles:
    verb:  {target: span, required: true}
    agent: {target: span, required: false}   # に-marked agent if present
  description: Passive voice; agent marked with に when expressed.
```

## Expansion policy

- Target ~80–150 rules for v1 (N5–N3 coverage) before freezing grammar-1.0.
- Every added rule needs: schema entry + ≥3 positive examples in the test
  fixtures + linter validation test.
- Renames/removals bump the major ruleset version (data migration required).
