"""Annotation teacher (M7, offline — ADR-007 judgment layer).

MeCab/UniDic gave us the mechanical ⟨T⟩ skeleton (surface, ruby, lemma, pos). The
LLM teacher now supplies what UniDic can't: contextual glosses, word groupings
⟨W⟩, the sentence translation ⟨S⟩, and grammar-role labels ⟨G⟩ against the frozen
grammar-1.0 inventory. This is the harder cousin of the M6 translation teacher —
same client discipline, but structured output constrained to token ids + valid
rule_ids so the result assembles into a linter-passing Document.

Design choices that keep it cheap + reliable:
  - The 122-rule inventory is a FIXED prefix in the system prompt → prompt caching
    makes it ~10× cheaper on repeat calls (cached input rate).
  - We hand the model the token ids so ⟨G⟩/⟨W⟩ targets are unambiguous indices,
    not spans it has to re-derive.
  - JSON output, then we VALIDATE against the deterministic tokens + the inventory
    and lint the assembled Document; failures are dropped (synthetic-data hygiene),
    not trusted.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kanjiland.format.grammar import load_ruleset  # noqa: E402

RULESET = "grammar-1.0"
API_URL = "https://api.openai.com/v1/chat/completions"


def inventory_digest() -> str:
    """Compact one-line-per-rule listing of grammar-1.0 for the system prompt:
    RULE_ID [level] name — roles: r1(span,req) r2(token,opt)."""
    rules = load_ruleset(RULESET)
    lines = []
    for rid, spec in rules.items():
        roles = " ".join(
            f"{rn}({rd['target']},{'req' if rd['required'] else 'opt'})"
            for rn, rd in spec["roles"].items()
        )
        lines.append(f"{rid} [{spec['level']}] {spec['name']} — roles: {roles}")
    return "\n".join(lines)


def system_prompt() -> str:
    return f"""You annotate Japanese sentences for a reading-comprehension dataset. You are
given a sentence already segmented into morpheme TOKENS (with ids, surface, POS,
dictionary form). Do NOT re-segment. Produce ONLY a JSON object:

{{
  "tokens":   [{{"id": <int>, "gloss": "<contextual English meaning HERE>"}}, ...],
  "words":    [{{"span": [start,end], "dict": "<dictionary form>", "gloss": "<meaning>"}}],
  "sentences":[{{"span": [start,end], "translation": "<natural English>"}}],
  "grammar":  [{{"rule": "<RULE_ID>", "roles": {{"<role>": <target>, ...}}}}]
}}

Rules:
- gloss EVERY token id exactly once (punctuation gloss = "").
- span is half-open [start,end) over token ids; sentences MUST tile the whole
  paragraph (cover every id, no gaps/overlap).
- ⟨W⟩ words group tokens into learner-facing units (a verb + its auxiliaries);
  omit single-token words that add nothing. Spans must not overlap.
- A grammar "target" is either a token id (int) or a span [start,end].
- Use ONLY these rule_ids, and supply their required roles. Grammar inventory
  ({RULESET}):

{inventory_digest()}

Emit valid JSON only, no prose."""


def _chat(model: str, sys_p: str, user_p: str, api_key: str, max_retries: int = 4) -> dict | None:
    import time

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        "response_format": {"type": "json_object"},
    }
    if model.startswith("gpt-5.6"):
        payload["reasoning_effort"] = "none"
    body = json.dumps(payload).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                API_URL, data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            return json.loads(out["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < max_retries - 1:
                time.sleep(2.0 * (2**attempt))
                continue
            return None
        except Exception:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(2.0 * (2**attempt))
                continue
            return None
    return None


def user_prompt(sentence: str, morphs) -> str:
    lines = [f"Sentence: {sentence}", "Tokens:"]
    for i, m in enumerate(morphs):
        lines.append(f"  {i}: {m.surface}  [{m.pos}]  dict={m.dictionary_form}")
    return "\n".join(lines)


def annotate_one(sentence: str, morphs, api_key: str, model: str = "gpt-5.6-luna") -> dict | None:
    """Return the teacher's raw JSON annotation (unvalidated) or None on failure."""
    return _chat(model, system_prompt(), user_prompt(sentence, morphs), api_key)
