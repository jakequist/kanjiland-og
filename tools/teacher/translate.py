"""OpenAI teacher-translation client (M6, offline data-gen — ADR-007).

This lives under tools/ and is NEVER imported by src/kanjiland: the teacher is an
OFFLINE label source (it manufactures synthetic Ja→En parallel data for
sequence-level KD), not part of the shipped from-scratch model or its runtime
(rule #1, ADR-010). Raw stdlib HTTP on purpose — no SDK dependency, and it keeps
the request/retry/usage-accounting logic visible and auditable.

Two API quirks this wraps so callers don't have to care:
  1. `reasoning_effort` is family-specific. Translation needs NO deliberation, and
     reasoning tokens are billed — gpt-5-mini spent 212 completion tokens on a
     7-word translation at its default effort, ~15× what the visible output costs.
     So we force the LEANEST supported setting per family and fall back gracefully
     when a model rejects the parameter (gpt-4.x doesn't know it at all).
  2. Rate limits / transient 5xx are normal at scale — exponential backoff retry.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

API_URL = "https://api.openai.com/v1/chat/completions"

# Lean reasoning setting per family (empirically probed 2026-07):
#   gpt-4.x / gpt-4o : parameter unknown -> must omit
#   gpt-5 / gpt-5-mini / gpt-5-nano : accept "minimal" (collapses reasoning spend)
#   gpt-5.6-*        : reject "minimal"; "none" is the lean setting
# Anything unmatched: try "minimal", and the 400-fallback below drops it if wrong.
def _effort_for(model: str) -> str | None:
    if model.startswith(("gpt-4.1", "gpt-4o", "gpt-4-")):
        return None
    if model.startswith("gpt-5.6"):
        return "none"
    if model.startswith("gpt-5"):
        return "minimal"
    return "minimal"


SYSTEM_PROMPT = (
    "You are a professional Japanese-to-English translator. "
    "Translate the user's Japanese text into natural, fluent English. "
    "Output ONLY the English translation — no notes, quotes, or romaji."
)


@dataclass
class Usage:
    """Running token/latency/error tally across a batch — drives the cost report."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    n: int = 0
    errors: int = 0
    seconds: float = 0.0

    def add(self, u: dict, dt: float) -> None:
        self.prompt_tokens += u.get("prompt_tokens", 0)
        self.completion_tokens += u.get("completion_tokens", 0)
        self.reasoning_tokens += u.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or 0
        self.n += 1
        self.seconds += dt


def _post(payload: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def translate_one(
    text: str,
    model: str,
    api_key: str,
    *,
    max_retries: int = 5,
    timeout: float = 120.0,
) -> tuple[str | None, dict, float]:
    """Translate one sentence. Returns (translation|None, usage_dict, seconds).

    Retries 429/5xx with exponential backoff; on a 400 that names
    `reasoning_effort`, drops the parameter and retries once (auto-adapts to a
    family whose effort vocabulary we guessed wrong).
    """
    payload = {"model": model, "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]}
    effort = _effort_for(model)
    if effort is not None:
        payload["reasoning_effort"] = effort

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            r = _post(payload, api_key, timeout)
            out = r["choices"][0]["message"]["content"].strip()
            return out, r.get("usage", {}), time.time() - t0
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 400 and "reasoning_effort" in body and "reasoning_effort" in payload:
                del payload["reasoning_effort"]  # this family doesn't take it — retry clean
                continue
            if e.code in (429, 500, 502, 503, 529) and attempt < max_retries - 1:
                time.sleep(2.0 * (2**attempt))  # 2,4,8,16s backoff on rate-limit/5xx
                continue
            return None, {"error": f"HTTP {e.code}: {body[:120]}"}, time.time() - t0
        except Exception as e:  # noqa: BLE001 — network flakiness shouldn't kill a 500k run
            if attempt < max_retries - 1:
                time.sleep(2.0 * (2**attempt))
                continue
            return None, {"error": f"{type(e).__name__}: {e}"}, time.time() - t0
    return None, {"error": "exhausted retries"}, time.time() - t0


def translate_many(
    texts: list[str],
    model: str,
    api_key: str,
    *,
    workers: int = 8,
    on_result=None,
) -> tuple[list[str | None], Usage]:
    """Concurrently translate many sentences, preserving input order.

    Concurrency is how a per-sentence pipeline stays fast without batching many
    sentences into one prompt (which risks misalignment + all-or-nothing failure).
    `workers` trades throughput against rate limits.
    """
    results: list[str | None] = [None] * len(texts)
    usage = Usage()

    def job(i: int) -> None:
        out, u, dt = translate_one(texts[i], model, api_key)
        results[i] = out
        if "error" in u:
            usage.errors += 1
        else:
            usage.add(u, dt)
        if on_result:
            on_result(i, out, u)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(job, range(len(texts))))
    return results, usage
