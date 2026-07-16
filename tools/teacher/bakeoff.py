"""Teacher bake-off (M6): compare candidate OpenAI models as the KD teacher.

We don't know a priori which model to distill from — "be empirical." This scores
each candidate's Ja→En on a held-out set WITH references (kftt-test), so quality
is measured objectively (chrF/BLEU vs the human reference), alongside the token
cost + latency that decide whether it's affordable at 500k-sentence scale.

    OPENAI_API_KEY=... uv run --extra eval python tools/teacher/bakeoff.py \
        --models gpt-4.1-mini,gpt-5-mini,gpt-5.6-luna,gpt-5.6-sol,gpt-5.6-terra \
        --n 120 --seed 1

Note: kftt-test is formal/Wikipedia-domain; it measures teacher quality on hard,
in-domain text. The chosen teacher then translates *monolingual* Ja (no refs) to
make the KD training set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from translate import translate_many  # noqa: E402 — sibling module under tools/teacher

KFTT = Path("data/processed/kftt-test.jsonl")

# USD per 1M tokens (input, output). Fill from the current OpenAI pricing page;
# None => unknown, cost is reported in TOKENS only and flagged. Token *usage* is
# always measured exactly from the API, so a missing price is a one-number fix.
PRICES: dict[str, tuple[float, float] | None] = {
    "gpt-4.1-mini": (0.40, 1.60),   # known reference tier
    "gpt-5-mini": (0.25, 2.00),     # confirm
    "gpt-5.6-luna": None,           # brand-new — needs price
    "gpt-5.6-sol": None,
    "gpt-5.6-terra": None,
}


def load_sample(n: int, seed: int) -> tuple[list[str], list[str]]:
    """Seeded sample of *real* sentences (skip markup/tiny fragments like the
    'InfoboxBuddhist' header lines) so chrF reflects genuine translation."""
    pairs = []
    for line in KFTT.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if 15 <= len(o["ja"]) <= 150 and any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in o["ja"]):
            pairs.append((o["ja"], o["en"]))
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:n]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def score(hyps: list[str], refs: list[str]) -> dict:
    import sacrebleu

    # blank failed translations to empty string so they score 0, not crash
    h = [x or "" for x in hyps]
    return {
        "chrf": sacrebleu.corpus_chrf(h, [refs]).score,
        "bleu": sacrebleu.corpus_bleu(h, [refs]).score,
    }


def cost_usd(model: str, prompt_tok: int, completion_tok: int) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return prompt_tok / 1e6 * p[0] + completion_tok / 1e6 * p[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("docs/reports/m6-teacher-bakeoff.md"))
    ap.add_argument("--dump", type=Path, default=Path("docs/reports/_m6_bakeoff_raw.json"))
    args = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("set OPENAI_API_KEY")

    srcs, refs = load_sample(args.n, args.seed)
    print(f"bake-off on {len(srcs)} kftt-test sentences, {len(args.models.split(','))} models\n")

    rows = []
    raw = {"n": len(srcs), "seed": args.seed, "models": {}}
    for model in args.models.split(","):
        t0 = time.time()
        hyps, u = translate_many(srcs, model, key, workers=args.workers)
        wall = time.time() - t0
        metrics = score(hyps, refs)
        per_sent_in = u.prompt_tokens / max(u.n, 1)
        per_sent_out = u.completion_tokens / max(u.n, 1)
        c = cost_usd(model, u.prompt_tokens, u.completion_tokens)
        per_1k = (c / max(u.n, 1) * 1000) if c is not None else None
        rows.append({
            "model": model, "chrf": metrics["chrf"], "bleu": metrics["bleu"],
            "in": per_sent_in, "out": per_sent_out, "reason": u.reasoning_tokens,
            "errors": u.errors, "wall": wall, "per_1k": per_1k,
            "proj_500k": (per_1k * 500 if per_1k is not None else None),
        })
        raw["models"][model] = {
            "metrics": metrics, "usage": vars(u), "hyps": hyps,
        }
        pk = f"${per_1k:.3f}" if per_1k is not None else "$? (add price)"
        print(f"  {model:16s} chrF={metrics['chrf']:5.2f} BLEU={metrics['bleu']:5.2f} "
              f"in/out={per_sent_in:.0f}/{per_sent_out:.0f} tok  errors={u.errors}  "
              f"{wall:4.0f}s  {pk}/1k")

    args.dump.parent.mkdir(parents=True, exist_ok=True)
    args.dump.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown report
    L = [
        "# M6 — Teacher bake-off",
        "",
        f"Ja→En on {len(srcs)} held-out kftt-test sentences (seed {args.seed}), scored vs "
        "the human reference. Quality = chrF/BLEU (higher better). Cost from measured "
        "token usage; `$?` = price not yet filled in `PRICES` (tools/teacher/bakeoff.py).",
        "",
        "| model | chrF | BLEU | in tok | out tok | reasoning | errors | $/1k | proj 500k |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in sorted(rows, key=lambda x: -x["chrf"]):
        pk = f"${r['per_1k']:.3f}" if r["per_1k"] is not None else "$?"
        p5 = f"${r['proj_500k']:.0f}" if r["proj_500k"] is not None else "$?"
        L.append(f"| {r['model']} | {r['chrf']:.2f} | {r['bleu']:.2f} | {r['in']:.0f} | "
                 f"{r['out']:.0f} | {r['reason']} | {r['errors']} | {pk} | {p5} |")
    L += ["", "Projected 500k = the KD dry-run corpus cost at this rate. Fill unknown "
          "prices in PRICES and re-render, or read tokens directly.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nreport -> {args.out}\nraw hyps -> {args.dump}")


if __name__ == "__main__":
    main()
