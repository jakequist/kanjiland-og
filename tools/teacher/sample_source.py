"""Sample monolingual Japanese source sentences for KD (M6, offline).

Produces the shared input for BOTH arms of the distillation experiment:
  - ja.txt      : Japanese sentences, one per line -> sent to the teacher
  - pairs.jsonl : {i, ja, en} aligned by line index -> `en` is the HUMAN reference
                  that the baseline arm trains on (and that we hold constant)

The two arms differ only in the target column (teacher-En vs this human-En), so
this file IS the controlled variable's anchor. We sample from KFTT (formal,
proper-name-heavy — where the luna teacher's accuracy matters and where our
kftt-test eval lives), length-filter to what a 128-token student can actually
learn, and HOLD OUT anything appearing in kftt-test to prevent train/test leak.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def has_jp(s: str) -> bool:
    return any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/processed/_m2_tmp/kftt.jsonl"))
    ap.add_argument("--holdout", type=Path, default=Path("data/processed/kftt-test.jsonl"))
    ap.add_argument("--n", type=int, default=185_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ja-min", type=int, default=10)
    ap.add_argument("--ja-max", type=int, default=160)   # ~fits 128-tok context + caps teacher cost
    ap.add_argument("--en-min", type=int, default=5)
    ap.add_argument("--en-max", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("data/processed/m6_kd"))
    args = ap.parse_args()

    # hold out test sentences (both sides, to be safe against near-dupes)
    hold = set()
    for line in args.holdout.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        hold.add(o["ja"])

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in args.src.read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        ja, en = o["ja"].strip(), o["en"].strip()
        if not (args.ja_min <= len(ja) <= args.ja_max and args.en_min <= len(en) <= args.en_max):
            continue
        if not has_jp(ja) or ja in hold or ja in seen:
            continue
        seen.add(ja)
        pairs.append((ja, en))

    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.n]

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "ja.txt").open("w", encoding="utf-8") as jf, \
         (args.out / "pairs.jsonl").open("w", encoding="utf-8") as pf:
        for i, (ja, en) in enumerate(pairs):
            jf.write(ja + "\n")
            pf.write(json.dumps({"i": i, "ja": ja, "en": en}, ensure_ascii=False) + "\n")

    # rough JA token estimate for cost projection (o200k ~ chars for JP; refined
    # by the actual usage the batch reports)
    ja_chars = sum(len(p[0]) for p in pairs)
    print(f"sampled {len(pairs)} pairs (of {len(seen)} eligible) -> {args.out}/")
    print(f"avg ja chars = {ja_chars/max(len(pairs),1):.1f} (~token/char for JP)")


if __name__ == "__main__":
    main()
