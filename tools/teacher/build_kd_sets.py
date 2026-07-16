"""Assemble the two matched training sets for the KD experiment (M6, offline).

From the shared source (pairs.jsonl) + the teacher's translations (teacher_en.jsonl)
it writes two corpora that differ in ONE thing — the English target:

    kd/{train,valid}.jsonl        (ja, en = TEACHER translation)
    baseline/{train,valid}.jsonl  (ja, en = HUMAN reference)

Synthetic-data hygiene (an explicit M6 learning target): a teacher can fail —
empty output, a refusal, an echo of the Japanese, or a wildly off-length answer.
Training on that garbage would unfairly handicap the KD arm, so we DROP those
rows. Crucially we drop the SAME row index from BOTH arms, so the two corpora
stay a matched pair (identical sentences, identical split) and the only variable
is teacher-vs-human targets. The drop count is itself a result — it measures how
clean the teacher's bulk output is.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REFUSAL = re.compile(r"\b(I('?m| am| cannot| can't)|sorry|as an ai|unable to)\b", re.I)


def cjk_frac(s: str) -> float:
    if not s:
        return 1.0
    cjk = sum(1 for c in s if "぀" <= c <= "ヿ" or "一" <= c <= "鿿")
    return cjk / len(s)


def teacher_ok(ja: str, en: str | None) -> bool:
    """Keep only clean English translations."""
    if not en or not en.strip():
        return False                              # empty / batch error
    en = en.strip()
    if REFUSAL.search(en[:40]):
        return False                              # model refused / hedged
    if cjk_frac(en) > 0.15:
        return False                              # still (mostly) Japanese / untranslated
    r = len(en) / max(len(ja), 1)
    if not (0.3 <= r <= 8.0):
        return False                              # implausible length vs source
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/processed/m6_kd"))
    ap.add_argument("--valid", type=int, default=2000, help="held-out valid rows")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    pairs = {json.loads(l)["i"]: json.loads(l) for l in (args.dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()}
    teach = {json.loads(l)["i"]: json.loads(l)["en"] for l in (args.dir / "teacher_en.jsonl").read_text(encoding="utf-8").splitlines()}

    kept, dropped = [], 0
    for i, p in pairs.items():
        en_t = teach.get(i)
        if teacher_ok(p["ja"], en_t):
            kept.append((p["ja"], p["en"], en_t.strip()))
        else:
            dropped += 1

    random.Random(args.seed).shuffle(kept)
    valid, train = kept[: args.valid], kept[args.valid:]
    print(f"kept {len(kept)} / {len(pairs)} ({dropped} dropped by hygiene) "
          f"-> {len(train)} train + {len(valid)} valid")

    def dump(arm: str, target_idx: int) -> None:
        d = args.dir / arm
        d.mkdir(parents=True, exist_ok=True)
        for split, rows in (("train", train), ("valid", valid)):
            with (d / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps({"ja": row[0], "en": row[target_idx]}, ensure_ascii=False) + "\n")

    dump("baseline", 1)   # human En
    dump("kd", 2)         # teacher En
    print(f"wrote {args.dir}/kd/ and {args.dir}/baseline/ (matched split, seed {args.seed})")


if __name__ == "__main__":
    main()
