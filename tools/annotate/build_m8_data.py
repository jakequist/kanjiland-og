"""Build the M8 annotation-model training set from M7 silver data (offline).

Source→target for M8 is (Japanese sentence → full annotation WIRE string). We
reuse the standard {ja, en} training pipeline by putting the wire in the `en`
field — the 16k tokenizer already has the PUA separators as special tokens (M1),
so the wire tokenizes natively.

Length-filter to what a 52M model can plausibly learn on this little data: the
annotation wire averages ~800 tokens (p95 ~1620), so we keep only src<=128 and
tgt<=1024-token examples (no truncation of kept data). Small on purpose — this is
the minimum-spend de-risk; expand the silver set later if the approach holds.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from kanjiland.tokenizer import Tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--silver", type=Path, default=Path("data/processed/m7_annot/silver.jsonl"))
    ap.add_argument("--tokenizer", type=Path, default=Path("data/processed/tokenizer-16k.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/m8_annot"))
    ap.add_argument("--max-src", type=int, default=128)
    ap.add_argument("--max-tgt", type=int, default=1024)
    ap.add_argument("--valid", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    tok = Tokenizer.load(str(args.tokenizer))
    kept = []
    for line in args.silver.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if len(tok.encode(r["ja"])) <= args.max_src and len(tok.encode(r["wire"])) <= args.max_tgt:
            kept.append({"ja": r["ja"], "en": r["wire"]})  # en field = annotation target

    random.Random(args.seed).shuffle(kept)
    nval = min(args.valid, len(kept) // 20)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("valid", kept[:nval]), ("train", kept[nval:])):
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} (src<={args.max_src}, tgt<={args.max_tgt} tok) -> "
          f"{len(kept)-nval} train + {nval} valid -> {args.out}/")


if __name__ == "__main__":
    main()
