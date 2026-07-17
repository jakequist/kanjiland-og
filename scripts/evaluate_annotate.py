"""M8 annotation-model eval: does the model emit VALID format?

The de-risk question for M8 isn't chrF — it's whether a from-scratch model can
generate our structured wire format at all. So we generate an annotation for each
held-out Japanese sentence and push it through the SAME parser + linter that gate
training data:

  parse-rate     : fraction of generations that parse as well-formed wire
  lint-pass-rate : fraction that ALSO satisfy every SPEC §7 invariant
  recon-rate     : fraction whose ⟨T⟩ surfaces reconstruct the input sentence
                   (the core "did it actually annotate THIS sentence" check)

    uv run python scripts/evaluate_annotate.py \
        --config configs/m8_annotate.yaml \
        --checkpoint checkpoints/m8-annotate/seed1/final.pt --limit 339
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from kanjiland.eval.translate import translate
from kanjiland.format.linter import lint
from kanjiland.format.parser import ParseError, parse
from kanjiland.model import ModelConfig, Transformer
from kanjiland.tokenizer import Tokenizer
from kanjiland.train.device import pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--valid", type=Path, default=Path("data/processed/m8_annot/valid.jsonl"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--show", type=int, default=2, help="example generations to print")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tok = Tokenizer.load(cfg["tokenizer"]["path"])
    device = pick_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ModelConfig.from_dict(ckpt["config"]["model"], vocab_size=tok.vocab_size)
    mcfg.pad_id = tok.pad_id
    model = Transformer(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"m8 eval (step {ckpt.get('step')}) on {device}")

    rows = [json.loads(l) for l in args.valid.read_text(encoding="utf-8").splitlines()][: args.limit]
    srcs = [r["ja"] for r in rows]
    gens = translate(
        model, tok, srcs, device, beam=args.beam,
        max_src=cfg["data"]["max_src_len"], max_len=cfg["data"]["max_tgt_len"],
        on_progress=lambda d, t: print(f"  {d}/{t}", flush=True) if d % 128 == 0 else None,
    )

    n = len(gens)
    parsed = lint_ok = recon_ok = 0
    shown = 0
    for src, wire in zip(srcs, gens):
        try:
            doc = parse(wire)
        except (ParseError, ValueError, Exception):  # noqa: BLE001
            continue
        parsed += 1
        viols = lint(doc, source_paragraphs=[src])
        if not viols:
            lint_ok += 1
        if not any(v.invariant == 3 for v in viols):  # reconstruction invariant
            recon_ok += 1
        if shown < args.show and not viols:
            print(f"\n--- valid generation ---\nJA: {src}")
            p = doc.paragraphs[0]
            print("EN:", p.sentences[0].translation if p.sentences else "(none)")
            print("grammar:", [g.rule_id for g in p.grammar])
            shown += 1

    print(f"\n=== M8 format eval on {n} held-out sentences ===")
    print(f"  parse-rate     : {parsed}/{n} ({100*parsed/n:.1f}%)")
    print(f"  lint-pass-rate : {lint_ok}/{n} ({100*lint_ok/n:.1f}%)")
    print(f"  reconstruct-ok : {recon_ok}/{n} ({100*recon_ok/n:.1f}%)")


if __name__ == "__main__":
    main()
