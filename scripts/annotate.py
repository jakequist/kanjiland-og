"""Runtime annotation entry point (M9) — the SHIPPED inference path.

Raw Japanese in → the full ⟨T⟩/⟨W⟩/⟨S⟩/⟨G⟩ annotation out, produced entirely by
the from-scratch model. Deliberately imports NO NLP dependency (rule #1): the M8
model learned segmentation/ruby itself, so there is no MeCab/UniDic at inference —
only the from-scratch tokenizer + transformer. This is what would run on-device.

    uv run python scripts/annotate.py --config configs/m8_annotate.yaml \
        --checkpoint checkpoints/m8-annotate/seed1/final.pt \
        --text "彼は古い寺を訪れた。" [--device cpu] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from kanjiland.format.parser import ParseError, parse
from kanjiland.model import ModelConfig, Transformer, greedy_decode
from kanjiland.tokenizer import Tokenizer


def load_model(config: Path, checkpoint: Path, device: str):
    cfg = yaml.safe_load(config.read_text())
    tok = Tokenizer.load(cfg["tokenizer"]["path"])
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    mcfg = ModelConfig.from_dict(ckpt["config"]["model"], vocab_size=tok.vocab_size)
    mcfg.pad_id = tok.pad_id
    model = Transformer(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, tok, cfg


def annotate(text: str, model, tok: Tokenizer, cfg: dict, device: str) -> str:
    """Japanese text → annotation wire string (the model does everything).

    Generation MUST run under bf16 autocast — the model trained that way, and this
    small model is numerically fragile: in fp32 its long autoregressive generation
    diverges into unparseable output. bf16 autocast works on both cuda and cpu.
    """
    ids = tok.encode(text)[: cfg["data"]["max_src_len"] - 1] + [tok.eos_id]
    src = torch.tensor([ids], device=device)
    dt = "cuda" if str(device).startswith("cuda") else "cpu"
    ys = None
    with torch.no_grad(), torch.autocast(device_type=dt, dtype=torch.bfloat16):
        ys = greedy_decode(model, src, tok.bos_id, tok.eos_id, tok.pad_id, cfg["data"]["max_tgt_len"])
    lst = ys[0].tolist()
    if lst and lst[0] == tok.bos_id:
        lst = lst[1:]
    if tok.eos_id in lst:
        lst = lst[: lst.index(tok.eos_id)]  # truncate at first EOS (match eval path)
    return tok.decode(lst)


def to_display(wire: str) -> dict:
    """Parse the wire into a UI-ready structure (same shape the demo renders)."""
    doc = parse(wire)
    p = doc.paragraphs[0]
    return {
        "tokens": [{"surface": t.surface, "ruby": list(t.ruby), "gloss": t.gloss,
                    "pos": t.pos, "dict": t.dictionary_form} for t in p.tokens],
        "translation": p.sentences[0].translation if p.sentences else "",
        "grammar": [{"rule": g.rule_id,
                     "roles": [{"role": r, **({"tok": v} if isinstance(v, int) else {"span": [v.start, v.end]})}
                               for r, v in g.roles]} for g in p.grammar],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--text", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", action="store_true", help="emit parsed display JSON")
    args = ap.parse_args()

    model, tok, cfg = load_model(args.config, args.checkpoint, args.device)
    wire = annotate(args.text, model, tok, cfg, args.device)
    if args.json:
        try:
            print(json.dumps(to_display(wire), ensure_ascii=False, indent=2))
        except (ParseError, ValueError) as e:
            sys.exit(f"model produced unparseable output: {e}\n---\n{wire}")
    else:
        from kanjiland.format import to_debug
        print(to_debug(wire))


if __name__ == "__main__":
    main()
