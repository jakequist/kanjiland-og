"""Evaluate an M3 checkpoint on a test set: chrF vs the word-substitution baseline.

    uv run python scripts/evaluate.py --config configs/m3_transformer_base.yaml \
        --checkpoint checkpoints/m3-transformer-base/final.pt --split test

chrF (character n-gram F-score) is our iteration metric (ADR-008) — robust for
Japanese→English and cheap to compute. The full metric stack (SacreBLEU + COMET)
is M4; this is the M3 "does it beat the baseline / is the English fluent" check.
Runs the baseline even without a --checkpoint, to print the floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from kanjiland.eval.baseline import build_lexicon, translate as baseline_translate
from kanjiland.model import ModelConfig, Transformer, beam_search, greedy_decode
from kanjiland.tokenizer import Tokenizer
from kanjiland.train.device import amp_context, pick_device


def _read_pairs(path: Path):
    ja, en = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ja.append(o["ja"])
            en.append(o["en"])
    return ja, en


def _ids_to_text(ids, tok: Tokenizer) -> str:
    lst = ids.tolist()
    if lst and lst[0] == tok.bos_id:
        lst = lst[1:]
    if tok.eos_id in lst:
        lst = lst[: lst.index(tok.eos_id)]
    return tok.decode(lst)


@torch.no_grad()
def model_translate(model, tok, sentences, device, beam, max_src, max_len, batch_size=64):
    """Translate ja sentences -> en strings (greedy if beam<=1 else beam)."""
    out: list[str] = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        enc = [tok.encode(s)[: max_src - 1] + [tok.eos_id] for s in chunk]
        width = max(len(e) for e in enc)
        src = torch.full((len(enc), width), tok.pad_id, dtype=torch.long)
        for j, e in enumerate(enc):
            src[j, : len(e)] = torch.tensor(e)
        src = src.to(device)
        with amp_context(device):
            if beam > 1:
                gen = beam_search(model, src, tok.bos_id, tok.eos_id, tok.pad_id, beam, max_len)
            else:
                gen = greedy_decode(model, src, tok.bos_id, tok.eos_id, tok.pad_id, max_len)
        out.extend(_ids_to_text(g, tok) for g in gen)
        print(f"  translated {min(i + batch_size, len(sentences))}/{len(sentences)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only N examples")
    args = ap.parse_args()

    import sacrebleu

    cfg = yaml.safe_load(args.config.read_text())
    tok = Tokenizer.load(cfg["tokenizer"]["path"])
    device = pick_device()

    test_path = Path(cfg["data"]["train"]).with_name(f"{args.split}.jsonl")
    ja, en = _read_pairs(test_path)
    if args.limit:
        ja, en = ja[: args.limit], en[: args.limit]
    print(f"eval {args.split}: {len(ja)} pairs")

    # --- baseline -----------------------------------------------------------
    train_ja, train_en = _read_pairs(Path(cfg["data"]["train"]))
    print("building baseline lexicon ...")
    lex = build_lexicon(zip(train_ja, train_en), tok, max_pairs=200_000)
    base_hyps = [baseline_translate(s, tok, lex) for s in ja]
    base_chrf = sacrebleu.corpus_chrf(base_hyps, [en]).score
    print(f"baseline chrF: {base_chrf:.2f}")

    # --- model --------------------------------------------------------------
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        mcfg = ModelConfig.from_dict(ckpt["config"]["model"], vocab_size=tok.vocab_size)
        mcfg.pad_id = tok.pad_id
        model = Transformer(mcfg).to(device).eval()
        model.load_state_dict(ckpt["model"])
        print(f"loaded {args.checkpoint} (step {ckpt.get('step')})")
        hyps = model_translate(
            model,
            tok,
            ja,
            device,
            args.beam,
            cfg["data"]["max_src_len"],
            cfg["data"]["max_tgt_len"],
        )
        model_chrf = sacrebleu.corpus_chrf(hyps, [en]).score
        print(f"\nmodel chrF:    {model_chrf:.2f}  (beam={args.beam})")
        print(f"baseline chrF: {base_chrf:.2f}")
        print(f"delta:         {model_chrf - base_chrf:+.2f}")
        print("\n--- samples ---")
        for s, h, r in list(zip(ja, hyps, en))[:5]:
            print(f"JA:  {s}\nGEN: {h}\nREF: {r}\n")


if __name__ == "__main__":
    main()
