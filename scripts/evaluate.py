"""M4 evaluation harness: one command scores any checkpoint and updates the
results doc.

    uv run python scripts/evaluate.py --config configs/m3_transformer_base.yaml \
        --checkpoint checkpoints/m3-transformer-base/final.pt \
        --test-sets kftt-test,m2-test --beam 4

Translates each test set with the model, scores it with chrF + SacreBLEU + COMET
(ADR-008), and upserts the numbers into docs/reports/m4-results.{json,md}. Runs
with ≥2 seeds accumulate into mean±std automatically (seed-variance protocol).

Built-in test sets: ``kftt-test`` (formal), ``m2-test`` (mixed). Also accepts a
raw ``*.jsonl`` path, or a SacreBLEU set like ``wmt20`` (fetched for ja-en).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from kanjiland.eval import metrics, results
from kanjiland.eval.translate import translate
from kanjiland.model import ModelConfig, Transformer
from kanjiland.tokenizer import Tokenizer
from kanjiland.train.device import pick_device

BUILTIN_TEST_SETS = {
    "kftt-test": "data/processed/kftt-test.jsonl",
    "m2-test": "data/processed/test.jsonl",
}


def _read_jsonl(path: Path) -> tuple[list[str], list[str]]:
    ja, en = [], []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ja.append(o["ja"])
            en.append(o["en"])
    return ja, en


def load_test_set(name: str) -> tuple[list[str], list[str]]:
    """Return (sources_ja, references_en) for a named/pathed test set."""
    if name in BUILTIN_TEST_SETS:
        return _read_jsonl(BUILTIN_TEST_SETS[name])
    if name.endswith(".jsonl"):
        return _read_jsonl(Path(name))
    if name.startswith("wmt"):  # standard set fetched via SacreBLEU
        from sacrebleu.utils import get_reference_files, get_source_file

        src = Path(get_source_file(name, "ja-en")).read_text(encoding="utf-8").splitlines()
        ref = Path(get_reference_files(name, "ja-en")[0]).read_text(encoding="utf-8").splitlines()
        return src, ref
    raise ValueError(f"unknown test set: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--test-sets", default="kftt-test,m2-test")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--metrics", default="chrf,bleu,comet")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="override the seed label")
    ap.add_argument("--results", type=Path, default=Path("docs/reports/m4-results.json"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    tok = Tokenizer.load(cfg["tokenizer"]["path"])
    device = pick_device()
    wanted = args.metrics.split(",")

    # --- load model ---------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mcfg = ModelConfig.from_dict(ckpt["config"]["model"], vocab_size=tok.vocab_size)
    mcfg.pad_id = tok.pad_id
    model = Transformer(mcfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    run = ckpt["config"].get("run_name", args.checkpoint.stem)
    seed = args.seed if args.seed is not None else ckpt["config"].get("seed", 0)
    print(f"eval {run} (seed {seed}, step {ckpt.get('step')}) on {device}")

    comet = metrics.CometScorer() if "comet" in wanted else None
    records = results.load(args.results)

    max_src = cfg["data"]["max_src_len"]
    max_len = cfg["data"]["max_tgt_len"]
    for ts in args.test_sets.split(","):
        srcs, refs = load_test_set(ts)
        if args.limit:
            srcs, refs = srcs[: args.limit], refs[: args.limit]
        print(f"\n=== {ts}: {len(srcs)} pairs, translating (beam={args.beam}) ===", flush=True)
        hyps = translate(
            model,
            tok,
            srcs,
            device,
            beam=args.beam,
            max_src=max_src,
            max_len=max_len,
            on_progress=lambda d, t: print(f"  {d}/{t}", flush=True) if d % 512 == 0 else None,
        )

        m: dict = {}
        sig = None
        if "chrf" in wanted:
            m["chrf"] = metrics.chrf(hyps, refs)
        if "bleu" in wanted:
            m["bleu"], sig = metrics.bleu(hyps, refs)
        if "comet" in wanted:
            print("  scoring COMET ...", flush=True)
            try:
                m["comet"] = comet.score(srcs, hyps, refs)
            except Exception as e:  # never let the heavy metric lose chrF/BLEU
                print(f"  COMET failed ({type(e).__name__}: {e}); recording null", flush=True)
                m["comet"] = None

        record = {
            "run": run,
            "test_set": ts,
            "seed": seed,
            "beam": args.beam,
            "n": len(srcs),
            "metrics": m,
            "bleu_signature": sig,
        }
        results.upsert(records, record)
        print("  " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))

    # --- persist + regenerate the results doc -------------------------------
    results.save(args.results, records)
    md_path = args.results.with_suffix(".md")
    md_path.write_text(results.render_markdown(records), encoding="utf-8")
    print(f"\nupdated {args.results} and {md_path}")


if __name__ == "__main__":
    main()
