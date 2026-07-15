"""Pre-tokenize the parallel corpus to the compact binary the trainer reads.

Runs the BPE encoder once, offline and in parallel across CPU cores, so training
never pays for Python tokenization. For each split it writes three files under
``<bin_dir>/<split>``:

    <split>.src.bin   uint16 stream of source (ja) token ids
    <split>.tgt.bin   uint16 stream of target (en) token ids
    <split>.idx.npy   (N, 4) int64: [src_off, src_len, tgt_off, tgt_len]

Special tokens are baked in here so the data layer stays a pure slice:
    src = <ja ids, truncated>  EOS
    tgt = BOS  <en ids, truncated>  EOS

Usage:
    uv run python scripts/pretokenize.py --config configs/m3_transformer_base.yaml
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import yaml

from kanjiland.tokenizer import Tokenizer

# Per-worker globals (set in the pool initializer so the tokenizer is loaded
# once per process, not pickled per task).
_TOK: Tokenizer | None = None
_MAX_SRC = 128
_MAX_TGT = 128


def _init_worker(tok_path: str, max_src: int, max_tgt: int) -> None:
    global _TOK, _MAX_SRC, _MAX_TGT
    _TOK = Tokenizer.load(tok_path)
    _MAX_SRC, _MAX_TGT = max_src, max_tgt


def _encode_chunk(lines: list[str]) -> list[tuple[list[int], list[int]]]:
    tok = _TOK
    out = []
    for line in lines:
        o = json.loads(line)
        # Truncate leaving room for the special tokens, then attach them.
        src = tok.encode(o["ja"])[: _MAX_SRC - 1] + [tok.eos_id]
        tgt = [tok.bos_id] + tok.encode(o["en"])[: _MAX_TGT - 2] + [tok.eos_id]
        out.append((src, tgt))
    return out


def _chunks(path: Path, size: int):
    buf: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            buf.append(line)
            if len(buf) >= size:
                yield buf
                buf = []
    if buf:
        yield buf


def pretokenize_split(
    jsonl: Path, out_prefix: Path, tok_path: str, max_src: int, max_tgt: int, workers: int
) -> int:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    src_f = open(f"{out_prefix}.src.bin", "wb")
    tgt_f = open(f"{out_prefix}.tgt.bin", "wb")
    idx: list[tuple[int, int, int, int]] = []
    src_off = tgt_off = 0
    n = 0

    with mp.Pool(workers, initializer=_init_worker, initargs=(tok_path, max_src, max_tgt)) as pool:
        # imap preserves order (so idx aligns with the input) while workers
        # encode chunks concurrently.
        for encoded in pool.imap(_encode_chunk, _chunks(jsonl, 20_000)):
            for src, tgt in encoded:
                src_f.write(np.asarray(src, dtype=np.uint16).tobytes())
                tgt_f.write(np.asarray(tgt, dtype=np.uint16).tobytes())
                idx.append((src_off, len(src), tgt_off, len(tgt)))
                src_off += len(src)
                tgt_off += len(tgt)
                n += 1
            if n % 1_000_000 < 20_000:
                print(f"  {out_prefix.name}: {n:,} pairs", flush=True)

    src_f.close()
    tgt_f.close()
    np.save(f"{out_prefix}.idx.npy", np.asarray(idx, dtype=np.int64))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--bin-dir", type=Path, default=None, help="override config data.bin_dir")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    bin_dir = args.bin_dir or Path(cfg.get("data", {}).get("bin_dir", "data/processed/tok"))
    args.bin_dir = bin_dir
    tok_path = cfg["tokenizer"]["path"]
    max_src = cfg["data"]["max_src_len"]
    max_tgt = cfg["data"]["max_tgt_len"]

    # Splits: train + valid from config; test.jsonl if present next to train.
    splits = {"train": Path(cfg["data"]["train"]), "valid": Path(cfg["data"]["valid"])}
    test = Path(cfg["data"]["train"]).with_name("test.jsonl")
    if test.exists():
        splits["test"] = test

    print(f"tokenizer={tok_path} max_src={max_src} max_tgt={max_tgt} workers={args.workers}")
    for name, path in splits.items():
        print(f"=== {name}: {path} ===", flush=True)
        n = pretokenize_split(path, args.bin_dir / name, tok_path, max_src, max_tgt, args.workers)
        print(f"  wrote {n:,} pairs -> {args.bin_dir / name}.*", flush=True)


if __name__ == "__main__":
    main()
