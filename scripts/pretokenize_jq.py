#
# import argparse
# import json
# import multiprocessing as mp
# from pathlib import Path
# from typing import List
#
# import numpy as np
# import yaml
#
# from kanjiland.tokenizer import Tokenizer
#
# import argparse
#
# _MAX_SRC = 128
# _MAX_TGT = 128
#
#
# def encode(ja, en) -> tuple[List[int], List[int]]:
#     src = Tokenizer.encode(ja)[: _MAX_SRC - 1] + [Tokenizer.eos_id]
#     tgt = [Tokenizer.bos_id] + Tokenizer.encode(en)[: _MAX_TGT - 2] + [Tokenizer.eos_id]
#     return (src, tgt)
#
#
#
# def pretokenize_split(
#     jsonl: Path, out_prefix: Path, tok_path: str, max_src: int, max_tgt: int, workers: int
# ) -> int:
#
#     print(f"jsonl: {jsonl}")
#     print(f"out_prefix: {out_prefix}")
#     print(f"tok_path: {tok_path}")
#     print(f"max_src: {max_src}")
#     print(f"max_tgt: {max_tgt}")
#
#     src_path = Path(f"{out_prefix}.src.bin")
#     tgt_path = Path(f"{out_prefix}.tgt.bin")
#     src_out = []
#     tgt_out = []
#
#     data = [json.loads(line) for line in Path.read_text(jsonl).splitlines()]
#
#     for d in data:
#         src = d["ja"]
#         tgt = d["en"]
#         (src_bin, tgt_bin) = encode(src, tgt)
#         src_out.append(src_bin)
#         tgt_out.append(tgt_bin)
#
#     tgt_path.write_bytes(np.array(tgt_out, dtype=np.uint16).tobytes())
#     src_path.write_bytes(np.array(src_out, dtype=np.uint16).tobytes())
#
#     bin_data = convert(data)
#     print(f"data size: {len(data)}")
#
#
#
#
#
#
#
#     print("\n\n\nDONE")
#     raise NotImplementedError
#
#
#
# def main() -> None:
#     ap = argparse.ArgumentParser(description=__doc__)
#     ap.add_argument("--config", required=True, type=Path)
#     ap.add_argument("--bin-dir", type=Path, default=None, help="override config data.bin_dir")
#     ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
#     args = ap.parse_args()
#
#     cfg = yaml.safe_load(args.config.read_text())
#     bin_dir = args.bin_dir or Path(cfg.get("data", {}).get("bin_dir", "data/processed/tok"))
#     args.bin_dir = bin_dir
#     tok_path = cfg["tokenizer"]["path"]
#     max_src = cfg["data"]["max_src_len"]
#     max_tgt = cfg["data"]["max_tgt_len"]
#
#     # Splits: train + valid from config; test.jsonl if present next to train.
#     splits = {"train": Path(cfg["data"]["train"]), "valid": Path(cfg["data"]["valid"])}
#     test = Path(cfg["data"]["train"]).with_name("test.jsonl")
#     if test.exists():
#         splits["test"] = test
#
#     print(f"tokenizer={tok_path} max_src={max_src} max_tgt={max_tgt} workers={args.workers}")
#     for name, path in splits.items():
#         print(f"=== {name}: {path} ===", flush=True)
#         n = pretokenize_split(path, args.bin_dir / name, tok_path, max_src, max_tgt, args.workers)
#         #print(f"  wrote {n:,} pairs -> {args.bin_dir / name}.*", flush=True)
#
#
# if __name__ == "__main__":
#     main()