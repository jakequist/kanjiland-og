"""Build the M2 parallel corpus: sources -> cleaning funnel -> train/valid/test.

Streaming end to end so it scales to JParaCrawl (~25M pairs) without holding the
corpus in memory:

  1. Each source streams through the cheap filters (``clean_stream``) and its
     survivors are written to a per-source temp JSONL.
  2. LaBSE-flagged sources (JParaCrawl) then stream through the GPU semantic
     filter in chunks.
  3. All survivors stream through a reservoir splitter -> train.jsonl on disk,
     with a uniform valid/test holdout carved off.

Usage:
    uv run python scripts/build_corpus.py --config configs/m2_corpus.yaml
    uv run python scripts/build_corpus.py --config configs/m2_corpus.yaml --download
    uv run python scripts/build_corpus.py --config configs/m2_corpus.yaml \
        --only tatoeba,kftt,jesc --no-wandb        # skip the big JParaCrawl run
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import yaml

from kanjiland.data import tatoeba
from kanjiland.data.corpus_io import read_jsonl, reservoir_split_to_file, write_jsonl
from kanjiland.data.filters import FilterConfig
from kanjiland.data.langid import LangIDConfig, LanguageIdentifier
from kanjiland.data.pipeline import clean_stream
from kanjiland.data.similarity import LaBSEConfig, LaBSEScorer
from kanjiland.data.sources import jesc, jparacrawl, kftt
from kanjiland.data.stats import Funnel, length_stats, render_report

SOURCES = {"tatoeba": tatoeba, "kftt": kftt, "jesc": jesc, "jparacrawl": jparacrawl}


def _source_pairs(name: str, src_cfg: dict):
    """Return the (ja, en) iterator for a configured source."""
    if name == "jparacrawl":
        return jparacrawl.iter_pairs(min_bicleaner=src_cfg.get("min_bicleaner"))
    return SOURCES[name].iter_pairs()


def _labse_filter_file(path: Path, scorer: LaBSEScorer, threshold: float, chunk: int) -> int:
    """Rewrite ``path`` keeping only pairs scoring >= threshold. Returns dropped."""
    tmp = path.with_suffix(".labse.jsonl")
    dropped = 0
    buf: list[tuple[str, str]] = []
    done = 0
    with tmp.open("w", encoding="utf-8") as out:

        def flush() -> None:
            nonlocal dropped, done
            if not buf:
                return
            for (ja, en), s in zip(buf, scorer.score(buf)):
                if s >= threshold:
                    out.write(json.dumps({"ja": ja, "en": en}, ensure_ascii=False) + "\n")
                else:
                    dropped += 1
            done += len(buf)
            print(f"    LaBSE scored {done} (dropped {dropped})", flush=True)
            buf.clear()

        for pair in read_jsonl(path):
            buf.append(pair)
            if len(buf) >= chunk:
                flush()
        flush()
    tmp.replace(path)
    return dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--download", action="store_true", help="download any missing sources first")
    ap.add_argument("--only", type=str, default=None, help="comma-separated subset of sources")
    ap.add_argument("--max-per-source", type=int, default=None, help="cap pairs read per source")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing phase-1 temp files (skip cheap filters), go straight to LaBSE + split",
    )
    ap.add_argument(
        "--keep-tmp",
        action="store_true",
        help="keep _m2_tmp/ after the run (lets you re-run LaBSE at a different threshold)",
    )
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed = cfg.get("seed", 1)
    from kanjiland.train.seed import seed_everything

    seed_everything(seed)

    fcfg = FilterConfig(**cfg["filters"])
    lid_cfg = LangIDConfig(**cfg["langid"])
    labse_cfg = LaBSEConfig(**cfg["labse"])
    identifier = LanguageIdentifier(lid_cfg.model_path) if lid_cfg.enabled else None

    source_cfgs = cfg["sources"]
    if args.only:
        wanted = set(args.only.split(","))
        source_cfgs = [s for s in source_cfgs if s["name"] in wanted]

    out_dir = Path(cfg["output"]["dir"])
    tmp_dir = out_dir / "_m2_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if not args.no_wandb:
        from kanjiland.train.wandb_init import init_run

        run = init_run(args.config)

    seen: set[bytes] = set()
    funnels: list[Funnel] = []
    source_files: list[tuple[str, Path, bool]] = []
    funnels_json = tmp_dir / "funnels.json"

    if args.resume:
        # Reuse the cheap-filter output from a previous run. Funnel counts come
        # from funnels.json if present; otherwise reconstruct what we can (kept
        # = line count, input unknown). Lets us re-run only the expensive LaBSE
        # + split phases (e.g. after switching LaBSE to fp16, or sweeping the
        # threshold) without re-doing the ~28M-pair cheap-filter pass.
        saved = json.loads(funnels_json.read_text()) if funnels_json.exists() else {}
        for src in source_cfgs:
            name = src["name"]
            kept_path = tmp_dir / f"{name}.jsonl"
            if not kept_path.exists():
                raise SystemExit(f"--resume: missing {kept_path}; run phase 1 first")
            if name in saved:
                d = saved[name]
                funnel = Funnel(name, d["input_pairs"], dict(d["dropped"]), d["kept"])
            else:
                k = sum(1 for _ in kept_path.open(encoding="utf-8"))
                funnel = Funnel(name, k, {}, k)
            funnels.append(funnel)
            source_files.append((name, kept_path, bool(src.get("labse"))))
        print(f"resumed: reusing phase-1 temp files for {[f.source for f in funnels]}", flush=True)
    else:
        # --- phase 1: cheap filters, streamed to per-source temp files ------
        for src in source_cfgs:
            name = src["name"]
            if args.download:
                print(f"downloading {name} ...")
                SOURCES[name].download()
            print(f"\n=== {name}: cleaning ===", flush=True)
            pairs = _source_pairs(name, src)
            if args.max_per_source:
                pairs = itertools.islice(pairs, args.max_per_source)
            funnel = Funnel(source=name)
            kept_path = tmp_dir / f"{name}.jsonl"
            n = write_jsonl(
                kept_path,
                clean_stream(
                    pairs,
                    fcfg,
                    funnel,
                    seen,
                    langid_cfg=lid_cfg,
                    identifier=identifier,
                ),
            )
            print(f"  kept {n}/{funnel.input_pairs} after cheap filters", flush=True)
            funnels.append(funnel)
            source_files.append((name, kept_path, bool(src.get("labse"))))
        # Persist funnels so a later --resume run can report accurate drops.
        funnels_json.write_text(
            json.dumps({f.source: f.as_dict() for f in funnels}, ensure_ascii=False, indent=2)
        )

    # --- phase 2: LaBSE semantic filter for flagged sources -----------------
    if labse_cfg.enabled and any(flag for _, _, flag in source_files):
        scorer = LaBSEScorer(labse_cfg)
        for name, path, use_labse in source_files:
            if not use_labse:
                continue
            print(f"\n=== {name}: LaBSE semantic filter (>= {labse_cfg.threshold}) ===", flush=True)
            dropped = _labse_filter_file(path, scorer, labse_cfg.threshold, chunk=50_000)
            funnel = next(f for f in funnels if f.source == name)
            funnel.drop("labse", dropped)
            funnel.kept -= dropped

    # --- phase 3: combine, sample lengths, split, write ---------------------
    ja_lens: list[int] = []
    en_lens: list[int] = []
    rng = random.Random(seed)
    LEN_SAMPLE = 200_000

    def tap(rows):
        """Pass rows through untouched while reservoir-sampling their lengths."""
        for i, (ja, en) in enumerate(rows):
            if len(ja_lens) < LEN_SAMPLE:
                ja_lens.append(len(ja))
                en_lens.append(len(en))
            else:
                j = rng.randint(0, i)
                if j < LEN_SAMPLE:
                    ja_lens[j] = len(ja)
                    en_lens[j] = len(en)
            yield ja, en

    all_rows = itertools.chain.from_iterable(read_jsonl(p) for _, p, _ in source_files)
    split = cfg["split"]
    train_path = out_dir / "train.jsonl"
    print("\n=== splitting ===", flush=True)
    valid, test, total = reservoir_split_to_file(
        tap(all_rows),
        seed,
        split["valid_size"],
        split["test_size"],
        train_path,
    )
    n_valid = write_jsonl(out_dir / "valid.jsonl", valid)
    n_test = write_jsonl(out_dir / "test.jsonl", test)
    n_train = total - n_valid - n_test
    split_counts = {"train": n_train, "valid": n_valid, "test": n_test}
    print(f"train {n_train}  valid {n_valid}  test {n_test}  (total {total})")

    # --- report + cleanup ---------------------------------------------------
    report = render_report(str(args.config), seed, funnels, ja_lens, en_lens, split_counts)
    report_path = Path(cfg["output"]["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nwrote {report_path}\n")
    print(report)

    if args.keep_tmp:
        print(f"kept temp files in {tmp_dir} (--keep-tmp)")
    else:
        for _, p, _ in source_files:
            p.unlink(missing_ok=True)
        funnels_json.unlink(missing_ok=True)
        tmp_dir.rmdir()

    if run is not None:
        for f in funnels:
            run.log({f"funnel/{f.source}/kept": f.kept, f"funnel/{f.source}/input": f.input_pairs})
        js, es = length_stats(ja_lens), length_stats(en_lens)
        run.log(
            {"corpus/total": total, "corpus/ja_len_p95": js["p95"], "corpus/en_len_p95": es["p95"]}
        )
        run.finish()


if __name__ == "__main__":
    main()
