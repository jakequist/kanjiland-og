"""Ablation sweep runner (M5): vary one axis over ≥2 seeds, train + eval each.

    # sequential (one GPU), streams to console:
    uv run python scripts/ablate.py --base configs/m5_ablation_base.yaml \
        --name pos --vary model.pos_encoding=rope,sinusoidal --seeds 1,2

    # PARALLEL across a multi-GPU box (one run per GPU, load-balanced queue):
    uv run python scripts/ablate.py --base ... --vary ... --devices 0,1,2,3,4,5,6,7

    # split across many single-GPU instances (run this per instance):
    uv run python scripts/ablate.py --base ... --vary ... --shard 0/4

For each (variant value × seed) it copies the base config, sets the dotted key
(e.g. ``model.pos_encoding``), stamps ``seed`` + a variant-only ``run_name``
(seed lives in the ckpt subdir), trains via train.py, then evals the final
checkpoint via evaluate.py into ``docs/reports/m5-results.json`` — keyed by
(run, test_set, seed) so seeds auto-aggregate to mean±std.

Parallelism model: each run needs exactly one GPU (a 52M model), so ``--devices``
runs one job per GPU concurrently — the cheap way to finish a sweep in hours on
a rented multi-GPU box. Parallel jobs log to ``logs/`` (console would interleave).
"""

from __future__ import annotations

import argparse
import copy
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

GEN_DIR = Path("configs/_generated")
LOG_DIR = Path("logs")


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def _set_dotted(d: dict, dotted: str, value) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


@dataclass
class Job:
    run_name: str
    seed: int
    train_cmd: list[str]
    eval_cmd: list[str]
    log: Path


def build_jobs(args) -> list[Job]:
    base = yaml.safe_load(Path(args.base).read_text())
    key, raw_values = args.vary.split("=")
    values = raw_values.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []
    for value in values:
        for seed in seeds:
            cfg = copy.deepcopy(base)
            _set_dotted(cfg, key, _coerce(value))
            run_name = f"m5-{args.name}-{value}"  # variant only; seed -> ckpt subdir
            cfg["run_name"] = run_name
            cfg["seed"] = seed
            gen = GEN_DIR / f"{run_name}-s{seed}.yaml"
            gen.write_text(yaml.safe_dump(cfg, sort_keys=False))
            ckpt = Path("checkpoints") / run_name / f"seed{seed}" / "final.pt"
            jobs.append(
                Job(
                    run_name=run_name,
                    seed=seed,
                    train_cmd=[sys.executable, "scripts/train.py", "--config", str(gen)],
                    eval_cmd=[
                        sys.executable,
                        "scripts/evaluate.py",
                        "--config",
                        str(gen),
                        "--checkpoint",
                        str(ckpt),
                        "--test-sets",
                        args.test_sets,
                        "--beam",
                        str(args.beam),
                        "--seed",
                        str(seed),
                        "--results",
                        str(args.results),
                    ],
                    log=LOG_DIR / f"{run_name}-s{seed}.log",
                )
            )
    return jobs


def run_job(job: Job, device: str | None, stream: bool) -> None:
    """Train then eval a single job, optionally pinned to one GPU."""
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = device  # this job sees only its GPU
    if stream:  # sequential mode: inherit stdout so the user watches live
        subprocess.run(job.train_cmd, env=env, check=True)
        subprocess.run(job.eval_cmd, env=env, check=True)
        return
    job.log.parent.mkdir(parents=True, exist_ok=True)
    with job.log.open("w") as f:  # parallel mode: per-run log (console would interleave)
        subprocess.run(job.train_cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        subprocess.run(job.eval_cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--name", required=True, help="ablation name, e.g. 'pos'")
    ap.add_argument("--vary", required=True, help="dotted.key=v1,v2,... (the axis)")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--test-sets", default="kftt-test")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--results", type=Path, default=Path("docs/reports/m5-results.json"))
    ap.add_argument("--devices", default=None, help="comma GPU ids to run across in parallel")
    ap.add_argument("--shard", default=None, help="i/N — run only this instance's slice of jobs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs(args)
    if args.shard:  # split across independent instances
        i, n = (int(x) for x in args.shard.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]

    print(
        f"ablation '{args.name}': {len(jobs)} runs"
        + (f" (shard {args.shard})" if args.shard else "")
    )
    if args.dry_run:
        for j in jobs:
            print(f"  {j.run_name} seed{j.seed}\n    train: {' '.join(j.train_cmd)}")
        print("(dry run — nothing executed)")
        return

    if args.devices:  # parallel: one worker per GPU pulling from a shared queue
        devices = args.devices.split(",")
        q: queue.Queue[Job] = queue.Queue()
        for j in jobs:
            q.put(j)

        def worker(dev: str) -> None:
            while True:
                try:
                    job = q.get_nowait()
                except queue.Empty:
                    return
                print(f"[gpu{dev}] start {job.run_name} seed{job.seed}", flush=True)
                try:
                    run_job(job, device=dev, stream=False)
                    print(
                        f"[gpu{dev}] done  {job.run_name} seed{job.seed} -> {job.log}", flush=True
                    )
                except subprocess.CalledProcessError:
                    print(
                        f"[gpu{dev}] FAILED {job.run_name} seed{job.seed} (see {job.log})",
                        flush=True,
                    )

        threads = [threading.Thread(target=worker, args=(d,)) for d in devices]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:  # sequential, one GPU, live output
        for job in jobs:
            print(f"\n=== {job.run_name} seed {job.seed} ===", flush=True)
            run_job(job, device=None, stream=True)

    print("\nablation complete.")


if __name__ == "__main__":
    main()
