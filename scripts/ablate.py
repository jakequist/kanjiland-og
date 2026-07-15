"""Ablation sweep runner (M5): vary one axis over ≥2 seeds, train + eval each.

    uv run python scripts/ablate.py --base configs/m5_ablation_base.yaml \
        --name pos --vary model.pos_encoding=rope,sinusoidal --seeds 1,2

For each (variant value × seed) it:
  1. copies the base config, sets the dotted key (e.g. ``model.pos_encoding``),
     stamps ``seed`` and a variant-only ``run_name`` (seed lives in the ckpt
     subdir), and writes it to ``configs/_generated/``;
  2. trains via ``scripts/train.py``;
  3. evaluates the final checkpoint via ``scripts/evaluate.py`` into
     ``docs/reports/m5-results.json`` — keyed by (run, test_set, seed), so the
     two seeds of a variant auto-aggregate to mean±std (seed-variance protocol).

Runs are sequential (single GPU). Use ``--dry-run`` to print the plan first.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

GEN_DIR = Path("configs/_generated")


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
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--name", required=True, help="ablation name, e.g. 'pos'")
    ap.add_argument("--vary", required=True, help="dotted.key=v1,v2,... (the axis)")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--test-sets", default="kftt-test")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--results", type=Path, default=Path("docs/reports/m5-results.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = yaml.safe_load(args.base.read_text())
    key, values = args.vary.split("=")
    values = values.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    plan = [(v, s) for v in values for s in seeds]
    print(f"ablation '{args.name}': {key} in {values} x seeds {seeds} = {len(plan)} runs\n")

    for value, seed in plan:
        cfg = copy.deepcopy(base)
        _set_dotted(cfg, key, _coerce(value))
        run_name = f"m5-{args.name}-{value}"  # variant only; seed -> ckpt subdir
        cfg["run_name"] = run_name
        cfg["seed"] = seed
        gen = GEN_DIR / f"{run_name}-s{seed}.yaml"
        gen.write_text(yaml.safe_dump(cfg, sort_keys=False))
        ckpt = Path("checkpoints") / run_name / f"seed{seed}" / "final.pt"

        train_cmd = [sys.executable, "scripts/train.py", "--config", str(gen)]
        eval_cmd = [
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
        ]
        print(f"=== {run_name} seed {seed} ===")
        if args.dry_run:
            print("  train:", " ".join(train_cmd))
            print("  eval: ", " ".join(eval_cmd))
            continue
        subprocess.run(train_cmd, check=True)
        subprocess.run(eval_cmd, check=True)

    print("\nablation complete." if not args.dry_run else "\n(dry run — nothing executed)")


if __name__ == "__main__":
    main()
