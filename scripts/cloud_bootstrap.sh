#!/usr/bin/env bash
# Node-side bootstrap for a rented GPU box (vast.ai etc). Turns a bare GPU
# instance into a ready-to-train kanjiland box, then optionally launches the M5
# ablation sweep across all local GPUs.
#
# This is the codified, battle-tested onboarding recipe. Every step here exists
# because a fresh cloud box tripped on it at least once (see docs/CLOUD.md
# "Onboarding gotchas") — do not "simplify" a step away without re-checking the
# gotcha it guards.
#
# Usage (env-configured; run FROM the box, or pipe over ssh via vast_up.sh):
#   CODE_URL=... DATA_URL=... [WANDB_API_KEY=...] \
#   [DEVICES=0,1,2,3,4,5,6,7] [SEEDS=1,2,3] [SHARD=0/2] [RUN_SWEEP=1] \
#     bash cloud_bootstrap.sh
set -euo pipefail

: "${CODE_URL:?need CODE_URL (public or presigned tarball of the repo)}"
: "${DATA_URL:?need DATA_URL (public or presigned tarball of data/processed)}"
DEVICES="${DEVICES:-0}"
SEEDS="${SEEDS:-1,2,3}"
SHARD="${SHARD:-}"                # optional i/N for multi-box splits
RUN_SWEEP="${RUN_SWEEP:-1}"       # 0 = just onboard, don't launch the sweep
WORKDIR="${WORKDIR:-$HOME/kanjiland}"
PYVER="${PYVER:-3.11}"            # 3.11 has manylinux wheels for the whole stack
export DEBIAN_FRONTEND=noninteractive

echo "== [1/6] build toolchain =="
# torch.compile's inductor backend generates C++ (not just Triton) and several
# sdist-only deps compile from source; stock GPU images ship no compiler, so
# `numpy`/`torch.compile` both fail without this. gcc up front > cryptic build
# errors 20 minutes into a sync. curl+git in case the base image is minimal.
if ! command -v gcc >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq build-essential curl git
fi

echo "== [2/6] uv =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "== [3/6] pull code + data =="
mkdir -p "$WORKDIR"; cd "$WORKDIR"
curl -fsSL "$CODE_URL" | tar xz              # -> src/, scripts/, configs/, ...
curl -fsSL "$DATA_URL" | tar xz              # -> data/processed/{tok,tok8k,tok32k,...}

echo "== [4/6] python $PYVER + deps =="
# The repo pins .python-version=3.13 (fine locally where a compiler exists), but
# 3.13 has wheel GAPS for pinned deps (e.g. numpy 1.26.4) -> source build -> slow
# or broken on a bare box. Force 3.11, which has wheels for torch-cu126, numpy,
# and the COMET eval chain. uv would otherwise silently fall back to its managed
# 3.13 and rebuild from source on every `uv run`.
echo "$PYVER" > .python-version
uv python install "$PYVER"
# `eval` extra only: `data` (fasttext-wheel/sentence-transformers) is the offline
# corpus pipeline — already done — and fasttext needs a C++ build we don't want
# to wait on here. Training + the M4 eval harness need only core + eval.
uv sync --python "$PYVER" --extra eval

echo "== [5/6] GPU check =="
# Fail loud NOW if CUDA isn't visible — better than discovering it 8 runs deep.
uv run python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False — driver/torch mismatch")
print(f"OK: {torch.cuda.device_count()} GPU(s), torch {torch.__version__}")
PY

echo "== [6/6] sweep =="
if [ "$RUN_SWEEP" != "1" ]; then
  echo "RUN_SWEEP=$RUN_SWEEP — onboarding only, not launching. Box is READY."
  exit 0
fi
# WANDB offline by default: no key needs to live on the box, runs still record
# and can be `wandb sync`'d later. Pass WANDB_API_KEY + WANDB_MODE=online to log
# live. setsid + </dev/null fully detaches so the launching ssh can return.
export WANDB_MODE="${WANDB_MODE:-offline}"
TQDM_DISABLE=1 setsid uv run python scripts/ablate.py \
  --base configs/m5_ablation_base.yaml --name all \
  --preset configs/m5_all_variants.yaml --seeds "$SEEDS" \
  --test-sets kftt-test --devices "$DEVICES" ${SHARD:+--shard "$SHARD"} \
  > sweep.log 2>&1 < /dev/null &
echo "sweep launched (devices=$DEVICES seeds=$SEEDS shard=${SHARD:-none}) -> $WORKDIR/sweep.log"
echo "results -> docs/reports/m5-results.json ; per-run logs -> logs/"
