#!/usr/bin/env bash
# Node-side bootstrap for a rented GPU box (vast.ai etc). Pulls code + data from
# presigned S3 URLs, syncs the env, and runs the M5 ablation sweep across all
# local GPUs. No cloud creds live on the node — downloads are time-limited
# signed URLs.
#
#   CODE_URL=... DATA_URL=... WANDB_API_KEY=... DEVICES=0,1,2,3,4,5,6,7 \
#     bash cloud_bootstrap.sh
set -euo pipefail

: "${CODE_URL:?need presigned CODE_URL}"
: "${DATA_URL:?need presigned DATA_URL}"
DEVICES="${DEVICES:-0}"
SEEDS="${SEEDS:-1,2}"
SHARD="${SHARD:-}"                       # optional i/N for multi-box splits

cd "${WORKDIR:-$HOME/kanjiland}" 2>/dev/null || { mkdir -p "$HOME/kanjiland"; cd "$HOME/kanjiland"; }

echo "== pull code =="; curl -fsSL "$CODE_URL" | tar xz
echo "== pull data =="; curl -fsSL "$DATA_URL" | tar xz     # -> data/processed/...

echo "== uv + deps =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Only core + eval: the `data` extra (fasttext-wheel/sentence-transformers) is
# for the offline corpus pipeline, already done, and fasttext needs a C++
# compiler that GPU images usually lack.
uv sync --extra eval

echo "== GPU check =="
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), 'x', torch.cuda.device_count())"

echo "== sweep (devices=$DEVICES seeds=$SEEDS shard=${SHARD:-none}) =="
TQDM_DISABLE=1 uv run python scripts/ablate.py \
  --base configs/m5_ablation_base.yaml --name all \
  --preset configs/m5_all_variants.yaml --seeds "$SEEDS" \
  --test-sets kftt-test --devices "$DEVICES" ${SHARD:+--shard "$SHARD"}

echo "== DONE — results in docs/reports/m5-results.json, logs in logs/ =="
