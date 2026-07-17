#!/usr/bin/env bash
# Provision ONE vast.ai GPU box and onboard it for kanjiland — search a good
# offer, rent it, wait for it to boot, then run cloud_bootstrap.sh on it. Run it
# once per box (pass a distinct SHARD to split a sweep across several).
#
# Prereqs (one-time, see docs/CLOUD.md):
#   - vastai CLI installed + authed (VAST_API_KEY)
#   - an SSH key registered with vast whose private half is $SSH_KEY locally
#   - CODE_URL / DATA_URL reachable from the box (public S3 objects are simplest;
#     presigned URLs from *temporary* STS creds expire in minutes — see gotchas)
#
# Usage:
#   CODE_URL=... DATA_URL=... [WANDB_API_KEY=...] [SHARD=0/2] \
#     bash scripts/vast_up.sh [label]
#
# Why bash + retries and not a one-shot `vastai create`: cloud hosts routinely
# fail to start the container ("failed to create task"). This script treats that
# as normal — it walks down the cheapest reliable offers until one actually boots.
set -euo pipefail

: "${CODE_URL:?need CODE_URL}"
: "${DATA_URL:?need DATA_URL}"
LABEL="${1:-kanjiland}"
NUM_GPUS="${NUM_GPUS:-8}"
GPU_NAME="${GPU_NAME:-RTX_4090}"
DISK="${DISK:-80}"                                    # GB; ~10GB data + venv + ckpts
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vast_kanjiland}"
MIN_RELIABILITY="${MIN_RELIABILITY:-0.98}"           # host uptime score
MIN_INET="${MIN_INET:-200}"                          # Mbit/s down — we pull ~10GB
MAX_CANDIDATES="${MAX_CANDIDATES:-8}"                 # offers to try before giving up
BOOT_TIMEOUT="${BOOT_TIMEOUT:-360}"                   # s to wait for 'running'
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== searching offers ($NUM_GPUS× $GPU_NAME, reliability>$MIN_RELIABILITY, inet>$MIN_INET) =="
# cuda_max_good is the max CUDA the host driver supports; torch-cu126 needs a
# driver new enough for 12.4+ (minor-version compat), else torch.cuda is False.
offers=$(vastai search offers \
  "reliability > $MIN_RELIABILITY num_gpus=$NUM_GPUS gpu_name=$GPU_NAME cuda_max_good >= 12.4 inet_down > $MIN_INET rentable=True" \
  -o dph_total --raw)
mapfile -t CAND < <(printf '%s' "$offers" | python3 -c '
import sys, json
for o in json.load(sys.stdin)[: int("'"$MAX_CANDIDATES"'")]:
    print(o["id"], round(o["dph_total"], 3))
')
[ "${#CAND[@]}" -gt 0 ] || { echo "no offers matched — relax MIN_* filters"; exit 1; }
echo "candidates (id dph): ${CAND[*]}"

inst=""; host=""; port=""
for line in "${CAND[@]}"; do
  oid="${line%% *}"; dph="${line##* }"
  echo "== renting offer $oid (\$$dph/hr) =="
  created=$(vastai create instance "$oid" --image "$IMAGE" --disk "$DISK" --ssh \
            --label "$LABEL" --raw) || { echo "  create failed, next"; continue; }
  iid=$(printf '%s' "$created" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("new_contract",""))')
  [ -n "$iid" ] || { echo "  no instance id, next"; continue; }
  echo "  instance $iid — waiting up to ${BOOT_TIMEOUT}s to boot"

  ok=0; waited=0
  while [ "$waited" -lt "$BOOT_TIMEOUT" ]; do
    sleep 15; waited=$((waited + 15))
    row=$(vastai show instance "$iid" --raw 2>/dev/null || true)
    read -r status host port < <(printf '%s' "$row" | python3 -c '
import sys, json
try: d = json.load(sys.stdin)
except Exception: print(" "); raise SystemExit
print(d.get("actual_status",""), d.get("ssh_host",""), d.get("ssh_port",""))
')
    echo "    [$waited s] status=$status"
    if [ "$status" = "running" ] && [ -n "$host" ] && [ -n "$port" ]; then ok=1; break; fi
  done
  if [ "$ok" != 1 ]; then
    echo "  did not boot in time — destroying $iid, trying next offer"
    echo y | vastai destroy instance "$iid" || true
    continue
  fi
  inst="$iid"; break
done
[ -n "$inst" ] || { echo "FATAL: no candidate offer booted"; exit 1; }

echo "== instance $inst up at $host:$port — waiting for sshd =="
for i in $(seq 1 20); do
  if ssh -i "$SSH_KEY" $SSH_OPTS -p "$port" "root@$host" true 2>/dev/null; then break; fi
  sleep 6
done

echo "== onboarding (piping cloud_bootstrap.sh) =="
# Pass config as env on the remote side; pipe the script over stdin so nothing
# needs to be pre-copied. RUN_SWEEP/SEEDS/SHARD/WANDB flow straight through.
ssh -i "$SSH_KEY" $SSH_OPTS -p "$port" "root@$host" \
  "CODE_URL='$CODE_URL' DATA_URL='$DATA_URL' \
   WANDB_API_KEY='${WANDB_API_KEY:-}' WANDB_MODE='${WANDB_MODE:-offline}' \
   DEVICES='${DEVICES:-$(seq -s, 0 $((NUM_GPUS-1)))}' SEEDS='${SEEDS:-1,2,3}' \
   SHARD='${SHARD:-}' RUN_SWEEP='${RUN_SWEEP:-1}' bash -s" < "$HERE/cloud_bootstrap.sh"

echo ""
echo "== READY =="
echo "  instance : $inst  (label=$LABEL)"
echo "  ssh      : ssh -i $SSH_KEY $SSH_OPTS -p $port root@$host"
echo "  monitor  : ssh ... 'tail -f ~/kanjiland/sweep.log'"
echo "  destroy  : bash scripts/vast_down.sh $LABEL   # or: vastai destroy instance $inst"
