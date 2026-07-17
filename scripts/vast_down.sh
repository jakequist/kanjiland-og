#!/usr/bin/env bash
# Destroy every vast.ai instance whose label matches (default: kanjiland*).
# A rented box bills by the second until destroyed, so this is the "stop paying"
# button — run it the moment a sweep's results are pulled.
#
# Usage:
#   bash scripts/vast_down.sh [label-prefix]     # default prefix: kanjiland
#   DRY_RUN=1 bash scripts/vast_down.sh          # list what would be destroyed
set -euo pipefail

PREFIX="${1:-kanjiland}"
rows=$(vastai show instances --raw)
mapfile -t IDS < <(printf '%s' "$rows" | python3 -c '
import sys, json
pref = "'"$PREFIX"'"
for d in json.load(sys.stdin):
    if str(d.get("label") or "").startswith(pref):
        print(d["id"], d.get("actual_status",""), str(d.get("label") or ""))
')

[ "${#IDS[@]}" -gt 0 ] || { echo "no instances with label prefix '$PREFIX'"; exit 0; }

echo "instances matching '$PREFIX*':"
printf '  %s\n' "${IDS[@]}"
if [ "${DRY_RUN:-0}" = 1 ]; then echo "(dry run — nothing destroyed)"; exit 0; fi

for line in "${IDS[@]}"; do
  iid="${line%% *}"
  echo "destroying $iid"
  echo y | vastai destroy instance "$iid" || echo "  (destroy failed for $iid — check manually)"
done
echo "done — verify with: vastai show instances"
