#!/usr/bin/env bash
# check_his_charmm.sh — count HSE (good) vs other histidine names (bad)
# across all .pdb files under ROOT, and print the ratio.
#
# Usage: ./check_his_charmm.sh [ROOT]
set -euo pipefail

ROOT="${1:-.}"

GOOD_PATTERN='^(ATOM  |HETATM).{11}HSE '
BAD_PATTERN='^(ATOM  |HETATM).{11}(HSD|HSP|HIS|HID|HIE|HIP) '

good=0
bad=0
while IFS= read -r -d '' f; do
  g=$(grep -cE "$GOOD_PATTERN" "$f" || true)
  b=$(grep -cE "$BAD_PATTERN" "$f" || true)
  good=$((good + g))
  bad=$((bad + b))
done < <(find "$ROOT" -type f -name '*.pdb' -print0)

echo "$good / $bad"
