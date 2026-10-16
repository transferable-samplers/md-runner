#!/bin/bash
#SBATCH -J demux_remd_1000k
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --partition=long-cpu,main-cpu
#SBATCH --mem=32G
#SBATCH -c 2
#SBATCH -t 8:00:00
#SBATCH --array=0-33
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --get-user-env

set -euo pipefail

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

# Demuxes the 1000K hot-equilibration REMD runs (4aa+8aa share the
# "many-1000K" scratch root, xl uses "xl-1000K") at ALL temperatures
# (--no-lowest-temp-only), unlike run_demux_remd_array.sh which only keeps
# the lowest temperature. --array size (0-31) == total unique sequences
# across both 1000K roots (20 many-1000K + 12 xl-1000K); update if the
# sequence lists change.
OUT_DIR=/network/scratch/t/tanc/remd-demux-v5

# Build the manifest once (idempotent); each task reads its own line.
# Line N (0-based) == SLURM_ARRAY_TASK_ID -> "<subset>\t<sequence>".
MANIFEST="$OUT_DIR/_seq_manifest.tsv"
mkdir -p "$OUT_DIR"
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ] || [ ! -s "$MANIFEST" ]; then
  python helpers/list_remd_sequences_1000k.py > "$MANIFEST"
fi
# Wait for the manifest to exist (covers tasks that start before task 0 writes it).
for _ in $(seq 1 60); do [ -s "$MANIFEST" ] && break; sleep 5; done

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
SUBSET=$(echo "$LINE" | cut -f1)
SEQUENCE=$(echo "$LINE" | cut -f2)
echo "Processing subset=$SUBSET sequence=$SEQUENCE"

python helpers/demux_all_remd_1000k.py \
  --out-dir "$OUT_DIR" \
  --subset "$SUBSET" \
  --sequence "$SEQUENCE" \
  --exact-sequence \
  --no-lowest-temp-only

echo "Done: $SUBSET/$SEQUENCE"
