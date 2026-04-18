#!/bin/bash
#SBATCH -J build_wds_assorted
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --partition=main-cpu
#SBATCH --open-mode=append

set -euo pipefail

echo "Node: $HOSTNAME"
echo "Job: $SLURM_JOB_ID"
echo "TMPDIR: $SLURM_TMPDIR"

ARCHIVE=/network/archive/t/tanc
SRC="$ARCHIVE/ASSORTED"
DST="$ARCHIVE/WDS_ASSORTED"

LOCAL_SRC="$SLURM_TMPDIR/ASSORTED"
LOCAL_OUT="$SLURM_TMPDIR/WDS_ASSORTED"

mkdir -p "$LOCAL_SRC" "$LOCAL_OUT"

echo "=== Copying $SRC -> $LOCAL_SRC ==="
time cp -r "$SRC/." "$LOCAL_SRC/"

echo "=== Generating sequence list ==="
SEQ_FILE="$SLURM_TMPDIR/sequences.txt"
ls "$LOCAL_SRC" | grep -E '\.trajectories\.npz$' | sed 's/\.trajectories\.npz$//' | sort > "$SEQ_FILE"
echo "Sequences: $(wc -l < "$SEQ_FILE")"

echo "=== Building webdataset (250 tars x 10 samples/seq = 1/5 of 12500 frames) ==="
time python helpers/build_webdataset_optimized.py \
    --sequence-file "$SEQ_FILE" \
    --downsampled-dir "$LOCAL_SRC" \
    --output-dir "$LOCAL_OUT" \
    --trajectory-file-suffix ".trajectories" \
    --num-tarfiles 250 \
    --samples-per-seq-per-tar 10 \
    --batch-size 32 \
    --max-workers-load 8 \
    --max-workers-tar 3

echo "=== Copying $LOCAL_OUT -> $DST ==="
mkdir -p "$DST"
time cp -r "$LOCAL_OUT/." "$DST/"

echo "=== Done ==="
