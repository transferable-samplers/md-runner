#!/bin/bash
#SBATCH -J build_wds_oligo
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --mem=64G
#SBATCH -t 96:00:00
#SBATCH -c 8
#SBATCH --partition=main-cpu
#SBATCH --tmp=5000G
#SBATCH --open-mode=append

set -euo pipefail

echo "Node: $HOSTNAME"
echo "Job: $SLURM_JOB_ID"
echo "TMPDIR: $SLURM_TMPDIR"

ARCHIVE=/network/archive/t/tanc
SRC="$ARCHIVE/OLIGO"
DST="$ARCHIVE/WDS_OLIGO"

LOCAL_SRC="$SLURM_TMPDIR/OLIGO"
LOCAL_OUT="$SLURM_TMPDIR/WDS_OLIGO"

mkdir -p "$LOCAL_SRC" "$LOCAL_OUT"

echo "=== Copying $SRC -> $LOCAL_SRC ==="
time cp -r "$SRC/." "$LOCAL_SRC/"

echo "=== Generating sequence list ==="
SEQ_FILE="$SLURM_TMPDIR/sequences.txt"
ls "$LOCAL_SRC" | grep -E '\.trajectories\.npz$' | sed 's/\.trajectories\.npz$//' | sort > "$SEQ_FILE"
echo "Sequences: $(wc -l < "$SEQ_FILE")"

echo "=== Building REMD webdataset (1 sample/replica/tar; auto 1250 tars; mixed temps) ==="
time python helpers/build_webdataset_remd.py \
    --sequence-file "$SEQ_FILE" \
    --downsampled-dir "$LOCAL_SRC" \
    --output-dir "$LOCAL_OUT" \
    --trajectory-file-suffix ".trajectories" \
    --samples-per-replica-per-tar 1 \
    --batch-size 64 \
    --max-workers-load 8 \
    --max-workers-tar 6

echo "=== Copying $LOCAL_OUT -> $DST ==="
mkdir -p "$DST"
time cp -r "$LOCAL_OUT/." "$DST/"

echo "=== Done ==="
