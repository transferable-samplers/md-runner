#!/bin/bash
# Interactive version. Example usage:
#   salloc --mem=128G -c 8 -t 96:00:00 --tmp=5000G --partition=main-cpu
#   bash sbatch/run_build_wds_oligo_interactive.sh

set -euo pipefail

: "${SLURM_TMPDIR:=/tmp/$USER-wds-oligo}"
mkdir -p "$SLURM_TMPDIR"

echo "Node: ${HOSTNAME:-unknown}"
echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "TMPDIR: $SLURM_TMPDIR"

ARCHIVE=/network/archive/t/tanc
SRC="$ARCHIVE/OLIGO"
DST="$ARCHIVE/WDS_OLIGO"

LOCAL_SRC="$SLURM_TMPDIR/OLIGO-DS"
LOCAL_CONV="$SLURM_TMPDIR/OLIGO-CONV"
LOCAL_OUT="$SLURM_TMPDIR/WDS_OLIGO"

mkdir -p "$LOCAL_SRC" "$LOCAL_CONV" "$LOCAL_OUT"

echo "=== Syncing $SRC -> $LOCAL_SRC ==="
time rsync -a --size-only "$SRC/" "$LOCAL_SRC/"

echo "=== Generating sequence list ==="
SEQ_FILE="$SLURM_TMPDIR/sequences.txt"
ls "$LOCAL_SRC" | grep -E '\.trajectories\.npz$' | sed 's/\.trajectories\.npz$//' | sort > "$SEQ_FILE"
echo "Sequences: $(wc -l < "$SEQ_FILE")"

echo "=== Converting trajectories to pre-shuffled flat .npy ==="
time python helpers/convert_trajectories.py \
    --src-dir "$LOCAL_SRC" \
    --dst-dir "$LOCAL_CONV" \
    --sequence-file "$SEQ_FILE" \
    --trajectory-file-suffix ".trajectories" \
    --seed 42 \
    --workers 8

echo "=== Building REMD webdataset (1 sample/replica/tar; auto 1250 tars; mixed temps) ==="
time python helpers/build_webdataset_remd.py \
    --sequence-file "$SEQ_FILE" \
    --converted-dir "$LOCAL_CONV" \
    --output-dir "$LOCAL_OUT" \
    --samples-per-replica-per-tar 1 \
    --seed 42 \
    --verbose

echo "=== Copying $LOCAL_OUT -> $DST ==="
mkdir -p "$DST"
time cp -r "$LOCAL_OUT/." "$DST/"

echo "=== Done ==="
