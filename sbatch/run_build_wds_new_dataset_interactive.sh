#!/bin/bash
# Interactive version. Example usage:
#   salloc --mem=128G -c 8 -t 96:00:00 --tmp=6000G --partition=main-cpu
#   bash sbatch/run_build_wds_new_dataset_interactive.sh
#
# Peak local disk usage (with the LOCAL_SRC cleanup below): ~4.8TiB, hit
# during the convert step (LOCAL_SRC ~3.1TiB still present + LOCAL_CONV
# filling up to ~1.68TiB); build step only needs LOCAL_CONV (~1.68TiB) +
# LOCAL_OUT (~2.14TiB) = ~3.82TiB. --tmp=6000G (~5.6TiB) leaves headroom.

set -euo pipefail

: "${SLURM_TMPDIR:=/tmp/$USER-wds-new-dataset}"
mkdir -p "$SLURM_TMPDIR"

echo "Node: ${HOSTNAME:-unknown}"
echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "TMPDIR: $SLURM_TMPDIR"

ARCHIVE=/network/archive/t/tanc
SRC="$ARCHIVE/new-dataset"
DST="$ARCHIVE/WDS_NEW_DATASET"

LOCAL_SRC="$SLURM_TMPDIR/new-dataset-DS"
LOCAL_CONV="$SLURM_TMPDIR/new-dataset-CONV"
LOCAL_OUT="$SLURM_TMPDIR/WDS_NEW_DATASET"

mkdir -p "$LOCAL_SRC" "$LOCAL_CONV" "$LOCAL_OUT"

echo "=== Syncing $SRC -> $LOCAL_SRC ==="
time rsync -a --size-only "$SRC/" "$LOCAL_SRC/"

echo "=== Generating sequence list ==="
SEQ_FILE="$SLURM_TMPDIR/sequences.txt"
ls "$LOCAL_SRC" | grep -E '\.trajectories\.npz$' | sed 's/\.trajectories\.npz$//' | sort > "$SEQ_FILE"
echo "Sequences: $(wc -l < "$SEQ_FILE")"

echo "=== Converting trajectories to pre-shuffled flat .npy (every other frame) ==="
time python helpers/convert_trajectories.py \
    --src-dir "$LOCAL_SRC" \
    --dst-dir "$LOCAL_CONV" \
    --sequence-file "$SEQ_FILE" \
    --trajectory-file-suffix ".trajectories" \
    --seed 42 \
    --workers 8 \
    --stride 2

# LOCAL_SRC (raw, compressed, all 5000 frames -- ~3.1TiB) is no longer needed
# once LOCAL_CONV exists; delete it now so it isn't sitting alongside
# LOCAL_CONV (~1.68TiB) and LOCAL_OUT (~2.14TiB) during the build step below
# (peak would otherwise be ~6.9TiB, blowing past --tmp). Source data is safe
# in $SRC on archive.
echo "=== Freeing $LOCAL_SRC ==="
rm -rf "$LOCAL_SRC"

# 36,000 sequences x 5000 frames/replica (5001 iterations, frame 0 dropped),
# then --stride 2 in convert_trajectories.py keeps every other frame in time
# order -> 2500 frames/replica -> 1 sample/replica/tar with --num-tarfiles
# 2500 lands on exactly one tar per (downsampled) frame index, same
# convention as run_build_wds_oligo_interactive.sh's auto-determined
# num_tarfiles=F//K.
echo "=== Building REMD webdataset (1 sample/replica/tar; 2500 tars; mixed temps) ==="
time python helpers/build_webdataset_remd.py \
    --sequence-file "$SEQ_FILE" \
    --converted-dir "$LOCAL_CONV" \
    --output-dir "$LOCAL_OUT" \
    --samples-per-replica-per-tar 1 \
    --num-tarfiles 2500 \
    --seed 42 \
    --verbose

echo "=== Copying $LOCAL_OUT -> $DST ==="
mkdir -p "$DST"
time cp -r "$LOCAL_OUT/." "$DST/"

echo "=== Done ==="
