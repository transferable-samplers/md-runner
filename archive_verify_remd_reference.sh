#!/bin/bash
#SBATCH -J archive_verify_remd_reference
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=4G
#SBATCH -t 48:00:00
#SBATCH --partition=unkillable-cpu
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --open-mode=append
#SBATCH --get-user-env
#SBATCH --array=0-3

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

SCRATCH=/network/scratch/t/tanc
ARCHIVE=/network/archive/t/tanc

SRCS=(
    "$SCRATCH/md-runner-remd-reference-alanine/"
    "$SCRATCH/md-runner-remd-reference-many/"
    "$SCRATCH/md-runner-remd-reference-many-300K/"
    "$SCRATCH/md-runner-remd-reference-xl/"
)

DSTS=(
    "$ARCHIVE/md-runner-remd-reference-alanine/"
    "$ARCHIVE/md-runner-remd-reference-many/"
    "$ARCHIVE/md-runner-remd-reference-many-300K/"
    "$ARCHIVE/md-runner-remd-reference-xl/"
)

SRC="${SRCS[$SLURM_ARRAY_TASK_ID]}"
DST="${DSTS[$SLURM_ARRAY_TASK_ID]}"
LOG="watch_folder/verify_diff_${SLURM_ARRAY_TASK_ID}.log"

echo "Checksum-verifying: $SRC vs $DST"
# --dry-run + --checksum: reads and hashes every file on both sides, reports
# any that differ (by content, not just size/mtime), copies nothing.
rsync -rtc --dry-run -i "$SRC" "$DST" > "$LOG" 2>&1

if [ -s "$LOG" ]; then
    echo "MISMATCHES FOUND for task $SLURM_ARRAY_TASK_ID, see $LOG"
    cat "$LOG"
else
    echo "CLEAN: no differences for task $SLURM_ARRAY_TASK_ID ($SRC vs $DST)"
fi
echo "Done task $SLURM_ARRAY_TASK_ID"
