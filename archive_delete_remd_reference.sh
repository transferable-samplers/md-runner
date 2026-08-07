#!/bin/bash
#SBATCH -J archive_delete_remd_reference
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=2G
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --open-mode=append
#SBATCH --get-user-env
#SBATCH --array=0-0

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

SCRATCH=/network/scratch/t/tanc
ARCHIVE=/network/archive/t/tanc

# alanine, many, many-300K already deleted (job 10283362, all COMPLETED 0:0).
# xl's checksum verification (job 10282579 task 3) came back CLEAN, so it's
# now cleared for deletion too.
DIRS=(
    md-runner-remd-reference-xl
)

NAME="${DIRS[$SLURM_ARRAY_TASK_ID]}"
SRC="$SCRATCH/$NAME"
DST="$ARCHIVE/$NAME"

if [ ! -d "$DST" ]; then
    echo "REFUSING: archive copy missing for $NAME ($DST not found)"
    exit 1
fi

echo "Deleting: $SRC (verified clean against $DST)"
rm -rf "$SRC"
echo "Done task $SLURM_ARRAY_TASK_ID"
