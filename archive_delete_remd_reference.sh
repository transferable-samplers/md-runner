#!/bin/bash
#SBATCH -J archive_delete_remd_reference
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=2G
#SBATCH -t 4:00:00
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --open-mode=append
#SBATCH --get-user-env
#SBATCH --array=0-2

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

SCRATCH=/network/scratch/t/tanc
ARCHIVE=/network/archive/t/tanc

# Only dirs whose checksum verification (job 10282579) came back CLEAN.
# md-runner-remd-reference-xl is intentionally excluded — its verification
# was still running when this was submitted.
DIRS=(
    md-runner-remd-reference-alanine
    md-runner-remd-reference-many
    md-runner-remd-reference-many-300K
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
