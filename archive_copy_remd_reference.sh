#!/bin/bash
#SBATCH -J archive_copy_remd_reference
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

# Source dirs (trailing slash so rsync copies contents into dst dir of the same name)
SRCS=(
    "$SCRATCH/md-runner-remd-reference-alanine/"
    "$SCRATCH/md-runner-remd-reference-many/"
    "$SCRATCH/md-runner-remd-reference-many-300K/"
    "$SCRATCH/md-runner-remd-reference-xl/"
)

# Destination dirs (top-level in archive, same names)
DSTS=(
    "$ARCHIVE/md-runner-remd-reference-alanine/"
    "$ARCHIVE/md-runner-remd-reference-many/"
    "$ARCHIVE/md-runner-remd-reference-many-300K/"
    "$ARCHIVE/md-runner-remd-reference-xl/"
)

SRC="${SRCS[$SLURM_ARRAY_TASK_ID]}"
DST="${DSTS[$SLURM_ARRAY_TASK_ID]}"

echo "Copying: $SRC -> $DST"
mkdir -p "$DST"
rsync -rtzv --size-only "$SRC" "$DST"
echo "Done task $SLURM_ARRAY_TASK_ID"
