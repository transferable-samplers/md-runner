#!/bin/bash
#SBATCH -J archive_copy
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=4G
#SBATCH -t 48:00:00
#SBATCH --partition=unkillable-cpu
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --open-mode=append
#SBATCH --get-user-env
#SBATCH --array=0-2

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

SCRATCH=/network/scratch/t/tanc
ARCHIVE=/network/archive/t/tanc

# Source dirs (trailing slash on renamed ones so rsync copies contents into dst)
SRCS=(
    "$SCRATCH/md-runner-eval"
    "$SCRATCH/md-runner-temperature"
    "$SCRATCH/md-log-8AA"
    "$SCRATCH/ablation_models"
)

# Destination dirs (including new name where applicable)
DSTS=(
    "$ARCHIVE/ManyPeptidesMDBackup/md-runner-eval/"
    "$ARCHIVE/ManyPeptidesMDBackup/md-runner-temperature/"
    "$ARCHIVE/ManyPeptidesMDBackup/md-log-8AA/"
    "$ARCHIVE/amortized_sampling_results/"
)

SRC="${SRCS[$SLURM_ARRAY_TASK_ID]}"
DST="${DSTS[$SLURM_ARRAY_TASK_ID]}"

echo "Copying: $SRC -> $DST"
mkdir -p "$DST"
rsync -rtzv --size-only "$SRC" "$DST"
echo "Done task $SLURM_ARRAY_TASK_ID"
