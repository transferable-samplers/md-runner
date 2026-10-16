#!/bin/bash
#SBATCH -J demux_oligopeptides_extra
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --partition=long-cpu,main-cpu,unkillable-cpu
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -t 8:00:00
#SBATCH --array=0-79
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --get-user-env

set -euo pipefail

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

# Demux all 1875 oligopeptides_extra REMD runs (all temperatures) into
# trajectories.npz + swap_rates.txt next to each run's remd.nc.
NUM_TASKS=80

cd "$SLURM_SUBMIT_DIR/helpers"
python demux_oligopeptides_extra_shard.py "$SLURM_ARRAY_TASK_ID" "$NUM_TASKS"
