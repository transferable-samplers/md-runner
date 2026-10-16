#!/bin/bash
#SBATCH -J demux_new_dataset
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --partition=long-cpu,main-cpu,unkillable-cpu
#SBATCH --mem=8G
#SBATCH -c 2
#SBATCH -t 12:00:00
#SBATCH --array=0-499%50
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --get-user-env

set -euo pipefail

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

# Demux all 36,000 canonical UniRef50Chunks/IDRomeChunks REMD runs (all
# temperatures) into a single flat directory: /network/archive/t/tanc/new-dataset.
# Source run directories come from remd_sequence_locations.csv (built by
# helpers/sequence_location_map.py), which already resolved each sequence to
# its single canonical completed (>= 25 ns) run across the four storage
# locations (uniref, idrome, extra, oligo-data).
#
# --array=0-499%50 (500 tasks, ~72 runs each, max 50 running concurrently) --
# update NUM_TASKS to match if you resubmit with a different --array range.
NUM_TASKS=500

cd "$SLURM_SUBMIT_DIR/helpers"
python demux_new_dataset_shard.py "$SLURM_ARRAY_TASK_ID" "$NUM_TASKS"
