#!/bin/bash
#SBATCH -J extract_small_positions
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --partition=long-cpu
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --array=0-11
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --get-user-env

set -euo pipefail

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

NUM_SHARDS=12

python src/extract_small_positions.py \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SLURM_ARRAY_TASK_ID"

echo "Done: shard $SLURM_ARRAY_TASK_ID/$NUM_SHARDS"
