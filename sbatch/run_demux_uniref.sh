#!/bin/bash
#SBATCH -J demux_uniref
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -c 2
#SBATCH --array=0-30
#SBATCH --open-mode=append

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

REMD_ROOT=/network/scratch/t/tanc/md-runner-remd-uniref/data/remd
OUT_DIR=/network/archive/t/tanc/OLIGO
CHUNK_SIZE=1000
STRIDE=4

python helpers/demux_slurm.py \
  --remd-root "$REMD_ROOT" \
  --out-dir "$OUT_DIR" \
  --stride "$STRIDE" \
  --chunk-size "$CHUNK_SIZE"
