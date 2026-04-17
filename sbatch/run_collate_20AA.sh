#!/bin/bash
#SBATCH -J collate_20AA
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH -c 2
#SBATCH --array=0-1
#SBATCH --open-mode=append

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

MD_ROOT=/network/archive/t/tanc/oligo/md-runner-20AA/data/md
OUT_DIR=/network/archive/t/tanc/ASSORTED
CHUNK_SIZE=1000
STRIDE=4

python helpers/collate_chunks.py \
  --md-root "$MD_ROOT" \
  --out-dir "$OUT_DIR" \
  --stride "$STRIDE" \
  --chunk-size "$CHUNK_SIZE"
