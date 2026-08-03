#!/bin/bash
#SBATCH -J upload_to_hf
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --partition=long-cpu
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --open-mode=append
#SBATCH --get-user-env

set -euo pipefail

echo "Node: $HOSTNAME"
echo "Job: $SLURM_JOB_ID"

python src/upload_to_hf.py

echo "Done."
