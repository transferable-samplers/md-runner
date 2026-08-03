#!/bin/bash
#SBATCH -J hot_md_dihedral_check_800
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=cn-g[001-029],cn-k[001-004],cn-b[001-005],cn-i001,cn-j001
#SBATCH --get-user-env

echo "Node: $HOSTNAME"

# Unconstrained baseline counterpart to run_hot_md_restrained_800.sh, at 800K -- long
# partition since main didn't have capacity.
/network/scratch/t/tanc/micromamba/envs/md-runner/bin/python src/generate_md.py \
  seq_name=MAPQTIAT \
  pdb_dir=/home/mila/t/tanc/scratch/remd-final/pdbs \
  temperature=800 \
  warmup_steps=20_000 \
  frame_interval=500 \
  frames_per_chunk=2_000 \
  time_ns=1000 \
  paths.scratch_dir=/network/scratch/t/tanc/md-runner-hot-md
