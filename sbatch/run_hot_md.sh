#!/bin/bash
#SBATCH -J hot_md_dihedral_check
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=24G
#SBATCH -t 48:00:00
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=cn-g[001-029],cn-k[001-004],cn-b[001-005],cn-i001,cn-j001
#SBATCH --get-user-env

echo "Node: $HOSTNAME"

# Unconstrained (generate_md.py always runs constraints=None) hot-MD run on MAPQTIAT (has
# Pro, 2x Thr, 1x Ile) at 1200K, to 1us, as the baseline counterpart to
# run_hot_md_restrained.sh -- both used to check whether the chirality/omega flat-bottom
# restraints (tol=25/80 deg) actually suppress flips relative to no restraint at all.
/network/scratch/t/tanc/micromamba/envs/md-runner/bin/python src/generate_md.py \
  seq_name=MAPQTIAT \
  pdb_dir=/home/mila/t/tanc/scratch/remd-final/pdbs \
  temperature=1200 \
  warmup_steps=20_000 \
  frame_interval=500 \
  frames_per_chunk=2_000 \
  time_ns=1000 \
  paths.scratch_dir=/network/scratch/t/tanc/md-runner-hot-md
