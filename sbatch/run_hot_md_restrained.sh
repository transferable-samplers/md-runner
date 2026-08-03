#!/bin/bash
#SBATCH -J hot_md_restrained_check
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

# Restrained hot-MD run on MAPQTIAT (has Pro, 2x Thr, 1x Ile -- exercises the backbone
# Calpha/Halpha AND the Thr/Ile secondary Cbeta/Hbeta chirality restraints, plus the omega
# restraint at a X-Pro-adjacent and normal peptide bonds), at 1200K only, to 1us. Uses the
# current chirality_tol_deg=25/omega_tol_deg=80 defaults (see configs/generate_remd.yaml) to
# check the tighter chirality tol (matching the reference formula's phitol=25) actually keeps
# the dihedral from crossing the planar/inversion point, unlike the earlier tol=40 attempt.
#
# Separate scratch_dir from the unrestrained baseline run so trajectories can't collide.
/network/scratch/t/tanc/micromamba/envs/md-runner/bin/python src/generate_md.py \
  seq_name=MAPQTIAT \
  pdb_dir=/home/mila/t/tanc/scratch/remd-final/pdbs \
  temperature=1200 \
  warmup_steps=20_000 \
  frame_interval=500 \
  frames_per_chunk=2_000 \
  time_ns=1000 \
  chirality_restraint=true \
  omega_restraint=true \
  paths.scratch_dir=/network/scratch/t/tanc/md-runner-hot-md-restrained
