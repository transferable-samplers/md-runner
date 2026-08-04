#!/bin/bash
#SBATCH -J generate_remd_xl_1000K
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH -t 7-00:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 8
#SBATCH --array=0-8
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=cn-g[001-029],cn-k[001-004],cn-b[001-005],cn-i001,cn-j001
#SBATCH --get-user-env

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

# ============================
# Configuration
# ============================
# 1000K hot-equilibration ladder for the xl sequence set: 300K-1000K, chirality+omega
# restraints on, n_states from SEQUENCE_N_STATES_EVAL_OVERRIDES (per-sequence, extrapolated
# from each sequence's actual production 300-450K n_states -- see src/generate_remd.py).
# See run_pdbs_4aa_scramble.sh / run_xl_scramble.sh for the reference format this is adapted
# from.
SEQ_FILE="sequences/xl.txt"
PDB_DIR="/network/scratch/t/tanc/md-runner-remd-reference-xl-1000K/data"  # flat *.pdb, not data/pdbs/
SCRATCH_DIR="/network/scratch/t/tanc/md-runner-remd-reference-xl-1000K"
MIN_TEMP=300
MAX_TEMP=1000
TIME_NS=10000            # <-- 10 us cap; exit_early_at_ns still stops at 2 us
EXIT_EARLY_NS=2000
SWAP_INTERVAL=1000        # <-- 1ps swap attempts (was coupled 1:1 with frame_interval/5ps); checkpoints/frames still every 5ps

TOTAL_PER_JOB=4          # <-- N sequences handled by this slurm task, one fully-parallel MPS wave
MAX_CONCURRENT=4          # <-- at most 4 python processes at a time (matches run_pdbs_4aa_scramble.sh)
TOTAL_SEQS=$(wc -l < "$SEQ_FILE")

# 3 instances (independent scramble seeds) per sequence. Array layout: one task per (seed, block).
SEEDS=(1 2 3)
BLOCKS=(0 1 2)            # ceil(12/4)=3 blocks

N_SEEDS=${#SEEDS[@]}
N_BLOCKS=${#BLOCKS[@]}

# Thermal scramble parameters: 1000K hold (matches max_temp), 10 ns hold to cross barriers
# (long enough for proline cis/trans isomerization at this temperature -- see Arrhenius
# extrapolation in the REMD ladder design notes). Uses the same restrained system as the
# main run (add_chirality_restraints/add_omega_restraints are added to `system` before
# thermal_scramble() runs, so the scramble hold is restrained too).
SCRAMBLE_HIGH_TEMP=1000
SCRAMBLE_RAMP_UP_PS=1000
SCRAMBLE_HOLD_PS=10000
SCRAMBLE_RAMP_DOWN_PS=5000
SCRAMBLE_EQUILIBRATE_PS=1000

# ----------------------------
# Map array task -> (seed, block)
# ----------------------------
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / N_BLOCKS ))
BLOCK_IDX=$(( SLURM_ARRAY_TASK_ID % N_BLOCKS ))
if [ "$SEED_IDX" -ge "$N_SEEDS" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds seed*block grid; exiting."
  exit 0
fi
SEED=${SEEDS[$SEED_IDX]}
BLOCK=${BLOCKS[$BLOCK_IDX]}
BASE_IDX=$(( BLOCK * TOTAL_PER_JOB ))

echo "Seed=$SEED  block=$BLOCK  base_idx=$BASE_IDX"

# ============================
# Start CUDA MPS
# ============================

# Create task-array-unique MPS directories
MPS_DIR=/tmp/$USER/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"
export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"

cleanup() {
  echo "Cleaning up..."
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  rm -rf "$MPS_DIR" || true
}
trap cleanup EXIT SIGUSR1 SIGTERM SIGINT

nvidia-cuda-mps-control -d
echo "MPS server started at $CUDA_MPS_PIPE_DIRECTORY"

# ----------------------------
# Launch sequences for this (seed, block)
# ----------------------------
echo "Launching up to $TOTAL_PER_JOB sequences starting at idx=$BASE_IDX with seed=$SEED"
echo "Concurrency cap: $MAX_CONCURRENT"

running=0

for ((k=0; k<TOTAL_PER_JOB; k++)); do
  IDX=$(( BASE_IDX + k ))
  if [ "$IDX" -ge "$TOTAL_SEQS" ]; then
    echo "Reached end of sequence file (idx=$IDX >= $TOTAL_SEQS); stopping launches."
    break
  fi
  echo "Launching seq_idx=$IDX seed=$SEED"
  python src/generate_remd.py \
    seq_idx=$IDX \
    seq_filename="$SEQ_FILE" \
    pdb_dir="$PDB_DIR" \
    n_states=eval-max \
    time_ns=$TIME_NS \
    constraints=null \
    timestep_fs=1.0 \
    frame_interval=5000 \
    swap_interval=$SWAP_INTERVAL \
    chirality_restraint=true \
    omega_restraint=true \
    scramble=true \
    scramble_seed=$SEED \
    scramble_high_temp=$SCRAMBLE_HIGH_TEMP \
    scramble_ramp_up_ps=$SCRAMBLE_RAMP_UP_PS \
    scramble_hold_ps=$SCRAMBLE_HOLD_PS \
    scramble_ramp_down_ps=$SCRAMBLE_RAMP_DOWN_PS \
    scramble_equilibrate_ps=$SCRAMBLE_EQUILIBRATE_PS \
    exit_early_at_ns=$EXIT_EARLY_NS \
    min_temp=$MIN_TEMP \
    max_temp=$MAX_TEMP \
    paths.scratch_dir=$SCRATCH_DIR &

  running=$(( running + 1 ))
  if [ "$running" -ge "$MAX_CONCURRENT" ]; then
    # wait for any one job to finish, then continue launching
    wait -n
    running=$(( running - 1 ))
  fi
done

# wait for remaining
wait
echo "All processes completed."
