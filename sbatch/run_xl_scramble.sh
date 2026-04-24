#!/bin/bash
#SBATCH -J generate_remd_reference_xl_scramble
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --array=0-11
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
SEQ_FILE="sequences/xl.txt"
TIME_NS=5000           # <-- 5 us reference run

TOTAL_PER_JOB=4        # <-- N total sequences handled by this slurm task
MAX_CONCURRENT=4        # <-- at most 4 python processes at a time
TOTAL_SEQS=$(wc -l < "$SEQ_FILE")

# Seeds to run (one seed per REMD run, different velocity seed -> different
# post-scramble starting structure). Array layout: one task per (seed, block).
SEEDS=(1 2 3 4)
BLOCKS=(0 1 2)        # must match the original run_xl.sh sequence coverage
N_SEEDS=${#SEEDS[@]}
N_BLOCKS=${#BLOCKS[@]}

# Thermal scramble parameters.
SCRAMBLE_HIGH_TEMP=600
SCRAMBLE_RAMP_UP_PS=1000
SCRAMBLE_HOLD_PS=1000
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
    n_states=auto \
    time_ns=$TIME_NS \
    constraints=null \
    timestep_fs=1.0 \
    frame_interval=5000 \
    scramble=true \
    scramble_seed=$SEED \
    scramble_high_temp=$SCRAMBLE_HIGH_TEMP \
    scramble_ramp_up_ps=$SCRAMBLE_RAMP_UP_PS \
    scramble_hold_ps=$SCRAMBLE_HOLD_PS \
    scramble_ramp_down_ps=$SCRAMBLE_RAMP_DOWN_PS \
    scramble_equilibrate_ps=$SCRAMBLE_EQUILIBRATE_PS \
    paths.scratch_dir=/network/scratch/t/tanc/md-runner-remd-reference-xl &

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
