#!/bin/bash
#SBATCH -J generate_remd_mps_pdbs_8aa
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --array=0-7
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
SEQ_FILE="sequences/pdbs_8aa.txt"
TIME_NS=2000           # <-- 2 us for 8AA sequences

TOTAL_PER_JOB=4        # <-- N total sequences handled by this slurm task
MAX_CONCURRENT=4        # <-- at most 4 python processes at a time
TOTAL_SEQS=$(wc -l < "$SEQ_FILE")

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
# Sequence indexing
# ----------------------------
# Each array task handles a block of TOTAL_PER_JOB sequences
BASE_IDX=$(( SLURM_ARRAY_TASK_ID * TOTAL_PER_JOB ))

echo "Launching up to $TOTAL_PER_JOB sequences starting at idx=$BASE_IDX"
echo "Concurrency cap: $MAX_CONCURRENT"

running=0

for ((k=0; k<TOTAL_PER_JOB; k++)); do
  IDX=$(( BASE_IDX + k ))
  if [ "$IDX" -ge "$TOTAL_SEQS" ]; then
    echo "Reached end of sequence file (idx=$IDX >= $TOTAL_SEQS); stopping launches."
    break
  fi
  echo "Launching seq_idx=$IDX"
  python src/generate_remd.py seq_idx=$IDX seq_filename="$SEQ_FILE" n_states=auto-max time_ns=$TIME_NS constraints=null timestep_fs=1.0 frame_interval=5000 paths.scratch_dir=/network/scratch/t/tanc/md-runner-remd-reference-many exit_early_at_ns=1000 min_temp=310 &

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
