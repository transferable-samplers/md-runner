#!/bin/bash
#SBATCH -J generate_remd_mps_retry
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --array=0-3
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
# Retry batch: the 14 sequences from run_oligopeptides_extra.sh (job 10420694) that never
# started, all killed by the same transient scratch I/O error (OSError: [Errno 121] Remote
# I/O error) while hydra wrote its run-dir config at job startup. PDBs already exist from the
# original build; reuse the same scratch dir.
SEQ_FILE="sequences/oligopeptides_extra_retry.txt"
SCRATCH_DIR="/network/scratch/t/tanc/md-runner-oligopeptides-extra"
PDB_DIR="$SCRATCH_DIR/data/pdbs"

TOTAL_PER_JOB=4         # <-- N sequences handled by this slurm task
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
  python src/generate_remd.py seq_idx=$IDX seq_filename="$SEQ_FILE" pdb_dir="$PDB_DIR" n_states=auto paths.scratch_dir="$SCRATCH_DIR" &

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
