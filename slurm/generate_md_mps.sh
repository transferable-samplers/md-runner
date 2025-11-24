#!/bin/bash
#SBATCH -J generate_md_mps
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --array=0-0
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
SEQ_FILE="sequences/example_sequences.txt"
PROCS_PER_GPU=4

echo "Processes per GPU: $PROCS_PER_GPU"

# ============================
# Start CUDA MPS
# ============================

# Create task-array-unique MPS directories
MPS_DIR=/tmp/$USER/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"

export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"

# Start the MPS server
nvidia-cuda-mps-control -d
echo "MPS server started at $CUDA_MPS_PIPE_DIRECTORY"

# ============================
# Compute starting index
# ============================
BASE_IDX=$(( SLURM_ARRAY_TASK_ID * PROCS_PER_GPU ))

echo "Launching MD jobs from indices $BASE_IDX to $(( BASE_IDX + PROCS_PER_GPU - 1 ))"

# ============================
# Launch N processes
# ============================
for ((i=0; i<PROCS_PER_GPU; i++)); do
    IDX=$(( BASE_IDX + i ))
    echo "Launching process for seq_idx=$IDX"
    python src/generate_md.py seq_idx=$IDX seq_filename=$SEQ_FILE &
done

wait
echo "All MD jobs completed."

# ============================
# Shut down MPS
# ============================
echo quit | nvidia-cuda-mps-control
echo "MPS stopped."

rm -rf "$MPS_DIR"
echo "Cleaned up MPS directory."

echo "Done."
