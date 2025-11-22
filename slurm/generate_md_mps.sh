#!/bin/bash
#SBATCH -J generate_md
#SBATCH -o watch_folder/%x_%A_%a.out       # A = array job ID, a = array task ID
#SBATCH --array=0-110                       # <-- adjust array length as needed
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90

echo "Running on node: $HOSTNAME"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Starting CUDA MPS…"

# ============================
# Create per-node MPS folders
# ============================
MPS_DIR=/tmp/$USER/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$MPS_DIR/pipe"
mkdir -p "$MPS_DIR/log"

export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"

# Start the MPS server
nvidia-cuda-mps-control -d
echo "MPS started at $CUDA_MPS_PIPE_DIRECTORY"

# ============================
# Compute 4 sequence indices
# ============================
BASE_IDX=$(( SLURM_ARRAY_TASK_ID * 4 ))

IDX1=$(( BASE_IDX + 0 ))
IDX2=$(( BASE_IDX + 1 ))
IDX3=$(( BASE_IDX + 2 ))
IDX4=$(( BASE_IDX + 3 ))

echo "Launching MD jobs for indices: $IDX1, $IDX2, $IDX3, $IDX4"

# ============================
# Launch 4 MD jobs in parallel
# ============================
python src/generate_md.py seq_idx=$IDX1 seq_filename=sequences/sequences.txt &
python src/generate_md.py seq_idx=$IDX2 seq_filename=sequences/sequences.txt &
python src/generate_md.py seq_idx=$IDX3 seq_filename=sequences/sequences.txt &
python src/generate_md.py seq_idx=$IDX4 seq_filename=sequences/sequences.txt &

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
