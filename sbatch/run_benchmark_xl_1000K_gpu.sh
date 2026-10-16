#!/bin/bash
#SBATCH -J bench_xl_1000K_gpu
#SBATCH -o watch_folder/%x_%A.out
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 8
#SBATCH --exclude=cn-g[001-029],cn-k[001-004],cn-b[001-005],cn-i001,cn-j001
#SBATCH --get-user-env

# GPU throughput benchmark for one xl 1000K sequence (PGPWLEEEEEAYGWMDF), same physics as
# run_xl_1000K.sh (300-450K ladder, n_states=auto -> 7 replicas for this sequence, 1 fs, no constraints, chirality+omega restraints,
# 5000-step swap interval) but scramble off and a short run so only the REMD loop is timed.
# Single process, no MPS (production packs 4 processes per GPU under MPS). Summarize with
#   python helpers/summarize_benchmark.py <hydra log>
echo "Node: $HOSTNAME"; nvidia-smi -L
source "$(dirname "$0")/benchmark_xl_1000K_common.sh" 2>/dev/null || source sbatch/benchmark_xl_1000K_common.sh
run_bench gpu_rtx8000 platform=gpu
