#!/bin/bash
#SBATCH -J bench_xl_1000K_cpu
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH --partition=long-cpu
#SBATCH -c 8
#SBATCH --array=0-1
#SBATCH --get-user-env

# CPU counterpart of run_benchmark_xl_1000K_gpu.sh: OpenMM CPU platform with 1 or 8 threads
# (array task 0 / 1) on 8 allocated cores.
THREADS_LIST=(1 8)
THREADS=${THREADS_LIST[$SLURM_ARRAY_TASK_ID]}
echo "Node: $HOSTNAME  threads=$THREADS"; lscpu | grep "Model name"
source sbatch/benchmark_xl_1000K_common.sh
run_bench cpu_${THREADS}thr platform=cpu platform_properties.Threads=$THREADS
