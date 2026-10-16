#!/bin/bash
# Shared launcher for run_benchmark_xl_1000K_{gpu,cpu}.sh. Usage: run_bench <tag> [extra hydra overrides]
BENCH_DIR=/home/mila/t/tanc/scratch/md-runner-benchmark
SEQ_FILE="sequences/xl.txt"
SEQ_IDX=8                 # PGPWLEEEEEAYGWMDF (line 9 of sequences/xl.txt)
TIME_NS=0.1               # 20 iterations x 5000 steps x 1 fs, per replica
WARMUP_NS=0.01            # 2 equilibration iterations

run_bench() {
  local tag=$1; shift
  local data_dir="$BENCH_DIR/data/bench_${tag}"
  rm -rf "$data_dir/remd"   # always start fresh so nothing resumes
  echo "bench tag=$tag  data_dir=$data_dir  extra: $*"
  python src/generate_remd.py \
    seq_idx=$SEQ_IDX \
    seq_filename="$SEQ_FILE" \
    pdb_dir="$BENCH_DIR/data/pdbs" \
    paths.scratch_dir="$BENCH_DIR" \
    paths.data_dir="$data_dir" \
    n_states=auto \
    time_ns=$TIME_NS \
    warmup_time_ns=$WARMUP_NS \
    constraints=null \
    timestep_fs=1.0 \
    frame_interval=5000 \
    chirality_restraint=true \
    omega_restraint=true \
    scramble=false \
    min_temp=300 \
    max_temp=450 \
    "$@"
}
