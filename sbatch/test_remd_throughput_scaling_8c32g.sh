#!/bin/bash
#SBATCH -J remd_throughput_scaling_8c32g
#SBATCH -o watch_folder/%x_%A_%a.out
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 8
#SBATCH --array=0-15
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=cn-g[001-029],cn-k[001-004],cn-b[001-005],cn-i001,cn-j001
#SBATCH --get-user-env

set -u

echo "Node: $HOSTNAME"
echo "SLURM array ID: $SLURM_ARRAY_TASK_ID"

mkdir -p watch_folder

# ============================
# Configuration
# ============================
# Resource-scaled repeat of test_remd_throughput_scaling.sh (same longest xl sequence,
# same timing method, same 2-ladder x N=1..8-concurrency condition matrix), run under
# a smaller allocation (8 CPU / 32GB RAM vs the original 16 CPU / 48GB) to see whether
# throughput is sensitive to host-side CPU/RAM headroom at high replica counts, or is
# purely GPU-bound as expected. Separate TEST_SCRATCH_ROOT so results never collide
# with the original 16c/48G sweep's data.
SEQ_FILE="sequences/xl.txt"
SEQ_IDX=11

TEST_SCRATCH_ROOT=/network/scratch/t/tanc/md-runner-throughput-test-xl-8c32g

N_SKIP=3
N_MEASURE=20
POLL_INTERVAL_S=5
MAX_WAIT_S=3600 # safety net per condition in case something hangs/errors

# ----------------------------
# Condition matrix: array task -> (ladder, N_REPLICAS, use_mps)
# 16 tasks = 2 ladders x 8 concurrency conditions (N=1 no-MPS, N=2..8 MPS)
# ----------------------------
LADDERS=(450K 1000K)
N_CONDS=8

LADDER_IDX=$(( SLURM_ARRAY_TASK_ID / N_CONDS ))
COND_IDX=$(( SLURM_ARRAY_TASK_ID % N_CONDS ))
if [ "$LADDER_IDX" -ge "${#LADDERS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID exceeds ladder*cond grid; exiting."
  exit 0
fi
LADDER=${LADDERS[$LADDER_IDX]}
N_REPLICAS=$(( COND_IDX + 1 )) # 1..8
if [ "$N_REPLICAS" -eq 1 ]; then
  USE_MPS=0
else
  USE_MPS=1
fi

echo "Condition: ladder=$LADDER n_replicas=$N_REPLICAS use_mps=$USE_MPS"

if [ "$LADDER" = "450K" ]; then
  # Mirrors run_xl.sh's REMD parameters (production ladder, n_states=auto).
  # PDB_DIR points at the -1000K reference's pdb store (same file, flat layout) since
  # md-runner-remd-reference-xl/data/pdbs was archived+deleted.
  PDB_DIR=/network/scratch/t/tanc/md-runner-remd-reference-xl-1000K/data
  MIN_TEMP=300
  MAX_TEMP=450
  N_STATES=auto
  EXTRA_ARGS=""
else
  # Mirrors run_xl_1000K.sh's REMD parameters (n_states=eval-max, restraints on).
  # NOTE: production run_xl_1000K.sh also runs a scramble=true pre-equilibration
  # (thermal_scramble, ~17ns hold+ramps) before REMD starts. That's a one-off
  # preprocessing cost, not REMD sampler throughput, so it's deliberately left off
  # here (scramble stays at its default false) to isolate REMD throughput itself.
  PDB_DIR=/network/scratch/t/tanc/md-runner-remd-reference-xl-1000K/data
  MIN_TEMP=300
  MAX_TEMP=1000
  N_STATES=eval-max
  EXTRA_ARGS="chirality_restraint=true omega_restraint=true"
fi

RUN_TAG="task${SLURM_ARRAY_TASK_ID}_${LADDER}_n${N_REPLICAS}_mps${USE_MPS}"
OUT_ROOT="$TEST_SCRATCH_ROOT/$RUN_TAG"
mkdir -p "$OUT_ROOT"
RESULTS_FILE="$OUT_ROOT/results.csv"
echo "replica,ladder,n_replicas,use_mps,n_measured,mean_iter_seconds,ns_per_day" > "$RESULTS_FILE"

SUMMARY_FILE="$TEST_SCRATCH_ROOT/summary_scaling.csv"
if [ ! -f "$SUMMARY_FILE" ]; then
  echo "ladder,n_replicas,use_mps,mean_ns_per_day,n_ok,n_failed" > "$SUMMARY_FILE"
fi

# ============================
# CUDA MPS (only for use_mps=1)
# ============================
if [ "$USE_MPS" -eq 1 ]; then
  MPS_DIR=/tmp/$USER/mps_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
  mkdir -p "$MPS_DIR/pipe" "$MPS_DIR/log"
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR/pipe"
  export CUDA_MPS_LOG_DIRECTORY="$MPS_DIR/log"

  cleanup() {
    echo "Cleaning up MPS..."
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    rm -rf "$MPS_DIR" || true
  }
  trap cleanup EXIT SIGUSR1 SIGTERM SIGINT

  nvidia-cuda-mps-control -d
  echo "MPS server started at $CUDA_MPS_PIPE_DIRECTORY"
else
  echo "Running WITHOUT MPS (no nvidia-cuda-mps-control daemon started)."
fi

# ----------------------------
# Launch N_REPLICAS concurrent identical copies (background, PIDs recorded)
# ----------------------------
declare -a PIDS
declare -a LOGS
for ((r = 0; r < N_REPLICAS; r++)); do
  data_dir="$OUT_ROOT/replica_${r}/data"
  log_file="$OUT_ROOT/replica_${r}.log"
  LOGS[$r]="$log_file"
  python src/generate_remd.py \
    seq_idx=$SEQ_IDX \
    seq_filename="$SEQ_FILE" \
    pdb_dir="$PDB_DIR" \
    n_states=$N_STATES \
    constraints=null \
    timestep_fs=1.0 \
    frame_interval=5000 \
    min_temp=$MIN_TEMP \
    max_temp=$MAX_TEMP \
    paths.data_dir="$data_dir" \
    $EXTRA_ARGS \
    > "$log_file" 2>&1 &
  PIDS[$r]=$!
  echo "Launched replica $r (pid=${PIDS[$r]}) -> $log_file"
done

# ----------------------------
# Poll until every replica has logged >= N_SKIP + N_MEASURE iterations, then kill all
# ----------------------------
NEEDED=$((N_SKIP + N_MEASURE))
waited=0
while :; do
  min_count=999999
  all_dead=1
  for ((r = 0; r < N_REPLICAS; r++)); do
    if kill -0 "${PIDS[$r]}" 2>/dev/null; then
      all_dead=0
    fi
    c=$(grep -c "Iteration took" "${LOGS[$r]}" 2>/dev/null)
    c=${c:-0}
    if [ "$c" -lt "$min_count" ]; then
      min_count=$c
    fi
  done
  echo "poll: min_iterations_logged=$min_count / $NEEDED (waited=${waited}s)"
  if [ "$min_count" -ge "$NEEDED" ]; then
    echo "All replicas reached $NEEDED logged iterations."
    break
  fi
  if [ "$all_dead" -eq 1 ]; then
    echo "All replica processes exited before reaching $NEEDED iterations; stopping poll."
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    echo "MAX_WAIT_S=$MAX_WAIT_S exceeded; stopping poll with whatever was logged."
    break
  fi
  sleep "$POLL_INTERVAL_S"
  waited=$((waited + POLL_INTERVAL_S))
done

echo "Killing replica processes..."
for ((r = 0; r < N_REPLICAS; r++)); do
  kill -TERM "${PIDS[$r]}" 2>/dev/null || true
done
sleep 2
for ((r = 0; r < N_REPLICAS; r++)); do
  kill -KILL "${PIDS[$r]}" 2>/dev/null || true
done
wait 2>/dev/null || true

# ----------------------------
# Parse per-replica "Iteration took X.XXXs" lines -> mean iteration time -> ns/day
# ----------------------------
for ((r = 0; r < N_REPLICAS; r++)); do
  grep -oE "Iteration took [0-9.]+s" "${LOGS[$r]}" | grep -oE "[0-9.]+" \
    | tail -n +"$((N_SKIP + 1))" | head -n "$N_MEASURE" \
    > "$OUT_ROOT/replica_${r}_iters.txt"
  n=$(wc -l < "$OUT_ROOT/replica_${r}_iters.txt")
  if [ "$n" -lt 1 ]; then
    echo "replica $r: no iterations logged (n=$n), see ${LOGS[$r]}" >&2
    echo "$r,$LADDER,$N_REPLICAS,$USE_MPS,0,FAILED,FAILED" >> "$RESULTS_FILE"
    continue
  fi
  mean_iter_s=$(awk '{sum+=$1; n++} END {printf "%.4f", sum/n}' "$OUT_ROOT/replica_${r}_iters.txt")
  # ns/iteration = frame_interval(5000) * timestep_fs(1.0) / 1e6; ns/day = that / mean_iter_s * 86400
  ns_day=$(awk -v mi="$mean_iter_s" 'BEGIN{printf "%.1f", (5000*1.0/1e6) / mi * 86400}')
  echo "replica $r: n=$n mean_iter=${mean_iter_s}s -> ${ns_day} ns/day"
  echo "$r,$LADDER,$N_REPLICAS,$USE_MPS,$n,$mean_iter_s,$ns_day" >> "$RESULTS_FILE"
done

echo "=== Throughput summary (ladder=$LADDER n_replicas=$N_REPLICAS use_mps=$USE_MPS) ==="
awk -F, '
  NR>1 {
    if ($7=="FAILED") { nfail++; next }
    sum+=$7; nok++
  }
  END {
    mean = (nok>0) ? sum/nok : 0
    printf "MEAN across %d/%d replicas: %.1f ns/day (%d failed)\n", nok, nok+nfail, mean, nfail
    print mean, nok+0, nfail+0
  }
' "$RESULTS_FILE" > "$OUT_ROOT/_summary_line.txt"
cat "$OUT_ROOT/_summary_line.txt"
read -r MEAN NOK NFAIL < <(tail -n1 "$OUT_ROOT/_summary_line.txt")
echo "$LADDER,$N_REPLICAS,$USE_MPS,$MEAN,$NOK,$NFAIL" >> "$SUMMARY_FILE"

echo "Results: $RESULTS_FILE"
echo "Summary: $SUMMARY_FILE"
echo "All done."
