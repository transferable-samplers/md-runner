#!/bin/bash
#SBATCH -J round_trips_audit
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --partition=long-cpu,main-cpu
#SBATCH --mem=12G
#SBATCH -c 6
#SBATCH -t 4:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --get-user-env

set -euo pipefail

echo "Node: $HOSTNAME"
echo "Job: $SLURM_JOB_ID"

# Round-trip audit of all 36,000 canonical UniRef50Chunks/IDRomeChunks REMD runs
# listed in remd_sequence_locations.csv (built by helpers/sequence_location_map.py).
# For every run, reads the replica->state trajectory from remd.nc and counts full
# 300K -> 450K -> 300K round trips per replica.
#
# Output is appended to round_trips_all.csv in the repo root; already-finished
# sequences are skipped, so a requeued or resubmitted job picks up where it left off.
# Reading is I/O bound on the archive filesystem, so workers > cores is fine.
PYTHON=/network/scratch/t/tanc/micromamba/envs/md-runner/bin/python
WORKERS=$(( SLURM_CPUS_PER_TASK * 4 ))

cd "$SLURM_SUBMIT_DIR"

echo "=== Counting round trips ($WORKERS workers) ==="
time "$PYTHON" helpers/round_trips_all.py "$WORKERS" remd_sequence_locations.csv round_trips_all.csv

echo "=== Summary ==="
"$PYTHON" helpers/round_trips_summary.py round_trips_all.csv

echo "=== Done ==="
