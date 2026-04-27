#!/bin/bash
#SBATCH -J xl_identity
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -c 8
#SBATCH --partition=long-cpu
#SBATCH --open-mode=append

set -euo pipefail

echo "Node: $HOSTNAME"
echo "Job: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"

cd "$SLURM_SUBMIT_DIR"

for CUTOFF in 0.5 0.4; do
    TAG="${CUTOFF/./_}"
    OUT="xl_hits_after_${TAG}.txt"
    echo
    echo "=== top hits after filtering PDBs with identity >= ${CUTOFF} ==="
    time python helpers/xl_sequence_identity.py \
        --exclude-cutoff "$CUTOFF" \
        --top 5 \
        --workers "$SLURM_CPUS_PER_TASK" \
        | tee "$OUT"
    echo "wrote $OUT"
done

for CUTOFF in 0.2 0.3 0.4; do
    TAG="${CUTOFF/./_}"
    DROP="drop_sequences_${TAG}.txt"
    echo
    echo "=== dropped PDBs at identity >= ${CUTOFF} (similarity script) ==="
    time python helpers/xl_sequence_similarity.py \
        --cutoff "$CUTOFF" \
        --drop-file "$DROP" \
        --workers "$SLURM_CPUS_PER_TASK"
    echo "wrote $DROP"
done

echo "=== Done ==="
