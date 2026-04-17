#!/bin/bash
#SBATCH -J check_copies
#SBATCH --mem=4G
#SBATCH -t 48:00:00
#SBATCH --partition=unkillable-cpu
#SBATCH --ntasks-per-node=1
#SBATCH -c 2
#SBATCH --open-mode=append
#SBATCH --get-user-env

echo "Node: $HOSTNAME"
echo "Starting verification job"

SCRATCH=/network/scratch/t/tanc
ARCHIVE=/network/archive/t/tanc

check_only () {
    SRC=$1
    DST=$2

    echo "========================================"
    echo "Checking: $SRC -> $DST"

    # Run checksum dry-run and filter out non-informative lines
    OUTPUT=$(rsync -rtvnc "$SRC" "$DST" \
        | grep -Ev "^(sending incremental file list|sent .* bytes|total size is)")

    if [ -z "$OUTPUT" ]; then
        echo "✅ MATCH (all files identical)"
    else
        echo "❌ DIFFERENCES FOUND:"
        echo "$OUTPUT"
        FAIL=1
    fi
}

FAIL=0

# ---- RUN ALL CHECKS ----

check_only "$SCRATCH/md-runner-eval"        "$ARCHIVE/ManyPeptidesMDBackup/"
check_only "$SCRATCH/md-runner-temperature" "$ARCHIVE/ManyPeptidesMDBackup/"
check_only "$SCRATCH/md-log-8AA"            "$ARCHIVE/ManyPeptidesMDBackup/"

echo "========================================"

if [ "$FAIL" -eq 0 ]; then
    echo "🎉 ALL CHECKS PASSED"
    exit 0
else
    echo "🚨 SOME TRANSFERS ARE INCOMPLETE OR DIFFERENT"
    exit 1
fi