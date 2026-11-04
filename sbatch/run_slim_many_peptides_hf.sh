#!/bin/bash
#SBATCH -J slim_many_peptides_hf
#SBATCH -o watch_folder/%x_%j.out
#SBATCH --partition=main-cpu
#SBATCH -c 7
#SBATCH --mem=28G
#SBATCH -t 5-00:00:00
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --get-user-env
#
# Drop velocities + subsample train x10 on transferable-samplers/many-peptides-md,
# then verify. Resumable: rerunning skips files already in manifest.jsonl.
#
#   sbatch sbatch/run_slim_many_peptides_hf.sh                        # slim (+ metadata verify at end)
#   MODE=dryrun LIMIT=20 sbatch sbatch/run_slim_many_peptides_hf.sh   # process 20 files, no upload
#   MODE=verify sbatch sbatch/run_slim_many_peptides_hf.sh            # full re-download content verify
#   TARGET_REVISION=slim sbatch ...                                   # commit to a branch instead of main
#
# Requires `hf auth login` (write token for the transferable-samplers org) beforehand.

set -euo pipefail

MODE=${MODE:-slim}
WORK_DIR=${WORK_DIR:-/network/scratch/t/tanc/hf_slim_many_peptides}
TARGET_REVISION=${TARGET_REVISION:-main}
WORKERS=${WORKERS:-$SLURM_CPUS_PER_TASK}
PY=/network/scratch/t/tanc/micromamba/envs/md-runner/bin/python

export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60

cd /home/mila/t/tanc/md-runner
echo "Node: $HOSTNAME  job: $SLURM_JOB_ID  mode: $MODE  work_dir: $WORK_DIR  target: $TARGET_REVISION"
mkdir -p "$WORK_DIR"

case "$MODE" in
  slim)
    "$PY" helpers/slim_many_peptides_hf.py \
      --work-dir "$WORK_DIR" \
      --workers "$WORKERS" \
      --shard-size 500 \
      --target-revision "$TARGET_REVISION"
    # cheap check: every file present on remote with the expected sha256
    "$PY" helpers/verify_many_peptides_hf.py \
      --manifest "$WORK_DIR/manifest.jsonl" \
      --revision "$TARGET_REVISION"
    ;;
  dryrun)
    "$PY" helpers/slim_many_peptides_hf.py \
      --work-dir "$WORK_DIR/dryrun" \
      --workers "$WORKERS" \
      --shard-size "${LIMIT:-20}" \
      --limit "${LIMIT:-20}" \
      --dry-run
    ;;
  verify)
    mkdir -p "$WORK_DIR/verify_tmp"
    "$PY" helpers/verify_many_peptides_hf.py \
      --manifest "$WORK_DIR/manifest.jsonl" \
      --revision "$TARGET_REVISION" \
      --redownload \
      --workers "$WORKERS" \
      --tmp-dir "$WORK_DIR/verify_tmp"
    ;;
  *)
    echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac

echo "Done: $MODE"
