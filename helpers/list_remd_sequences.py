#!/usr/bin/env python3
"""Print one '<subset>\t<sequence>' line per unique REMD sequence.

Deterministic order (used as the sbatch-array manifest): roots in REMD_ROOTS
order, sequences sorted within each subset. The line number (0-based) is the
SLURM_ARRAY_TASK_ID that should process that sequence.
"""

from __future__ import annotations

from remd_progress import REMD_ROOTS, find_remd_runs


def subset_tag(root) -> str:
    return root.parents[1].name.replace("md-runner-remd-reference-", "")


def main() -> None:
    for root in REMD_ROOTS:
        if not root.exists():
            continue
        tag = subset_tag(root)
        seqs = sorted({r.name.split("_")[0] for r in find_remd_runs(root)})
        for seq in seqs:
            print(f"{tag}\t{seq}")


if __name__ == "__main__":
    main()
