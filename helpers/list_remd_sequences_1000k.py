#!/usr/bin/env python3
"""Print one '<subset>\t<sequence>' line per unique REMD sequence, for the 1000K
hot-equilibration runs (4aa/8aa share the "many-1000K" root, xl uses "xl-1000K").

Deterministic order (used as the sbatch-array manifest): roots in ROOTS_1000K
order, sequences sorted within each subset. The line number (0-based) is the
SLURM_ARRAY_TASK_ID that should process that sequence.
"""

from __future__ import annotations

from pathlib import Path

from remd_progress import find_remd_runs

ROOTS_1000K = [
    Path("/network/scratch/t/tanc/md-runner-remd-reference-many-1000K/data/remd"),
    Path("/network/scratch/t/tanc/md-runner-remd-reference-xl-1000K/data/remd"),
]


def subset_tag(root: Path) -> str:
    return root.parents[1].name.replace("md-runner-remd-reference-", "")


def main() -> None:
    for root in ROOTS_1000K:
        if not root.exists():
            continue
        tag = subset_tag(root)
        seqs = sorted({r.name.split("_")[0] for r in find_remd_runs(root)})
        for seq in seqs:
            print(f"{tag}\t{seq}")


if __name__ == "__main__":
    main()
