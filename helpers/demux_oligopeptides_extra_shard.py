#!/usr/bin/env python3
"""Demultiplex a shard of the oligopeptides_extra REMD runs (all temperatures).

Intended to be driven by sbatch/run_demux_oligopeptides_extra.sh as a Slurm array:
each array task processes every Nth run dir (runs[task_id::num_tasks]) so the
1875 runs under REMD_ROOT are split evenly across array tasks.

Reuses demux_all.py's find_remd_runs()/process_run_dir() so output format matches
that script exactly: one trajectories.npz (all temperatures) + swap_rates.txt
written into each run directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

from demux_all import find_remd_runs, process_run_dir

REMD_ROOT = Path("/network/scratch/t/tanc/md-runner-oligopeptides-extra/data/remd")


def main() -> None:
    task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    runs = find_remd_runs(REMD_ROOT)
    shard = runs[task_id::num_tasks]
    print(f"Task {task_id}/{num_tasks}: {len(shard)} of {len(runs)} runs")

    ok = bad = 0
    for i, d in enumerate(shard, 1):
        print(f"[{i}/{len(shard)}] {d.name}")
        n_frames = process_run_dir(d, out_dir=None, save_swap_rates=True, stride=1)
        if n_frames is not None:
            ok += 1
        else:
            bad += 1

    print(f"Task {task_id}/{num_tasks} done: {ok} ok, {bad} failed")


if __name__ == "__main__":
    main()
