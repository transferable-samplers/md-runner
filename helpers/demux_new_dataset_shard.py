#!/usr/bin/env python3
"""Demultiplex a shard of the canonical 36,000 UniRef50Chunks/IDRomeChunks
REMD runs (all temperatures) into /network/archive/t/tanc/new-dataset.

Source run directories come from remd_sequence_locations.csv (built by
helpers/sequence_location_map.py) rather than a single remd/ root, since the
canonical run per sequence is spread across four storage locations (uniref,
idrome, extra, oligo-data) -- the CSV already resolved each sequence to its
one completed (>= 25 ns) run.

Intended to be driven by sbatch/run_demux_new_dataset.sh as a Slurm array:
each array task processes every Nth row (rows[task_id::num_tasks]).

Reuses demux_all.py's process_run_dir(), so output format matches it exactly:
one <run_dir.name>.trajectories.npz (all temperatures) + <run_dir.name>.swap_rates.txt
per sequence, written into OUT_DIR. Since run_dir.name always starts with the
sequence (see UniRef50Chunks.csv / IDRomeChunks.csv), outputs stay keyed by
sequence via that prefix.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from demux_all import process_run_dir

CSV_PATH = Path(__file__).resolve().parent.parent / "remd_sequence_locations.csv"
OUT_DIR = Path("/network/archive/t/tanc/new-dataset")


def load_canonical_run_dirs() -> list[Path]:
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    missing = [row["sequence"] for row in rows if not row["run_dir"]]
    if missing:
        raise RuntimeError(f"{len(missing)} rows in {CSV_PATH} have no run_dir (not done)")
    return [Path(row["run_dir"]) for row in rows]


def main() -> None:
    task_id = int(sys.argv[1])
    num_tasks = int(sys.argv[2])

    runs = load_canonical_run_dirs()
    shard = runs[task_id::num_tasks]
    print(f"Task {task_id}/{num_tasks}: {len(shard)} of {len(runs)} runs")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = bad = skipped = 0
    for i, d in enumerate(shard, 1):
        out_nc = OUT_DIR / f"{d.name}.trajectories.npz"
        if out_nc.exists():
            print(f"[{i}/{len(shard)}] skip {d.name} (already exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(shard)}] {d.name}")
        n_frames = process_run_dir(d, out_dir=OUT_DIR, save_swap_rates=True, stride=1)
        if n_frames is not None:
            ok += 1
        else:
            bad += 1

    print(f"Task {task_id}/{num_tasks} done: {ok} ok, {bad} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
