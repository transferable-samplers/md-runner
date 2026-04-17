#!/usr/bin/env python3
"""Collate chunk_*.npz files from MD runs into single trajectory .npz files.

Designed for SLURM array parallelism. Each task processes a chunk of run dirs.

Output per run: {run_name}.trajectories.npz with key "positions" (n_frames, n_atoms, 3).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


def find_md_runs(md_root: Path) -> list[Path]:
    """Return run directories containing a chunks/ subdirectory with .npz files."""
    runs = []
    for d in md_root.iterdir():
        if not d.is_dir():
            continue
        chunks_dir = d / "chunks"
        if chunks_dir.is_dir() and any(chunks_dir.glob("chunk_*.npz")):
            runs.append(d)
    return sorted(runs)


def collate_run(run_dir: Path, out_dir: Path, stride: int = 1) -> Optional[int]:
    """Concatenate all chunk_*.npz into a single trajectories.npz (positions only).

    Returns the total number of frames before striding, or None on failure.
    """
    chunks_dir = run_dir / "chunks"

    try:
        chunk_files = sorted(chunks_dir.glob("chunk_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
        if not chunk_files:
            print(f"  [SKIP] no chunk files in {chunks_dir}")
            return None

        all_positions = []
        for cf in chunk_files:
            data = np.load(cf)
            all_positions.append(data["positions"])

        positions = np.concatenate(all_positions, axis=0)  # (total_frames, n_atoms, 3)
        n_frames_raw = positions.shape[0]

        if stride > 1:
            positions = positions[::stride]

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_dir.name}.trajectories.npz"
        np.savez_compressed(out_path, positions=positions)

        print(f"[OK] {run_dir.name} -> {out_path.name}  frames={n_frames_raw}  pos={positions.shape}")
        return n_frames_raw

    except Exception as e:
        print(f"[FAIL] {run_dir}: {e}", file=sys.stderr)
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collate MD chunk trajectories. Supports SLURM array parallelism.",
    )
    p.add_argument("--md-root", type=Path, required=True, help="Path to a data/md/ directory")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for .npz and .csv files")
    p.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (e.g. --stride 4)")
    p.add_argument("--chunk-size", type=int, default=None, help="Runs per SLURM array task. If unset, process all.")
    args = p.parse_args()

    if not args.md_root.exists():
        print(f"Error: md root does not exist: {args.md_root}", file=sys.stderr)
        sys.exit(1)

    all_runs = find_md_runs(args.md_root)
    if not all_runs:
        print(f"Error: no run dirs with chunks/ found in {args.md_root}", file=sys.stderr)
        sys.exit(1)

    # SLURM array slicing
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is not None and args.chunk_size is not None:
        task_id = int(task_id)
        start = task_id * args.chunk_size
        end = start + args.chunk_size
        runs = all_runs[start:end]
        print(f"SLURM task {task_id}: runs [{start}:{end}] ({len(runs)} of {len(all_runs)})")
    else:
        runs = all_runs
        print(f"Processing all {len(runs)} runs")

    if not runs:
        print("No runs for this task, exiting.")
        sys.exit(0)

    ok = 0
    bad = 0
    results = []
    for i, d in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {d.name}")
        n_frames = collate_run(d, args.out_dir, args.stride)
        if n_frames is not None:
            ok += 1
            results.append((d.name, n_frames, "ok"))
        else:
            bad += 1
            results.append((d.name, 0, "fail"))

    print(f"\nSummary: {ok} succeeded, {bad} failed")

    # Write CSV log
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{task_id}" if task_id is not None else ""
    log_path = args.out_dir / f"collate_log{suffix}.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "frames", "status"])
        w.writerows(results)
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
