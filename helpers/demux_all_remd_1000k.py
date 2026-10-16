#!/usr/bin/env python3
"""Demultiplex the 1000K hot-equilibration REMD runs (4aa/8aa/xl) into
per-temperature .npz arrays.

Same output layout, .npz contents, and CLI as demux_all_remd.py, but reads
from ROOTS_1000K (many-1000K/xl-1000K) instead of REMD_ROOTS in
remd_progress.py -- those are the separate 300-450K production runs.

Output layout:
  <out-dir>/<subset>/<sequence>_<temperature>_<N>.npz

Where:
  <subset>      is "many-1000K" / "xl-1000K" (from ROOTS_1000K tag)
  <sequence>    is the leading token of the run dir name
  <temperature> is K to one decimal (e.g. "1000.0")
  <N>           is 0 for the no-scramble (original) run, 1 for scramble1, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demux_all_remd import process_run_dir, replica_index, subset_tag
from remd_progress import find_remd_runs, replica_label

ROOTS_1000K = [
    Path("/network/scratch/t/tanc/md-runner-remd-reference-many-1000K/data/remd"),
    Path("/network/scratch/t/tanc/md-runner-remd-reference-xl-1000K/data/remd"),
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Demultiplex 1000K REMD runs (many-1000K/xl-1000K) into per-temperature .npz arrays.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Root output directory; per-subset subdirs are created underneath.",
    )
    p.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Only process this subset (many-1000K/xl-1000K).",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Only process runs whose directory name contains this string.",
    )
    p.add_argument(
        "--exact-sequence",
        action="store_true",
        help="Treat --sequence as the exact leading token (run.name.split('_')[0]) "
        "instead of a substring.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth frame (e.g. --stride 4 keeps every 4th frame).",
    )
    p.add_argument(
        "--save-swap-rates",
        action="store_true",
        default=True,
        help="Also write neighbor swap rates next to the .npz files (default: True).",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a run if any of its expected .npz files already exist (default: True). "
        "Pass --no-skip-existing to overwrite.",
    )
    p.add_argument(
        "--lowest-temp-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only write the lowest-temperature trajectory per run (default: True). "
        "Pass --no-lowest-temp-only to write all temperatures.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_ok = 0
    total_bad = 0

    for root in ROOTS_1000K:
        if not root.exists():
            print(f"# skipping missing root: {root}")
            continue
        tag = subset_tag(root)
        if args.subset and tag != args.subset:
            continue

        runs = find_remd_runs(root)
        if args.sequence:
            if args.exact_sequence:
                runs = [r for r in runs if r.name.split("_")[0] == args.sequence]
            else:
                runs = [r for r in runs if args.sequence in r.name]
        if not runs:
            print(f"# no runs under {root}")
            continue

        out_subset_dir = args.out_dir / tag
        print(f"\n=== {tag} ({len(runs)} runs) -> {out_subset_dir} ===")

        for i, d in enumerate(runs, 1):
            seq = d.name.split("_")[0]
            n_idx = replica_index(replica_label(d.name))
            if args.skip_existing:
                existing = list(out_subset_dir.glob(f"{seq}_*_{n_idx}.npz"))
                if existing:
                    print(f"[{i}/{len(runs)}] skip {d.name} ({len(existing)} files exist)")
                    continue

            print(f"[{i}/{len(runs)}] {d.name}")
            n_frames = process_run_dir(
                d,
                out_subset_dir,
                args.save_swap_rates,
                args.stride,
                lowest_temp_only=args.lowest_temp_only,
            )
            if n_frames is not None:
                total_ok += 1
            else:
                total_bad += 1

    print(f"\nSummary: {total_ok} succeeded, {total_bad} failed")


if __name__ == "__main__":
    main()
