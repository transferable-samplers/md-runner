#!/usr/bin/env python3
"""Demultiplex all REMD runs across REMD_ROOTS into per-temperature .npz arrays.

Output layout:
  <out-dir>/<subset>/<sequence>_<temperature>_<N>.npz

Where:
  <subset>      is "alanine" / "many" / "xl" (from REMD_ROOTS tag)
  <sequence>    is the leading token of the run dir name (e.g. "AA")
  <temperature> is K to one decimal (e.g. "300.0")
  <N>           is 0 for the no-scramble (original) run, 1 for scramble1, etc.

Each .npz contains a single key "positions": float32 array of shape
(n_ckpt, n_atoms, 3) in nm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from demux_all import (
    _safe_close_reporter,
    compute_neighbor_swap_rates,
    demultiplex_trajectories_from_reporter,
    load_temperatures,
)
from openmmtools import multistate
from remd_progress import REMD_ROOTS, find_remd_runs, replica_label


def replica_index(label: str) -> int:
    """original -> 0, scrambleK -> K."""
    if label == "original":
        return 0
    return int(label.replace("scramble", ""))


def subset_tag(root: Path) -> str:
    return root.parents[1].name.replace("md-runner-remd-reference-", "")


def process_run_dir(
    run_dir: Path,
    out_subset_dir: Path,
    save_swap_rates: bool,
    stride: int,
    lowest_temp_only: bool = False,
) -> Optional[int]:
    nc_path = run_dir / "remd.nc"
    ckpt_path = run_dir / "remd_checkpoint.nc"
    reporter = None

    seq = run_dir.name.split("_")[0]
    n_idx = replica_index(replica_label(run_dir.name))

    try:
        reporter = multistate.MultiStateReporter(
            str(nc_path),
            checkpoint_storage=str(ckpt_path),
            open_mode="r",
        )

        positions = demultiplex_trajectories_from_reporter(reporter)
        n_frames_raw = positions.shape[1]
        if stride > 1:
            positions = positions[:, ::stride]
        temps = load_temperatures(reporter)

        n_states = positions.shape[0]
        if len(temps) != n_states:
            raise RuntimeError(
                f"Temperature count ({len(temps)}) != state count ({n_states})",
            )

        if lowest_temp_only:
            keep = [int(np.argmin(temps))]
        else:
            keep = list(range(n_states))

        out_subset_dir.mkdir(parents=True, exist_ok=True)
        for i in keep:
            t = temps[i]
            out_path = out_subset_dir / f"{seq}_{t:.1f}_{n_idx}.npz"
            np.savez_compressed(out_path, positions=positions[i].astype(np.float32))

        if save_swap_rates:
            rates = compute_neighbor_swap_rates(reporter)
            swap_path = out_subset_dir / f"{seq}_{n_idx}.swap_rates.txt"
            np.savetxt(swap_path, rates)

        print(
            f"[OK] {run_dir.name} -> {out_subset_dir.name}/{seq}_*_{n_idx}.npz  "
            f"frames={n_frames_raw}  states={n_states}",
        )
        return n_frames_raw

    except Exception as e:
        print(f"[FAIL] {run_dir}: {e}", file=sys.stderr)
        return None

    finally:
        if reporter is not None:
            _safe_close_reporter(reporter)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Demultiplex all REMD runs across REMD_ROOTS into per-temperature .npz arrays.",
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
        help="Only process this subset (alanine/many/xl).",
    )
    p.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Only process runs whose directory name contains this string.",
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

    for root in REMD_ROOTS:
        if not root.exists():
            print(f"# skipping missing root: {root}")
            continue
        tag = subset_tag(root)
        if args.subset and tag != args.subset:
            continue

        runs = find_remd_runs(root)
        if args.sequence:
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
