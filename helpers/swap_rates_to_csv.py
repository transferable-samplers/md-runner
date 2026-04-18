#!/usr/bin/env python3
"""Aggregate all *.swap_rates.txt files in a directory into a single CSV.

Each input file contains a 1D array of length (n_replicas - 1) — the neighbor
exchange acceptance rates produced by helpers/demux_all.py.

Output columns: sequence, n_replicas, rate_0, rate_1, ..., rate_{N-2}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True, help="Directory containing *.swap_rates.txt files")
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    args = p.parse_args()

    files = sorted(args.in_dir.glob("*.swap_rates.txt"))
    if not files:
        raise SystemExit(f"No *.swap_rates.txt files found in {args.in_dir}")

    rows = []
    max_pairs = 0
    for f in files:
        sequence = f.name[: -len(".swap_rates.txt")]
        rates = np.atleast_1d(np.loadtxt(f)).astype(float)
        n_pairs = len(rates)
        max_pairs = max(max_pairs, n_pairs)
        rows.append((sequence, n_pairs + 1, rates))

    header = ["sequence", "n_replicas"] + [f"rate_{i}" for i in range(max_pairs)]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for sequence, n_replicas, rates in rows:
            row = [sequence, n_replicas] + [f"{r:.6f}" for r in rates]
            row += [""] * (max_pairs - len(rates))
            w.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
