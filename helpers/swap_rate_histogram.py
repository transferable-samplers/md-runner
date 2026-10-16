#!/usr/bin/env python3
"""Histogram of neighbor swap acceptance rates across the oligo REMD runs.

Uses remd_sequence_locations.csv (written by helpers/sequence_location_map.py)
to find each sequence's canonical run directory and reads the `swap_rates.txt`
that save_swap_rates() in src/generate_remd.py wrote there (one rate per
neighboring state pair).

Writes swap_rates_all.csv (one row per sequence) and swap_rate_hist.png.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOCATIONS_CSV = Path("remd_sequence_locations.csv")
OUT_CSV = Path("swap_rates_all.csv")
OUT_PNG = Path("swap_rate_hist.png")


def read_rates(row: dict) -> tuple[dict, np.ndarray | None]:
    path = Path(row["run_dir"]) / "swap_rates.txt"
    try:
        return row, np.atleast_1d(np.loadtxt(path))
    except (OSError, ValueError):
        return row, None


def main() -> None:
    with open(LOCATIONS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["run_dir"]]

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(read_rates, rows))

    found = [(row, rates) for row, rates in results if rates is not None and rates.size > 0]
    missing = [row for row, rates in results if rates is None or rates.size == 0]
    print(f"Runs in {LOCATIONS_CSV}: {len(rows)}")
    print(f"  with swap_rates.txt: {len(found)}")
    print(f"  missing/unreadable:  {len(missing)}")
    for tag in sorted({r["location"] for r in missing}):
        print(f"    {tag}: {sum(1 for r in missing if r['location'] == tag)}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "source", "location", "n_states", "mean_rate", "min_rate", "max_rate", "rates"])
        for row, rates in found:
            writer.writerow(
                [
                    row["sequence"],
                    row["source"],
                    row["location"],
                    len(rates) + 1,
                    f"{rates.mean():.4f}",
                    f"{rates.min():.4f}",
                    f"{rates.max():.4f}",
                    " ".join(f"{r:.4f}" for r in rates),
                ]
            )

    all_rates = np.concatenate([rates for _, rates in found])
    min_rates = np.array([rates.min() for _, rates in found])
    for name, x in [("all neighbor pairs", all_rates), ("per-run minimum", min_rates)]:
        q = np.percentile(x, [1, 5, 50, 95, 99])
        print(
            f"{name}: n={x.size} mean={x.mean():.3f} min={x.min():.3f} max={x.max():.3f} "
            f"p1/p5/p50/p95/p99={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}/{q[3]:.3f}/{q[4]:.3f}"
        )

    bins = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    panels = [
        (all_rates, f"All neighbor pairs (n = {all_rates.size:,})", "Neighbor pairs"),
        (min_rates, f"Lowest pair per run (n = {min_rates.size:,} runs)", "Runs"),
    ]
    for ax, (x, title, ylabel) in zip(axes, panels):
        ax.hist(x, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(x), color="#333333", linewidth=1, linestyle="--")
        ax.annotate(
            f"median {np.median(x):.2f}",
            xy=(np.median(x), 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -12),
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("Swap acceptance rate")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("REMD neighbor swap acceptance rates, UniRef50 + IDRome chunks", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_CSV} and {OUT_PNG}")


if __name__ == "__main__":
    main()
