#!/usr/bin/env python3
"""Swap acceptance rate vs system size (residues and atoms) for the oligo REMD runs.

Reads swap_rates_all.csv (written by helpers/swap_rate_histogram.py), counts
atoms in each sequence's input PDB, and writes swap_rates_size.csv plus
swap_rate_vs_size.png.
"""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATES_CSV = Path("swap_rates_all.csv")
OUT_CSV = Path("swap_rates_size.csv")
OUT_PNG = Path("swap_rate_vs_size.png")

PDB_DIRS = [
    Path("/network/archive/t/tanc/oligo/data/pdbs"),
    Path("/network/archive/t/tanc/oligo/md-runner-idrome/data/pdbs"),
    Path("/home/mila/t/tanc/scratch/md-runner-oligopeptides-extra/data/pdbs"),
    Path("/network/archive/t/tanc/oligo/md-runner-remd-uniref/data/pdbs"),
]


def count_atoms(sequence: str) -> int:
    for d in PDB_DIRS:
        path = d / f"{sequence}.pdb"
        try:
            return sum(1 for line in path.read_bytes().splitlines() if line.startswith((b"ATOM", b"HETATM")))
        except OSError:
            continue
    return -1


def load() -> list[dict]:
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            return list(csv.DictReader(f))
    with open(RATES_CSV) as f:
        rows = list(csv.DictReader(f))
    with ThreadPoolExecutor(max_workers=32) as pool:
        n_atoms = list(pool.map(count_atoms, [r["sequence"] for r in rows]))
    for r, n in zip(rows, n_atoms):
        r["length"] = len(r["sequence"])
        r["n_atoms"] = n
    print(f"PDB not found for {sum(1 for n in n_atoms if n < 0)} / {len(rows)} sequences")
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    rows = [r for r in load() if int(r["n_atoms"]) > 0]
    length = np.array([int(r["length"]) for r in rows])
    n_states = np.array([int(r["n_states"]) for r in rows])
    n_atoms = np.array([int(r["n_atoms"]) for r in rows])
    mean_rate = np.array([float(r["mean_rate"]) for r in rows])
    min_rate = np.array([float(r["min_rate"]) for r in rows])

    print(f"{'len':>4} {'n_st':>4} {'runs':>6} {'atoms p5-p95':>13} {'mean':>6} {'p5':>6} {'p95':>6} {'min-pair p5':>11} {'r(atoms)':>8}")
    for L in sorted(set(length)):
        for n in sorted(set(n_states[length == L])):
            m = (length == L) & (n_states == n)
            a, x = n_atoms[m], mean_rate[m]
            r = np.corrcoef(a, x)[0, 1] if m.sum() > 2 else float("nan")
            print(
                f"{L:>4} {n:>4} {m.sum():>6} {np.percentile(a, 5):>6.0f}-{np.percentile(a, 95):<6.0f} "
                f"{x.mean():>6.3f} {np.percentile(x, 5):>6.3f} {np.percentile(x, 95):>6.3f} "
                f"{np.percentile(min_rate[m], 5):>11.3f} {r:>8.2f}"
            )

    print("R^2 of mean rate at fixed n_states: linear in n_atoms vs linear in length")
    for n in sorted(set(n_states)):
        m = n_states == n
        r2_atoms = np.corrcoef(n_atoms[m], mean_rate[m])[0, 1] ** 2
        r2_len = np.corrcoef(length[m], mean_rate[m])[0, 1] ** 2
        print(f"  n_states={n}: atoms {r2_atoms:.3f}  length {r2_len:.3f}  (runs {m.sum()})")

    ink, muted = "#333333", "#777777"
    fig, (ax_len, ax_atoms) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    # Left: one interval per (length, n_states) pair; each length has a lower and a higher replica count.
    pair_colors = {"lower": "#2a78d6", "higher": "#eb6834"}
    for L in sorted(set(length)):
        options = sorted(set(n_states[length == L]))
        for n in options:
            which = "lower" if n == options[0] else "higher"
            x = L + (-0.17 if which == "lower" else 0.17)
            v = mean_rate[(length == L) & (n_states == n)]
            lo, hi = np.percentile(v, [5, 95])
            ax_len.plot([x, x], [lo, hi], color=pair_colors[which], linewidth=2, solid_capstyle="round")
            ax_len.plot(x, v.mean(), "o", color=pair_colors[which], markersize=6, markeredgecolor="white", markeredgewidth=1)
            ax_len.annotate(str(n), (x, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=ink)
    for which, label in [("lower", "lower replica count for that length"), ("higher", "higher replica count")]:
        ax_len.plot([], [], "o-", color=pair_colors[which], linewidth=2, label=label)
    ax_len.legend(frameon=False, fontsize=9, loc="lower right")
    ax_len.set_xticks(sorted(set(length)))
    ax_len.set_xlabel("Sequence length (residues)")
    ax_len.set_ylabel("Mean swap acceptance rate of run")
    ax_len.set_title("By (length, replica count): mean and 5th-95th percentile\nnumber above each bar = replicas", fontsize=10, loc="left")

    # Right: every run against its atom count; replica count is ordinal, so one hue, light to dark.
    ramp = dict(zip(sorted(set(n_states)), ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]))
    for n in sorted(set(n_states)):
        m = n_states == n
        ax_atoms.scatter(n_atoms[m], mean_rate[m], s=3, color=ramp[n], alpha=0.35, linewidths=0, rasterized=True)
        ax_atoms.annotate(
            f"{n} replicas", (np.percentile(n_atoms[m], 99.5), np.percentile(mean_rate[m], 1)),
            xytext=(6, -8), textcoords="offset points", fontsize=9, color=ink,
        )
    ax_atoms.set_xlim(right=n_atoms.max() + 45)
    ax_atoms.set_xlabel("Number of atoms")
    ax_atoms.set_title("Every run against its atom count\nshade = replica count", fontsize=10, loc="left")

    for ax in (ax_len, ax_atoms):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted)
    fig.suptitle("REMD swap acceptance vs system size, 300-450 K ladder", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
