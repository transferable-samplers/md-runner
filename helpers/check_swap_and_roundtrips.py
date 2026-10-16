#!/usr/bin/env python3
"""
check_swap_and_roundtrips.py

For finished REMD runs (>= TARGET_NS), report:
  - neighbor-state swap acceptance rates (recomputed from remd.nc, same definition
    as save_swap_rates() in src/generate_remd.py)
  - number of round trips per replica between the coldest (state 0) and hottest
    (state n_states-1) rungs of the temperature ladder -- a standard REMD mixing
    diagnostic (Katzgraber, Trebst, Huse & Troyer 2006): a replica is labeled "up"
    on touching state 0 and "down" on touching state n_states-1; a round trip is
    counted each time an "up"-labeled replica returns to state 0 after having
    reached the top (i.e. one full bottom -> top -> bottom cycle).

Writes swap_roundtrips.csv with one row per finished run.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from openmmtools import multistate

# ---------------- CONFIG ----------------
TARGET_NS = 25

# MUST match the run's config
TIMESTEP_FS = 2.0
STEPS_PER_ITERATION = 2500

REMD_ROOT = Path("/network/scratch/t/tanc/md-runner-oligopeptides-extra/data/remd")
OUT_CSV = Path("swap_roundtrips.csv")
# ----------------------------------------


def find_remd_runs(remd_root: Path) -> list[Path]:
    return sorted(
        d for d in remd_root.iterdir()
        if d.is_dir()
        and (d / "remd.nc").exists()
        and (d / "remd_checkpoint.nc").exists()
    )


def iterations_to_ns(n_iter: int) -> float:
    return (n_iter * STEPS_PER_ITERATION * TIMESTEP_FS) / 1e6


def count_round_trips(state_traj: np.ndarray, n_states: int) -> int:
    """Count bottom -> top -> bottom cycles for a single replica's state trajectory."""
    label = None
    count = 0
    for s in state_traj:
        if s == 0:
            if label == "down":
                count += 1
            label = "up"
        elif s == n_states - 1:
            if label == "up":
                label = "down"
    return count


def analyze_run(run_dir: Path) -> dict | None:
    reporter = None
    try:
        reporter = multistate.MultiStateReporter(
            str(run_dir / "remd.nc"),
            checkpoint_storage=str(run_dir / "remd_checkpoint.nc"),
            open_mode="r",
        )
        analysis = reporter._storage_analysis
        n_iter = analysis.variables["states"].shape[0]
        time_ns = iterations_to_ns(n_iter)
        if time_ns < TARGET_NS:
            return None

        accepted = np.array(analysis.variables["accepted"][:])  # (iter, state_i, state_j)
        proposed = np.array(analysis.variables["proposed"][:])
        n_states = accepted.shape[1]

        swap_rates = np.zeros(n_states - 1)
        for i in range(n_states - 1):
            n_acc = accepted[:, i, i + 1].sum()
            n_prop = proposed[:, i, i + 1].sum()
            swap_rates[i] = n_acc / n_prop if n_prop > 0 else 0.0

        all_states = np.array(analysis.variables["states"][:])  # (n_iter, n_replicas)
        n_replicas = all_states.shape[1]
        round_trips = [
            count_round_trips(all_states[:, r], n_states) for r in range(n_replicas)
        ]

        return {
            "run": run_dir.name,
            "sequence": run_dir.name.split("_")[0],
            "time_ns": time_ns,
            "n_states": n_states,
            "mean_swap_rate": float(swap_rates.mean()),
            "min_swap_rate": float(swap_rates.min()),
            "swap_rates": swap_rates,
            "round_trips": round_trips,
            "mean_round_trips": float(np.mean(round_trips)),
            "min_round_trips": int(np.min(round_trips)),
            "total_round_trips": int(np.sum(round_trips)),
        }

    except Exception as e:
        print(f"  ! failed on {run_dir.name}: {e}", file=sys.stderr)
        return None

    finally:
        if reporter is not None:
            try:
                reporter._storage_analysis.close()
                reporter._storage_checkpoint.close()
                reporter._storage.close()
            except Exception:
                pass


def main() -> None:
    remd_root = Path(sys.argv[1]) if len(sys.argv) > 1 else REMD_ROOT
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_CSV

    if not remd_root.exists():
        print(f"Error: {remd_root} does not exist", file=sys.stderr)
        sys.exit(1)

    runs = find_remd_runs(remd_root)
    print(f"Scanning {len(runs)} run directories under {remd_root}...")

    results = []
    max_pairs = 0
    for run in runs:
        r = analyze_run(run)
        if r is not None:
            results.append(r)
            max_pairs = max(max_pairs, len(r["swap_rates"]))

    header = [
        "sequence", "run", "time_ns", "n_states", "n_replicas",
        "mean_swap_rate", "min_swap_rate",
        "mean_round_trips", "min_round_trips", "total_round_trips",
    ] + [f"swap_rate_{i}" for i in range(max_pairs)] + [f"round_trips_replica_{i}" for i in range(max_pairs + 1)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in results:
            row = [
                r["sequence"], r["run"], f"{r['time_ns']:.1f}", r["n_states"], len(r["round_trips"]),
                f"{r['mean_swap_rate']:.4f}", f"{r['min_swap_rate']:.4f}",
                f"{r['mean_round_trips']:.2f}", r["min_round_trips"], r["total_round_trips"],
            ]
            row += [f"{x:.4f}" for x in r["swap_rates"]] + [""] * (max_pairs - len(r["swap_rates"]))
            row += list(r["round_trips"]) + [""] * (max_pairs + 1 - len(r["round_trips"]))
            w.writerow(row)

    print(f"Analyzed {len(results)} finished runs (>= {TARGET_NS} ns)")
    if results:
        mean_swap = np.mean([r["mean_swap_rate"] for r in results])
        mean_rt = np.mean([r["mean_round_trips"] for r in results])
        n_zero_rt = sum(1 for r in results if r["min_round_trips"] == 0)
        print(f"  overall mean swap rate:      {mean_swap:.3f}")
        print(f"  overall mean round trips:    {mean_rt:.2f} per replica")
        print(f"  runs with >=1 replica stuck (0 round trips): {n_zero_rt}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
