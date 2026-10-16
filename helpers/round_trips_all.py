#!/usr/bin/env python3
"""Round trips per replica for every run in remd_sequence_locations.csv.

Same round-trip definition as helpers/check_swap_and_roundtrips.py (one full
state 0 -> state n_states-1 -> state 0 cycle), but reads only the `states`
variable of remd.nc with netCDF4 rather than opening a MultiStateReporter, so
it is fast enough to sweep all ~36k runs on the archive filesystem.

Appends to round_trips_all.csv as it goes and skips sequences already there,
so it can be re-run after an interruption.
"""
from __future__ import annotations

import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import netCDF4
import numpy as np

LOCATIONS_CSV = Path(sys.argv[2] if len(sys.argv) > 2 else "remd_sequence_locations.csv")
OUT_CSV = Path(sys.argv[3] if len(sys.argv) > 3 else "round_trips_all.csv")
FIELDS = ["sequence", "n_states", "n_iter", "mean_round_trips", "min_round_trips", "max_round_trips", "round_trips"]


def count_round_trips(state_traj: np.ndarray, n_states: int) -> int:
    """Count bottom -> top -> bottom cycles for a single replica's state trajectory."""
    # Only visits to the two end states matter; collapse consecutive repeats and
    # count each top visit that is followed by a return to the bottom.
    ends = state_traj[(state_traj == 0) | (state_traj == n_states - 1)]
    if ends.size == 0:
        return 0
    ends = ends[np.r_[True, ends[1:] != ends[:-1]]]
    first_bottom = np.argmax(ends == 0) if (ends == 0).any() else ends.size
    ends = ends[first_bottom:]
    return int((ends == 0).sum() - 1) if ends.size else 0


def analyze(row: dict) -> dict | None:
    try:
        with netCDF4.Dataset(Path(row["run_dir"]) / "remd.nc") as ds:
            states = np.array(ds.variables["states"][:])  # (n_iter, n_replicas)
    except Exception as e:
        print(f"  ! failed on {row['sequence']}: {e}", file=sys.stderr)
        return None
    n_states = states.shape[1]
    trips = [count_round_trips(states[:, r], n_states) for r in range(n_states)]
    return {
        "sequence": row["sequence"],
        "n_states": n_states,
        "n_iter": states.shape[0] - 1,
        "mean_round_trips": f"{np.mean(trips):.3f}",
        "min_round_trips": min(trips),
        "max_round_trips": max(trips),
        "round_trips": " ".join(map(str, trips)),
    }


def main() -> None:
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    with open(LOCATIONS_CSV) as f:
        rows = [r for r in csv.DictReader(f) if r["run_dir"]]
    done = set()
    if OUT_CSV.exists():
        with open(OUT_CSV) as f:
            done = {r["sequence"] for r in csv.DictReader(f)}
    todo = [r for r in rows if r["sequence"] not in done]
    print(f"{len(done)} already done, {len(todo)} to do", flush=True)

    with open(OUT_CSV, "a", newline="") as f, Pool(n_workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for i, result in enumerate(pool.imap_unordered(analyze, todo, chunksize=8), 1):
            if result is not None:
                writer.writerow(result)
            if i % 500 == 0:
                f.flush()
                print(f"{i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
