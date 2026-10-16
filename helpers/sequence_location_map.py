#!/usr/bin/env python3
"""Build a sequence -> canonical REMD run-directory CSV for the UniRef50Chunks
and IDRomeChunks subsequences.

Trajectories for these two sets are scattered across several storage
locations, and a handful of sequences have more than one completed (>= 25 ns)
run. This picks one canonical location per sequence:

    1. Prefer a location that actually reached 25 ns over one that didn't.
    2. Among locations that reached 25 ns, prefer by LOCATION_PRIORITY order
       (oligo-data, then idrome/uniref, then idrome-2/extra -- see below).
    3. Among remaining ties, prefer the higher iteration count.

Progress is read from each run's `remd_real_time_analysis.yaml` (the last
logged '- iteration: N' entry) rather than opening `remd.nc` directly --
opening ~40k netCDF files over the archive filesystem is far slower than
reading the small yaml logs, and the yaml's final entry reliably shows
percent_complete: 100.0 for genuinely complete runs.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

TARGET_ITER = 5000  # 5000 * 2500 steps/iter * 2.0 fs = 25 ns
STEPS_PER_ITERATION = 2500
TIMESTEP_FS = 2.0

ITER_RE = re.compile(rb"^- iteration:\s*(\d+)", re.MULTILINE)

# tag -> remd root directory
LOCATIONS = {
    "uniref": Path("/network/archive/t/tanc/oligo/md-runner-remd-uniref/data/remd"),
    "idrome": Path("/network/archive/t/tanc/oligo/md-runner-idrome/data/remd"),
    "idrome-2": Path("/network/archive/t/tanc/oligo/md-runner-idrome/data/remd-2"),
    "extra": Path("/home/mila/t/tanc/scratch/md-runner-oligopeptides-extra/data/remd"),
    "oligo-data": Path("/network/archive/t/tanc/oligo/data/remd"),
}

# lower rank = more preferred, used only to break ties between locations that
# are equally complete (see module docstring). idrome-2 never uniquely
# completes a sequence the primary idrome dir didn't already finish, so
# idrome is preferred over it to keep the primary/retry split clean.
LOCATION_PRIORITY = {
    "oligo-data": 0,
    "idrome": 1,
    "uniref": 2,
    "idrome-2": 3,
    "extra": 4,
}

CHUNK_CSVS = {
    "uniref50": Path("UniRef50Chunks.csv"),
    "idrome": Path("IDRomeChunks.csv"),
}


def iterations_to_ns(n_iter: int) -> float:
    return (n_iter * STEPS_PER_ITERATION * TIMESTEP_FS) / 1e6


def last_iteration(yaml_path: Path) -> int:
    matches = ITER_RE.findall(yaml_path.read_bytes())
    return int(matches[-1]) if matches else -1


def find_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [d for d in root.iterdir() if d.is_dir() and (d / "remd_real_time_analysis.yaml").exists()]


def load_chunks(path: Path) -> set[str]:
    with open(path) as f:
        return {row["chunk"] for row in csv.DictReader(f)}


def main() -> None:
    # sequence -> list of (location_tag, run_dir, n_iter)
    candidates: dict[str, list[tuple[str, Path, int]]] = {}

    for tag, root in LOCATIONS.items():
        for run_dir in find_run_dirs(root):
            seq = run_dir.name.split("_")[0]
            n_iter = last_iteration(run_dir / "remd_real_time_analysis.yaml")
            candidates.setdefault(seq, []).append((tag, run_dir, n_iter))

    def rank(candidate: tuple[str, Path, int]) -> tuple[bool, int, int]:
        tag, _, n_iter = candidate
        done = n_iter >= TARGET_ITER
        return (not done, LOCATION_PRIORITY.get(tag, 99), -n_iter)

    expected: dict[str, str] = {}
    for source, csv_path in CHUNK_CSVS.items():
        for seq in load_chunks(csv_path):
            expected[seq] = source

    rows = []
    n_missing = 0
    n_ambiguous = 0
    for seq, source in sorted(expected.items()):
        options = candidates.get(seq, [])
        n_done = sum(1 for _, _, n_iter in options if n_iter >= TARGET_ITER)
        if n_done > 1:
            n_ambiguous += 1
        if not options:
            n_missing += 1
            rows.append(
                {
                    "sequence": seq,
                    "source": source,
                    "location": "",
                    "run_dir": "",
                    "iterations": "",
                    "ns": "",
                    "status": "missing",
                    "n_locations_with_25ns": 0,
                }
            )
            continue

        best = min(options, key=rank)
        tag, run_dir, n_iter = best
        status = "done" if n_iter >= TARGET_ITER else "incomplete"
        rows.append(
            {
                "sequence": seq,
                "source": source,
                "location": tag,
                "run_dir": str(run_dir),
                "iterations": n_iter,
                "ns": round(iterations_to_ns(n_iter), 3),
                "status": status,
                "n_locations_with_25ns": n_done,
            }
        )

    out_path = Path("remd_sequence_locations.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "source",
                "location",
                "run_dir",
                "iterations",
                "ns",
                "status",
                "n_locations_with_25ns",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    n_done_total = sum(1 for r in rows if r["status"] == "done")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  done:       {n_done_total}")
    print(f"  incomplete: {sum(1 for r in rows if r['status'] == 'incomplete')}")
    print(f"  missing:    {n_missing}")
    print(f"  sequences with >1 location reaching 25ns (resolved via priority): {n_ambiguous}")


if __name__ == "__main__":
    main()
