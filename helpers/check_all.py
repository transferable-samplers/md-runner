#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

from openmmtools import multistate

# ---------------- CONFIG ----------------
TARGET_NS = 25

# MUST match your simulation setup
TIMESTEP_FS = 2.0
STEPS_PER_ITERATION = 2500
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


def classify_run(run_dir: Path) -> tuple[bool, float]:
    """
    Returns:
        (done, time_ns)
    """
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

        return time_ns >= TARGET_NS, time_ns

    except Exception:
        return False, float("nan")

    finally:
        if reporter is not None:
            try:
                reporter._storage_analysis.close()
                reporter._storage_checkpoint.close()
                reporter._storage.close()
            except Exception:
                pass


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: check_whats_done.py /path/to/data/remd", file=sys.stderr)
        sys.exit(1)

    remd_root = Path(sys.argv[1])
    if not remd_root.exists():
        print(f"Error: {remd_root} does not exist", file=sys.stderr)
        sys.exit(1)

    done = []
    not_done = []

    for run in find_remd_runs(remd_root):
        is_done, time_ns = classify_run(run)
        if is_done:
            done.append((run.name, time_ns))
        else:
            not_done.append((run.name, time_ns))

    # Write outputs next to where script is run
    with open("done.txt", "w") as f:
        for name, time_ns in done:
            f.write(f"{name}  {time_ns:.1f} ns\n")

    with open("not_done.txt", "w") as f:
        for name, time_ns in not_done:
            f.write(f"{name.split('_')[0]}  {time_ns:.1f} ns\n")


if __name__ == "__main__":
    main()
