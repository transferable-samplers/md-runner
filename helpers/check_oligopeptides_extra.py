#!/usr/bin/env python3
"""
check_oligopeptides_extra.py

Adapted from check_all.py for the oligopeptides "extra" (gap-fill) batch:
same done/not_done classification (>= TARGET_NS iterations in remd.nc), but
also cross-references against sequences/missing_pdbs.txt so sequences with
no run directory at all (not yet started) are reported separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

from openmmtools import multistate

# ---------------- CONFIG ----------------
TARGET_NS = 25

# MUST match sbatch/run_oligopeptides_extra.sh / configs/generate_remd.yaml
TIMESTEP_FS = 2.0
STEPS_PER_ITERATION = 2500

REMD_ROOT = Path("/network/scratch/t/tanc/md-runner-oligopeptides-extra/data/remd")
SEQ_FILE = Path("sequences/missing_pdbs.txt")
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


def read_sequences(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def main() -> None:
    remd_root = Path(sys.argv[1]) if len(sys.argv) > 1 else REMD_ROOT
    seq_file = Path(sys.argv[2]) if len(sys.argv) > 2 else SEQ_FILE

    if not remd_root.exists():
        print(f"Error: {remd_root} does not exist", file=sys.stderr)
        sys.exit(1)

    expected = read_sequences(seq_file)
    expected_set = set(expected)

    done: list[tuple[str, float]] = []
    not_done: list[tuple[str, float]] = []
    seen_sequences: set[str] = set()

    for run in find_remd_runs(remd_root):
        seq = run.name.split("_")[0]
        seen_sequences.add(seq)
        is_done, time_ns = classify_run(run)
        if is_done:
            done.append((seq, time_ns))
        else:
            not_done.append((seq, time_ns))

    not_started = sorted(expected_set - seen_sequences)

    with open("done.txt", "w") as f:
        for name, time_ns in done:
            f.write(f"{name}  {time_ns:.1f} ns\n")

    with open("not_done.txt", "w") as f:
        for name, time_ns in not_done:
            f.write(f"{name}  {time_ns:.1f} ns\n")

    with open("not_started.txt", "w") as f:
        for name in not_started:
            f.write(f"{name}\n")

    print(f"Expected sequences ({seq_file}): {len(expected)}")
    print(f"Done (>= {TARGET_NS} ns):        {len(done)}")
    print(f"Not done (started, < {TARGET_NS} ns): {len(not_done)}")
    print(f"Not started (no run dir):     {len(not_started)}")
    print("Wrote done.txt, not_done.txt, not_started.txt")


if __name__ == "__main__":
    main()
