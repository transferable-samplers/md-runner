#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from openmmtools import multistate

TARGET_NS = 25
TIMESTEP_FS = 2.0
STEPS_PER_ITERATION = 2500

REMD_ROOTS = [
    Path("/network/scratch/t/tanc/md-runner-remd-reference-alanine/data/remd"),
    Path("/network/scratch/t/tanc/md-runner-remd-reference-many/data/remd"),
    Path("/network/scratch/t/tanc/md-runner-remd-reference-xl/data/remd"),
]


def find_remd_runs(remd_root: Path) -> list[Path]:
    return sorted(
        d
        for d in remd_root.iterdir()
        if d.is_dir() and (d / "remd.nc").exists() and (d / "remd_checkpoint.nc").exists()
    )


def iterations_to_ns(n_iter: int) -> float:
    return (n_iter * STEPS_PER_ITERATION * TIMESTEP_FS) / 1e6


def run_time_ns(run_dir: Path) -> float:
    reporter = None
    try:
        reporter = multistate.MultiStateReporter(
            str(run_dir / "remd.nc"),
            checkpoint_storage=str(run_dir / "remd_checkpoint.nc"),
            open_mode="r",
        )
        analysis = reporter._storage_analysis
        n_iter = analysis.variables["states"].shape[0]
        return iterations_to_ns(n_iter)
    except Exception:
        return float("nan")
    finally:
        if reporter is not None:
            try:
                reporter._storage_analysis.close()
                reporter._storage_checkpoint.close()
                reporter._storage.close()
            except Exception:
                pass


def replica_label(name: str) -> str:
    # e.g. "AA_300K-450K_4_1.0_5000_scramble2" -> "scramble2"
    #      "AA_300K-450K_4_1.0_5000"           -> "original"
    if "_scramble" in name:
        return "scramble" + name.rsplit("_scramble", 1)[1]
    return "original"


def replica_sort_key(label: str) -> tuple[int, int]:
    if label == "original":
        return (0, 0)
    return (1, int(label.replace("scramble", "")))


def main() -> None:
    # tag -> seq -> list of (replica_label, time_ns, full_name)
    by_tag: dict[str, dict[str, list[tuple[str, float, str]]]] = defaultdict(
        lambda: defaultdict(list),
    )

    for root in REMD_ROOTS:
        if not root.exists():
            print(f"# skipping missing root: {root}")
            continue
        tag = root.parents[1].name.replace("md-runner-remd-reference-", "")
        for run in find_remd_runs(root):
            seq = run.name.split("_")[0]
            label = replica_label(run.name)
            by_tag[tag][seq].append((label, run_time_ns(run), run.name))

    for tag in [r.parents[1].name.replace("md-runner-remd-reference-", "") for r in REMD_ROOTS]:
        if tag not in by_tag:
            continue
        print(f"\n=== {tag} ===")
        for seq in sorted(by_tag[tag], key=lambda s: (len(s), s)):
            runs = sorted(by_tag[tag][seq], key=lambda r: replica_sort_key(r[0]))
            n_done = sum(1 for _, t, _ in runs if t >= TARGET_NS)
            print(f"\n{seq}  ({n_done}/{len(runs)} >= {TARGET_NS} ns)")
            for label, time_ns, _ in runs:
                mark = "x" if time_ns >= TARGET_NS else " "
                print(f"  [{mark}] {label:11s} {time_ns:6.1f} ns")


if __name__ == "__main__":
    main()
