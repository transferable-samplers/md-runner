#!/usr/bin/env python3
"""Demultiplex all REMD runs into per-temperature trajectories.

Output: one `trajectories.npz` per run directory (or in --out-dir if specified).
Location: /network/scratch/t/tanc/md-runner-short/data/remd/<run_name>/trajectories.npz

NPZ keys:
  "temperatures"        - float32 array of temperatures in Kelvin, shape (n_states,)
  "{T}_positions"       - float32 positions in nm, shape (n_frames, n_atoms, 3)
  "{T}_velocities"      - float32 velocities in nm/ps, shape (n_frames, n_atoms, 3)

  where {T} is the temperature formatted to 1 decimal place, e.g. "300.0", "336.0".

Example:
  data = np.load("trajectories.npz")
  temps = data["temperatures"]           # [300.0, 336.0, 377.0, 450.0]
  pos = data["300.0_positions"]          # (n_frames, n_atoms, 3)

Frame count to simulation time: n_frames * timestep_fs * frame_interval / 1e6 = ns
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from openmmtools import multistate


def find_remd_runs(remd_root: Path) -> list[Path]:
    """Return run directories containing both remd.nc and remd_checkpoint.nc."""
    runs = []
    for d in remd_root.iterdir():
        if not d.is_dir():
            continue
        nc_path = d / "remd.nc"
        ckpt_path = d / "remd_checkpoint.nc"
        if nc_path.exists() and ckpt_path.exists():
            runs.append(d)
    return sorted(runs)


def demultiplex_trajectories_from_reporter(
    reporter: multistate.MultiStateReporter,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Demultiplex using checkpoint positions/velocities and analysis state assignments.

    Returns:
      positions: (n_states, n_ckpt, n_atoms, 3) float32
      velocities: (n_states, n_ckpt, n_atoms, 3) float32
    """
    checkpoint = reporter._storage_checkpoint
    analysis = reporter._storage_analysis
    checkpoint_interval = int(checkpoint.CheckpointInterval)

    positions = np.array(checkpoint.variables["positions"][:])  # (n_ckpt, replica, atom, 3)
    velocities = np.array(checkpoint.variables["velocities"][:])  # (n_ckpt, replica, atom, 3)
    all_states = np.array(analysis.variables["states"][:])  # (n_iter, replica)

    n_ckpt, n_replicas, n_atoms, _ = positions.shape

    ckpt_indices = np.arange(n_ckpt) * checkpoint_interval
    if ckpt_indices[-1] >= all_states.shape[0]:
        raise RuntimeError(
            f"Checkpoint has {n_ckpt} frames at interval={checkpoint_interval}, "
            f"but analysis only has {all_states.shape[0]} iterations. "
            f"Last ckpt index {ckpt_indices[-1]} is out of range.",
        )

    states_at_ckpt = all_states[ckpt_indices]  # (n_ckpt, replica)

    demux_pos = np.zeros((n_replicas, n_ckpt, n_atoms, 3), dtype=np.float32)
    demux_vel = np.zeros((n_replicas, n_ckpt, n_atoms, 3), dtype=np.float32)

    for frame in range(n_ckpt):
        order = np.argsort(states_at_ckpt[frame])
        demux_pos[:, frame] = positions[frame, order]
        demux_vel[:, frame] = velocities[frame, order]

    return demux_pos, demux_vel


def compute_neighbor_swap_rates(reporter: multistate.MultiStateReporter) -> np.ndarray:
    """Compute neighbor swap acceptance rates (state i <-> i+1) over whole run."""
    analysis = reporter._storage_analysis
    accepted = np.array(analysis.variables["accepted"][:])  # (iter, state_i, state_j)
    proposed = np.array(analysis.variables["proposed"][:])

    n_states = accepted.shape[1]
    rates = np.zeros(n_states - 1, dtype=np.float64)
    for i in range(n_states - 1):
        n_acc = accepted[:, i, i + 1].sum()
        n_prop = proposed[:, i, i + 1].sum()
        rates[i] = (n_acc / n_prop) if n_prop > 0 else 0.0
    return rates


def load_temperatures(reporter: multistate.MultiStateReporter) -> np.ndarray:
    """Extract temperatures (in Kelvin) from the reporter's thermodynamic states."""
    from simtk import unit

    states = reporter.read_thermodynamic_states()[0]
    temps = np.array([s.temperature.value_in_unit(unit.kelvin) for s in states], dtype=np.float32)
    return temps


def _safe_close_reporter(reporter: multistate.MultiStateReporter) -> None:
    """Close underlying netcdf files to avoid descriptor leaks / weird crashes."""
    for attr in ["_storage_analysis", "_storage_checkpoint", "_storage"]:
        obj = getattr(reporter, attr, None)
        try:
            if obj is not None and hasattr(obj, "close"):
                obj.close()
        except Exception:
            pass


def process_run_dir(run_dir: Path, out_dir: Optional[Path], save_swap_rates: bool) -> bool:
    nc_path = run_dir / "remd.nc"
    ckpt_path = run_dir / "remd_checkpoint.nc"
    reporter = None

    try:
        reporter = multistate.MultiStateReporter(
            str(nc_path),
            checkpoint_storage=str(ckpt_path),
            open_mode="r",
        )

        positions, velocities = demultiplex_trajectories_from_reporter(reporter)
        temps = load_temperatures(reporter)

        n_states = positions.shape[0]
        if len(temps) != n_states:
            raise RuntimeError(
                f"Temperature count ({len(temps)}) != state count ({n_states})",
            )

        if out_dir is None:
            out_path = run_dir / "trajectories.npz"
            swap_path = run_dir / "swap_rates.txt"
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_dir.name}.trajectories.npz"
            swap_path = out_dir / f"{run_dir.name}.swap_rates.txt"

        # Keys: "temperatures", "{T}_positions", "{T}_velocities"
        data = {"temperatures": temps.astype(np.float32)}
        for i, t in enumerate(temps):
            key = f"{t:.1f}"
            data[f"{key}_positions"] = positions[i]  # (n_ckpt, n_atoms, 3)
            data[f"{key}_velocities"] = velocities[i]  # (n_ckpt, n_atoms, 3)
        np.savez_compressed(out_path, **data)

        if save_swap_rates:
            rates = compute_neighbor_swap_rates(reporter)
            np.savetxt(swap_path, rates)

        print(f"[OK] {run_dir.name} -> {out_path.name}  pos={positions.shape} vel={velocities.shape}")
        return True

    except Exception as e:
        print(f"[FAIL] {run_dir}: {e}", file=sys.stderr)
        return False

    finally:
        if reporter is not None:
            _safe_close_reporter(reporter)


def main() -> None:
    p = argparse.ArgumentParser(description="Demultiplex all REMD runs under a remd/ root directory (SERIAL).")
    p.add_argument("--remd-root", type=Path, required=True, help="Path to .../data/remd directory")
    p.add_argument("--out-dir", type=Path, default=None, help="If set, write outputs here instead of per-run dir")
    p.add_argument(
        "--save-swap-rates",
        action="store_true",
        help="Also compute neighbor swap rates and write swap_rates.txt",
    )
    args = p.parse_args()

    remd_root: Path = args.remd_root
    if not remd_root.exists():
        print(f"Error: remd root does not exist: {remd_root}", file=sys.stderr)
        sys.exit(1)

    runs = find_remd_runs(remd_root)
    if not runs:
        print(f"Error: no run dirs with remd.nc + remd_checkpoint.nc found in {remd_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(runs)} REMD run directories under {remd_root}")

    ok = 0
    bad = 0
    for i, d in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] Processing {d.name}")
        if process_run_dir(d, args.out_dir, args.save_swap_rates):
            ok += 1
        else:
            bad += 1

    print(f"\nSummary: {ok} succeeded, {bad} failed")


if __name__ == "__main__":
    main()
