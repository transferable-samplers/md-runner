"""Post-process a hot-MD run produced by generate_md.py (see sbatch/run_hot_md.sh) to
measure the empirical thermal spread of the chirality/omega flat-bottom restraint dihedrals.

Used to size chirality_tol_deg / omega_tol_deg (and sanity-check the force constants) in
configs/generate_remd.yaml against real thermal fluctuations, rather than a sqrt(T)
extrapolation guess. Runs on CPU against the saved trajectory chunks -- no GPU/sbatch needed.
"""

import argparse
from pathlib import Path

import numpy as np
import rootutils
from openmm import unit
from openmm.app import PDBFile

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.generate_remd import _dihedral_angle, _residue_atom_indices  # noqa: E402


def build_torsion_list(topology):
    """Same atom-selection logic as add_chirality_restraints/add_omega_restraints in
    generate_remd.py, but just collecting the atom quads to track rather than restraining them.
    """
    chirality = []
    for residue in topology.residues():
        if residue.name == "GLY":
            continue
        atoms = _residue_atom_indices(residue)
        if not {"N", "CA", "C", "CB"} <= atoms.keys():
            continue
        # Atom order (CA, N, C, CB)/(CA, N, C, HA) matches the (Cα, N, C, Cβ)/(Cα, N, C, Hα)
        # improper convention.
        chirality.append(("CA", residue.index, (atoms["CA"], atoms["N"], atoms["C"], atoms["CB"])))
        if "HA" in atoms:
            chirality.append(("HA", residue.index, (atoms["CA"], atoms["N"], atoms["C"], atoms["HA"])))
        if residue.name == "THR" and {"OG1", "CG2"} <= atoms.keys():
            chirality.append(("THR_CB", residue.index, (atoms["CB"], atoms["CA"], atoms["OG1"], atoms["CG2"])))
            if "HB" in atoms:
                chirality.append(("THR_HB", residue.index, (atoms["CB"], atoms["CA"], atoms["OG1"], atoms["HB"])))
        elif residue.name == "ILE" and {"CG1", "CG2"} <= atoms.keys():
            chirality.append(("ILE_CB", residue.index, (atoms["CB"], atoms["CA"], atoms["CG1"], atoms["CG2"])))
            if "HB" in atoms:
                chirality.append(("ILE_HB", residue.index, (atoms["CB"], atoms["CA"], atoms["CG1"], atoms["HB"])))

    omega = []
    for chain in topology.chains():
        residues = list(chain.residues())
        for i in range(len(residues) - 1):
            res_i, res_j = residues[i], residues[i + 1]
            if res_j.name == "PRO":
                continue
            atoms_i = _residue_atom_indices(res_i)
            atoms_j = _residue_atom_indices(res_j)
            if not ({"CA", "C"} <= atoms_i.keys() and {"N", "CA"} <= atoms_j.keys()):
                continue
            omega.append(("omega", res_i.index, (atoms_i["CA"], atoms_i["C"], atoms_j["N"], atoms_j["CA"])))

    return chirality + omega


def circular_mean_std(degrees):
    rad = np.radians(degrees)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    mean = np.degrees(np.arctan2(mean_sin, mean_cos))
    resultant = np.sqrt(mean_sin**2 + mean_cos**2)
    std = np.degrees(np.sqrt(-2 * np.log(resultant))) if resultant > 0 else float("nan")
    return mean, std


def _dihedral_angle_batch(p0, p1, p2, p3):
    """Vectorized version of _dihedral_angle over a batch of frames, shape (n, 3) per arg."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def compute_torsion_series(md_output_dir: Path, torsions, n_samples: int = 1_000_000) -> dict:
    """Stream chunk_*.npz files one at a time, computing dihedral angles per sampled frame
    immediately rather than accumulating raw positions -- peak memory is one chunk's worth of
    positions (a few MB) instead of the full multi-GB trajectory.

    Samples are taken at a fixed stride so they're spread evenly across the entire run.
    Returns {(kind, resnum): degrees_array}.
    """
    chunks_dir = md_output_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files found in {chunks_dir}")

    frames_per_chunk = np.load(chunk_files[0])["positions"].shape[0]
    last_chunk_frames = np.load(chunk_files[-1])["positions"].shape[0]
    total_frames = frames_per_chunk * (len(chunk_files) - 1) + last_chunk_frames
    stride = max(1, total_frames // n_samples)

    series = {(kind, resnum): [] for kind, resnum, _ in torsions}
    global_idx = 0
    for chunk_file in chunk_files:
        pos = np.load(chunk_file)["positions"]
        n = pos.shape[0]
        offset = (-global_idx) % stride
        local_indices = np.arange(offset, n, stride)
        if local_indices.size:
            sampled_pos = pos[local_indices]
            for kind, resnum, quad in torsions:
                angles = _dihedral_angle_batch(*(sampled_pos[:, i, :] for i in quad))
                series[(kind, resnum)].append(angles)
        global_idx += n
        del pos

    return {key: np.degrees(np.concatenate(vals)) for key, vals in series.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--md_output_dir",
        type=Path,
        required=True,
        help="Output dir from generate_md.py (i.e. the dir containing chunks/), e.g. "
        "{data_dir}/md/{seq_name}_{temperature}_{frame_interval}_{frames_per_chunk}",
    )
    parser.add_argument("--pdb_path", type=Path, required=True, help="Same PDB the generate_md.py run used")
    parser.add_argument("--temp_K", type=float, default=1200.0, help="For display only; not stored by generate_md.py")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1_000_000,
        help="Approx. number of frames to subsample, spread evenly across the full trajectory",
    )
    args = parser.parse_args()

    pdb = PDBFile(str(args.pdb_path))
    topology = pdb.getTopology()
    native_positions = np.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))

    torsions = build_torsion_list(topology)
    series = compute_torsion_series(args.md_output_dir, torsions, args.n_samples)
    n_frames = next(iter(series.values())).shape[0]
    print(
        f"Tracking {len(torsions)} restrained-candidate dihedrals in {args.pdb_path.stem}, "
        f"{n_frames} frames at {args.temp_K:.0f} K",
    )

    theta0s = {}
    print("\n=== Native (t=0) reference angles ===")
    for kind, resnum, quad in torsions:
        theta0 = np.degrees(_dihedral_angle(*(native_positions[i] for i in quad)))
        theta0s[(kind, resnum)] = theta0
        print(f"  {kind:8s} res {resnum:2d}: {theta0:7.2f} deg")

    # Deviation from each torsion's own native value, wrapped into (-180, 180] so omega's
    # theta0=180 (right on the wrap boundary) doesn't spuriously register as a flip -- this is
    # exactly the same wrapped-difference the flat-bottom restraint itself uses.
    print(f"\n=== At {args.temp_K:.0f} K, {n_frames} frames (deviation from native theta0) ===")
    print(
        f"{'residue':>7} {'kind':>8} {'theta0':>8} {'mean_d':>8} {'std_d':>7} "
        f"{'p95|d|':>7} {'min_d':>8} {'max_d':>8}  flip?",
    )
    for kind, resnum, quad in torsions:
        theta0 = theta0s[(kind, resnum)]
        vals = series[(kind, resnum)]
        delta = (vals - theta0 + 180) % 360 - 180
        mean_d, std_d = circular_mean_std(delta)
        p95_abs_d = np.percentile(np.abs(delta), 95)
        min_d, max_d = delta.min(), delta.max()
        # Direction that moves theta toward 0 (planar/transition-state): negative delta if
        # theta0 > 0, positive delta if theta0 < 0. A "flip" here means the excursion in that
        # direction actually reached (or passed) the planar point, not just a same-sign wobble.
        toward_planar_d = min_d if theta0 > 0 else max_d
        flip = "*** FLIP" if abs(toward_planar_d) >= abs(theta0) else ""
        print(
            f"{resnum:7d} {kind:>8} {theta0:8.2f} {mean_d:8.2f} {std_d:7.2f} "
            f"{p95_abs_d:7.2f} {min_d:8.2f} {max_d:8.2f}  {flip}",
        )

    # Pooled across residues within each restraint group (chirality_tol_deg/k and
    # omega_tol_deg/k in configs/generate_remd.yaml are each a single global value applied to
    # every residue of that group), treating residues as exchangeable draws from one
    # distribution and assuming the per-frame deviations are approximately Gaussian.
    print("\n=== Pooled across residues (Gaussian 3*std) ===")
    groups = {"chirality": ("CA", "HA", "THR_CB", "THR_HB", "ILE_CB", "ILE_HB"), "omega": ("omega",)}
    for group_name, kinds in groups.items():
        deltas = []
        for kind, resnum, quad in torsions:
            if kind not in kinds:
                continue
            theta0 = theta0s[(kind, resnum)]
            vals = series[(kind, resnum)]
            deltas.append((vals - theta0 + 180) % 360 - 180)
        if not deltas:
            continue
        pooled = np.concatenate(deltas)
        _, pooled_std = circular_mean_std(pooled)
        pooled_max = np.max(np.abs(pooled))
        print(
            f"  {group_name:10s} n={pooled.size:>9d}  std={pooled_std:6.2f} deg  "
            f"3*std={3 * pooled_std:6.2f} deg  max={pooled_max:6.2f} deg",
        )


if __name__ == "__main__":
    main()
