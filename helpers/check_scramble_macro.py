"""Compare post-scramble PDBs to their original input structures via ABEGO.

For each *_scramble{N} REMD output, load `scramble_snapshot.pdb` and the original
input PDB at `data/pdbs/<seq>.pdb`, classify each residue into an ABEGO region
(B1/B2 -> B and E1/E2 -> E), then report the per-residue ABEGO string and the
fraction of residues that changed bin between pre- and post-scramble.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import mdtraj as md
import numpy as np

SCRATCH = Path("/network/scratch/t/tanc/md-runner-remd-reference-xl/data")
REMD_DIR = SCRATCH / "remd"
PDB_DIR = SCRATCH / "pdbs"
OUT_CSV = Path("/home/mila/t/tanc/md-runner/scramble_abego.csv")

DIR_RE = re.compile(r"^(?P<seq>[A-Z]+)_\d+K-\d+K_\d+_[\d.]+_\d+_scramble(?P<seed>\d+)$")

RegionDict = dict[str, tuple[tuple[int, int], tuple[int, int]]]

ABEGO_REGIONS: RegionDict = {
    "A": ((-180, 0), (-75, 50)),
    "B1": ((-180, 0), (50, 180)),
    "B2": ((-180, 0), (-180, -75)),
    "G": ((0, 180), (-100, 100)),
    "E1": ((0, 180), (100, 180)),
    "E2": ((0, 180), (-180, -100)),
}

LABEL_MERGES: dict[str, str] = {"B1": "B", "B2": "B", "E1": "E", "E2": "E"}


def _wrap(a: float) -> float:
    """Wrap angle to half-open [-180, 180) so 180 maps to -180."""
    return ((a + 180.0) % 360.0) - 180.0


def _classify_residue(phi_deg: float, psi_deg: float) -> str:
    phi_deg = _wrap(phi_deg)
    psi_deg = _wrap(psi_deg)
    for name, ((phi_lo, phi_hi), (psi_lo, psi_hi)) in ABEGO_REGIONS.items():
        if phi_lo <= phi_deg < phi_hi and psi_lo <= psi_deg < psi_hi:
            return LABEL_MERGES.get(name, name)
    return "?"


def _phi_psi_deg(traj: md.Trajectory) -> tuple[np.ndarray, np.ndarray]:
    phi_atoms, phi = md.compute_phi(traj)
    psi_atoms, psi = md.compute_psi(traj)
    # phi has no value for residue 0; psi has no value for last residue.
    # Pad both arrays so column j corresponds to residue j (indexed 0..n_res-1),
    # with NaN where the dihedral is undefined.
    n_res = traj.topology.n_residues
    n_frames = traj.n_frames
    phi_full = np.full((n_frames, n_res), np.nan)
    psi_full = np.full((n_frames, n_res), np.nan)
    for k, atoms in enumerate(phi_atoms):
        res_idx = traj.topology.atom(atoms[2]).residue.index
        phi_full[:, res_idx] = phi[:, k]
    for k, atoms in enumerate(psi_atoms):
        res_idx = traj.topology.atom(atoms[1]).residue.index
        psi_full[:, res_idx] = psi[:, k]
    return np.degrees(phi_full), np.degrees(psi_full)


def abego_string(traj: md.Trajectory) -> list[str]:
    phi_deg, psi_deg = _phi_psi_deg(traj)
    n_res = phi_deg.shape[1]
    out = []
    for j in range(n_res):
        if np.isnan(phi_deg[0, j]) or np.isnan(psi_deg[0, j]):
            out.append("-")  # terminal residue: phi/psi undefined
        else:
            out.append(_classify_residue(float(phi_deg[0, j]), float(psi_deg[0, j])))
    return out


def main() -> None:
    rows: list[dict] = []
    runs = sorted(d for d in REMD_DIR.iterdir() if d.is_dir() and "scramble" in d.name)
    print(f"{'sequence':25s} {'seed':>4s}  ref_ABEGO -> post_ABEGO   (changed/total)")
    print("-" * 100)
    for run in runs:
        m = DIR_RE.match(run.name)
        if not m:
            continue
        seq = m.group("seq")
        seed = int(m.group("seed"))
        snap = run / "scramble_snapshot.pdb"
        ref = PDB_DIR / f"{seq}.pdb"
        if not snap.exists() or not ref.exists():
            continue
        ref_traj = md.load(str(ref))
        post_traj = md.load(str(snap))
        ref_ab = abego_string(ref_traj)
        post_ab = abego_string(post_traj)
        if len(ref_ab) != len(post_ab):
            print(f"skip {run.name}: residue mismatch {len(ref_ab)} vs {len(post_ab)}")
            continue
        # Count differences only over interior residues with defined dihedrals
        compare = [(r, p) for r, p in zip(ref_ab, post_ab) if r not in ("-", "?") and p not in ("-", "?")]
        n_compare = len(compare)
        n_diff = sum(1 for r, p in compare if r != p)
        frac_changed = n_diff / n_compare if n_compare else 0.0
        ref_str = "".join(ref_ab)
        post_str = "".join(post_ab)
        print(f"{seq:25s} {seed:>4d}  {ref_str} -> {post_str}   ({n_diff}/{n_compare}, {frac_changed:.0%})")
        rows.append(
            {
                "seq": seq,
                "seed": seed,
                "n_res": len(ref_ab),
                "n_compared": n_compare,
                "n_changed": n_diff,
                "frac_changed": frac_changed,
                "ref_abego": ref_str,
                "post_abego": post_str,
            },
        )

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seq", "seed", "n_res", "n_compared", "n_changed", "frac_changed", "ref_abego", "post_abego"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # Per-sequence summary
    by_seq: dict[str, list[dict]] = {}
    for r in rows:
        by_seq.setdefault(r["seq"], []).append(r)
    print(f"\n{'sequence':25s} {'n_res':>5s}  ref_ABEGO              mean_frac_changed   seeds_identical_to_ref")
    print("-" * 110)
    for seq in sorted(by_seq):
        rs = by_seq[seq]
        mean_chg = float(np.mean([r["frac_changed"] for r in rs]))
        identical = sum(1 for r in rs if r["n_changed"] == 0)
        ref_str = rs[0]["ref_abego"]
        print(f"{seq:25s} {rs[0]['n_res']:>5d}  {ref_str:22s}  {mean_chg:>5.0%}              {identical}/{len(rs)}")


if __name__ == "__main__":
    main()
