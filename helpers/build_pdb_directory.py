#!/usr/bin/env python3
"""Build a flat directory of `<sequence>.pdb` files for the sequences listed in
remd_sequence_locations.csv (see sequence_location_map.py).

Each row's `run_dir` points at .../data/remd/<run>, and the starting structure
for that run lives at the sibling .../data/pdbs/<sequence>.pdb. This mostly
holds, but not always: e.g. every "uniref" row's own data/pdbs dir is empty --
those pdbs actually live under the "oligo-data" location's data/pdbs dir
instead. So for each sequence we first check its own location's pdbs dir, then
fall back to the other locations (in LOCATION_PRIORITY order) if not found
there. This resolves all 36,000 rows in the current CSV.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

# tag -> data/pdbs directory. idrome and idrome-2 share the same data dir
# (see sequence_location_map.py's LOCATIONS), hence the same pdbs dir.
PDBS_DIRS = {
    "oligo-data": Path("/network/archive/t/tanc/oligo/data/pdbs"),
    "idrome": Path("/network/archive/t/tanc/oligo/md-runner-idrome/data/pdbs"),
    "uniref": Path("/network/archive/t/tanc/oligo/md-runner-remd-uniref/data/pdbs"),
    "idrome-2": Path("/network/archive/t/tanc/oligo/md-runner-idrome/data/pdbs"),
    "extra": Path("/home/mila/t/tanc/scratch/md-runner-oligopeptides-extra/data/pdbs"),
}

# same priority used in sequence_location_map.py, for reproducible fallback order.
LOCATION_PRIORITY = ["oligo-data", "idrome", "uniref", "idrome-2", "extra"]


def find_pdb(seq: str, own_location: str) -> Path | None:
    order = [own_location] + [t for t in LOCATION_PRIORITY if t != own_location]
    for tag in order:
        p = PDBS_DIRS[tag] / f"{seq}.pdb"
        if p.exists():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="remd_sequence_locations.csv")
    p.add_argument("--sequence-file", default=None, help="If provided, only build pdbs for listed sequences")
    p.add_argument("--dst-dir", required=True)
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copy (faster, but not self-contained)")
    args = p.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    if args.sequence_file:
        with open(args.sequence_file) as f:
            wanted = {line.strip() for line in f if line.strip() and not line.startswith("#")}
        rows = [r for r in rows if r["sequence"] in wanted]
        missing_from_csv = wanted - {r["sequence"] for r in rows}
        if missing_from_csv:
            print(f"WARNING: {len(missing_from_csv)} sequences from --sequence-file not found in {args.csv}")

    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_skip = 0
    unresolved: list[str] = []
    for row in rows:
        seq = row["sequence"]
        dst = dst_dir / f"{seq}.pdb"
        if dst.exists():
            n_skip += 1
            continue

        src = find_pdb(seq, row["location"])
        if src is None:
            unresolved.append(seq)
            continue

        if args.symlink:
            dst.symlink_to(src)
        else:
            shutil.copyfile(src, dst)
        n_ok += 1

    print(f"Wrote {n_ok} pdbs to {dst_dir} ({n_skip} already present, {len(unresolved)} unresolved)")
    if unresolved:
        unresolved_path = dst_dir.parent / "unresolved_pdbs.txt"
        unresolved_path.write_text("\n".join(unresolved) + "\n")
        print(f"Unresolved sequences written to {unresolved_path}")


if __name__ == "__main__":
    main()
