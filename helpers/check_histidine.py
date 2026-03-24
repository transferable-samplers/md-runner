#!/usr/bin/env python3
"""
Check PDB files with 'H' in the name to verify:
1. Only HIE residues (no HID or HIS)
2. HE1 atom exists on every HIE residue
3. HD1 atom does not exist on any HIE residue
4. All residue names are in the known standard set
5. All atom names are in the known set for their residue
"""

from pathlib import Path
from collections import defaultdict


PDB_DIR = Path("../scratch/md-runner-remd-uniref/data/pdbs")

# Standard backbone atoms shared by all residues
# H1/H2/H3 are N-terminal hydrogens (naming varies by force field)
_BB = {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3", "HA"}

# Per-residue expected atom names (backbone + sidechain)
VALID_ATOMS: dict[str, set[str]] = {
    "ALA": _BB | {"CB", "HB1", "HB2", "HB3"},
    "ARG": _BB | {"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
                  "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "HE",
                  "HH11", "HH12", "HH21", "HH22"},
    "ASN": _BB | {"CB", "CG", "OD1", "ND2", "HB2", "HB3", "HD21", "HD22"},
    "ASP": _BB | {"CB", "CG", "OD1", "OD2", "HB2", "HB3"},
    "CYS": _BB | {"CB", "SG", "HB2", "HB3", "HG"},
    "GLN": _BB | {"CB", "CG", "CD", "OE1", "NE2",
                  "HB2", "HB3", "HG2", "HG3", "HE21", "HE22"},
    "GLU": _BB | {"CB", "CG", "CD", "OE1", "OE2", "HB2", "HB3", "HG2", "HG3"},
    "GLY": {"N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3", "HA2", "HA3"},
    "HIE": _BB | {"CB", "CG", "ND1", "CD2", "CE1", "NE2",
                  "HB2", "HB3", "HE1", "HD2", "HE2"},
    "ILE": _BB | {"CB", "CG1", "CG2", "CD1",
                  "HB", "HG12", "HG13", "HG21", "HG22", "HG23",
                  "HD11", "HD12", "HD13"},
    "LEU": _BB | {"CB", "CG", "CD1", "CD2",
                  "HB2", "HB3", "HG", "HD11", "HD12", "HD13",
                  "HD21", "HD22", "HD23"},
    "LYS": _BB | {"CB", "CG", "CD", "CE", "NZ",
                  "HB2", "HB3", "HG2", "HG3", "HD2", "HD3",
                  "HE2", "HE3", "HZ1", "HZ2", "HZ3"},
    "MET": _BB | {"CB", "CG", "SD", "CE", "HB2", "HB3", "HG2", "HG3",
                  "HE1", "HE2", "HE3"},
    "PHE": _BB | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
                  "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HZ"},
    "PRO": {"N", "CA", "C", "O", "OXT", "CD", "CG", "CB",
            "HA", "HB2", "HB3", "HG2", "HG3", "HD2", "HD3", "H2", "H3"},
    "SER": _BB | {"CB", "OG", "HB2", "HB3", "HG"},
    "THR": _BB | {"CB", "OG1", "CG2", "HB", "HG1", "HG21", "HG22", "HG23"},
    "TRP": _BB | {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3",
                  "CZ2", "CZ3", "CH2",
                  "HB2", "HB3", "HD1", "HE1", "HE3", "HZ2", "HZ3", "HH2"},
    "TYR": _BB | {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH",
                  "HB2", "HB3", "HD1", "HD2", "HE1", "HE2", "HH"},
    "VAL": _BB | {"CB", "CG1", "CG2", "HB",
                  "HG11", "HG12", "HG13", "HG21", "HG22", "HG23"},
}

VALID_RESNAMES = set(VALID_ATOMS.keys())


def check_pdb(pdb_path):
    issues = []
    # Map resname -> set of atom names seen (across all residues of that type)
    resname_atoms: dict[str, set[str]] = defaultdict(set)
    # Map (resname, resseq) -> set of atom names, for per-residue HIE checks
    hie_residues: dict[tuple, set[str]] = defaultdict(set)

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            resname = line[17:20].strip()
            atomname = line[12:16].strip()
            resseq = line[22:26].strip()
            chain = line[21]
            resname_atoms[resname].add(atomname)
            if resname == "HIE":
                hie_residues[(chain, resseq)].add(atomname)

    # 1. Check for unknown residue names
    for resname in resname_atoms:
        if resname not in VALID_RESNAMES:
            issues.append(f"unknown residue name: {resname!r}")

    # 2. Check for forbidden histidine variants
    for bad in ("HID", "HIS", "HIP"):
        if bad in resname_atoms:
            issues.append(f"contains forbidden histidine variant: {bad}")

    # 3. Check atom names against the per-residue whitelist
    for resname, atomnames in resname_atoms.items():
        if resname not in VALID_ATOMS:
            continue  # already flagged above
        unknown_atoms = atomnames - VALID_ATOMS[resname]
        if unknown_atoms:
            issues.append(
                f"{resname} has unexpected atom(s): {', '.join(sorted(unknown_atoms))}"
            )

    # 4. Per-HIE-residue checks: HE1 must exist, HD1 must not
    for (chain, resseq), atoms in hie_residues.items():
        tag = f"HIE {chain}:{resseq}"
        if "HE1" not in atoms:
            issues.append(f"{tag} missing HE1")
        if "HD1" in atoms:
            issues.append(f"{tag} has HD1 (should not exist)")

    return issues


def main():
    pdb_files = sorted(p for p in PDB_DIR.glob("*.pdb") if "H" in p.name)

    if not pdb_files:
        print(f"No PDB files with 'H' in name found in {PDB_DIR}")
        return

    print(f"Checking {len(pdb_files)} PDB files...\n")

    all_ok = True
    for pdb in pdb_files:
        issues = check_pdb(pdb)
        if issues:
            all_ok = False
            print(f"FAIL {pdb.name}")
            for issue in issues:
                print(f"     - {issue}")
        else:
            print(f"OK   {pdb.name}")

    print()
    if all_ok:
        print("All files passed.")
    else:
        print("Some files have issues (see above).")


if __name__ == "__main__":
    main()
