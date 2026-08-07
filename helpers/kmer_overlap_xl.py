#!/usr/bin/env python3
"""Check for shared k-mer substrings between xl.txt targets and the training
pool, under the Murphy10 reduced amino-acid alphabet.

Variant of kmer_overlap.py (which compares many-peptides train/val/test
splits verbatim, over the full 20-letter alphabet). Here train/test is the
xl setup (PDB_DIRS vs sequences/xl.txt, same pools as xl_sequence_identity.py
and xl_leakage_scan.py) and sequences are reduced to 10 letters before
k-mers are extracted, so a shared k-mer also catches near-substring matches
that differ only by conservative substitutions (e.g. L/V/I/M become
indistinguishable once reduced) -- an exact-alphabet k-mer match can miss
these while they still dominate a local structural motif for a short,
disordered peptide.

Murphy10 groups (murphy_10_tab); unlisted residues (C, A, G, P, H) are
singleton groups that map to themselves:
  L: L V I M   aliphatic hydrophobic
  S: S T       small, hydroxyl
  F: F Y W     aromatic
  E: E D N Q   acidic / amide
  K: K R       basic (long)
  C: C
  A: A
  G: G
  P: P
  H: H

Note the reduced alphabet has only 10 symbols, so the k-mer space is
smaller than the full-alphabet case (10**4 = 10,000 possible 4-mers vs
20**4 = 160,000): expect a somewhat higher baseline overlap rate at small k
purely by chance. Treat k=4 as a sanity floor and weight k=6 more heavily.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xl_sequence_identity import collect_pdb_sequences, read_targets  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = REPO_ROOT / "sequences" / "xl.txt"

PDB_DIRS = [
    Path("/network/scratch/t/tanc/PDBS_OLIGO"),
    Path("/network/scratch/t/tanc/PDBS_ASSORTED"),
    Path("/network/scratch/t/tanc/ATONG01/transferable-samplers/many-peptides-md/pdbs/train"),
]
DIR_LABELS = {
    PDB_DIRS[0]: "PDBS_OLIGO",
    PDB_DIRS[1]: "PDBS_ASSORTED",
    PDB_DIRS[2]: "many-peptides/train",
}

KMER_SIZES = [4, 5, 6]

# Longest shared reduced-alphabet substring, reported as a fraction of the
# target's own length rather than a fixed k: a 5-mer hit is much more likely
# to be coincidental in a 20-residue peptide than in a 9-residue one, so
# "worst case" is judged by coverage of the target, not absolute length.
COVERAGE_CUTOFFS = [0.2, 0.3, 0.4, 0.5, 0.6]
COVERAGE_FLOOR_K = min(KMER_SIZES)

MURPHY_10_GROUPS = {
    "L": "LVIM",
    "S": "ST",
    "F": "FYW",
    "E": "EDNQ",
    "K": "KR",
    "C": "C",
    "A": "A",
    "G": "G",
    "P": "P",
    "H": "H",
}
murphy_10_tab = {aa: group for group, members in MURPHY_10_GROUPS.items() for aa in members}


def reduce_seq(seq: str) -> str:
    return "".join(murphy_10_tab[aa] for aa in seq)


def kmers(seq: str, k: int) -> set[str]:
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def train_kmer_index(train: list[str], k: int) -> dict[str, set[str]]:
    """reduced k-mer -> set of original train sequences containing it."""
    index: dict[str, set[str]] = {}
    for s in train:
        for km in kmers(reduce_seq(s), k):
            index.setdefault(km, set()).add(s)
    return index


def longest_shared_substring(
    target: str,
    get_index,
    floor: int,
) -> tuple[int, set[str]]:
    """Largest k <= len(target) for which target has a shared reduced k-mer
    with train (searched downward from len(target) since existence of a
    shared (k+1)-mer implies a shared k-mer); (0, set()) if none found down
    to floor."""
    rt = reduce_seq(target)
    for k in range(len(rt), floor - 1, -1):
        index = get_index(k)
        shared = kmers(rt, k) & index.keys()
        if shared:
            matched: set[str] = set()
            for km in shared:
                matched |= index[km]
            return k, matched
    return 0, set()


def print_dir_breakdown(dropped: set[str], seq_to_dir: dict[str, str], totals: dict[str, int]) -> None:
    counts = Counter(seq_to_dir[s] for s in dropped)
    for name in sorted(totals):
        n = counts.get(name, 0)
        total = totals[name]
        pct = 100 * n / total if total else 0.0
        print(f"    {name:45s} {n:>6}/{total:<6} ({pct:.1f}%)")


def dropped_at_cutoff(targets: list[str], get_index, cutoff: float, floor: int) -> set[str]:
    """All train seqs sharing a reduced substring of length >= cutoff * len(target)
    (and >= floor) with any target -- not just each target's single *longest*
    match, since a train seq can be redundant with a target at the cutoff
    length without being that target's best match."""
    dropped: set[str] = set()
    for t in targets:
        rt = reduce_seq(t)
        min_k = max(floor, math.ceil(cutoff * len(t)))
        for k in range(min_k, len(rt) + 1):
            index = get_index(k)
            shared = kmers(rt, k) & index.keys()
            for km in shared:
                dropped |= index[km]
    return dropped


def print_coverage_table(rows: list[tuple[str, int, int, float, set[str]]]) -> None:
    header = ["target", "len", "longest", "coverage"]
    header += [f">={int(c * 100)}%" for c in COVERAGE_CUTOFFS]
    widths = [max(len(header[0]), max((len(r[0]) for r in rows), default=0))]
    widths += [max(len(h), 8) for h in header[1:]]

    def row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n=== coverage summary ===")
    print(row(header))
    print(row(["-" * w for w in widths]))
    for t, length, k, coverage, _ in rows:
        cells = [t, str(length), str(k), f"{100 * coverage:.1f}%"]
        cells += ["yes" if coverage >= c else "" for c in COVERAGE_CUTOFFS]
        print(row(cells))


def print_coverage_section(
    targets: list[str],
    train: list[str],
    get_index,
    seq_to_dir: dict[str, str],
    totals_by_dir: dict[str, int],
) -> None:
    print("\n=== longest shared reduced substring, as coverage of target length ===")
    rows: list[tuple[str, int, int, float, set[str]]] = []
    for t in targets:
        k, matched = longest_shared_substring(t, get_index, COVERAGE_FLOOR_K)
        coverage = k / len(t) if t else 0.0
        rows.append((t, len(t), k, coverage, matched))
        flags = " ".join(f">={int(c * 100)}%" for c in COVERAGE_CUTOFFS if coverage >= c)
        print(
            f"  {t:24s} len={len(t):3d}  longest_shared={k:2d}  "
            f"coverage={100 * coverage:5.1f}%  {flags}",
        )
        if matched:
            print(f"    -> {len(matched)} train seq(s), e.g. {sorted(matched)[:5]}")

    for cutoff in COVERAGE_CUTOFFS:
        hits = [t for t, _, _, cov, _ in rows if cov >= cutoff]
        pct = 100 * len(hits) / len(rows) if rows else 0.0
        print(f"targets with coverage >= {int(cutoff * 100)}%: {len(hits)}/{len(rows)} ({pct:.1f}%)")
        dropped = dropped_at_cutoff(targets, get_index, cutoff, COVERAGE_FLOOR_K)
        drop_pct = 100 * len(dropped) / len(train) if train else 0.0
        print(
            f"total train seqs dropped at coverage >= {int(cutoff * 100)}%: "
            f"{len(dropped)}/{len(train)} ({drop_pct:.1f}%)",
        )
        print_dir_breakdown(dropped, seq_to_dir, totals_by_dir)

    print_coverage_table(rows)


def print_summary_table(targets: list[str], stats: dict[str, dict[int, tuple[int, int]]]) -> None:
    header = ["target"]
    for k in KMER_SIZES:
        header += [f"{k}mer hits", f"{k}mer seqs"]
    widths = [max(len(header[0]), max((len(t) for t in targets), default=0))]
    widths += [max(len(h), 9) for h in header[1:]]

    def row(cells: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n=== summary ===")
    print(row(header))
    print(row(["-" * w for w in widths]))
    for t in targets:
        cells = [t]
        for k in KMER_SIZES:
            n_shared, n_seqs = stats.get(t, {}).get(k, (0, 0))
            cells += [str(n_shared), str(n_seqs)]
        print(row(cells))


def main() -> int:
    targets = read_targets(DEFAULT_TARGETS)
    pdb_seqs = collect_pdb_sequences(PDB_DIRS)
    train = [s for s, _ in pdb_seqs]
    seq_to_dir = {s: DIR_LABELS.get(path.parent, path.parent.name) for s, path in pdb_seqs}
    totals_by_dir = Counter(seq_to_dir.values())
    print(f"targets(test)={len(targets)} train={len(train)}", file=sys.stderr)
    print(f"train pool by source dir: {dict(sorted(totals_by_dir.items()))}", file=sys.stderr)

    index_cache: dict[int, dict[str, set[str]]] = {}

    def get_index(k: int) -> dict[str, set[str]]:
        if k not in index_cache:
            index_cache[k] = train_kmer_index(train, k)
        return index_cache[k]

    stats: dict[str, dict[int, tuple[int, int]]] = {}

    for k in KMER_SIZES:
        index = get_index(k)
        print(f"\n=== {k}-mers (Murphy10) ===")
        print(f"unique reduced {k}-mers in train: {len(index)}")

        eligible = [t for t in targets if len(t) >= k]
        n_hit = 0
        dropped: set[str] = set()
        for t in eligible:
            shared_kmers: set[str] = set()
            hit_train_seqs: set[str] = set()
            for km in kmers(reduce_seq(t), k):
                if km in index:
                    shared_kmers.add(km)
                    hit_train_seqs |= index[km]
            stats.setdefault(t, {})[k] = (len(shared_kmers), len(hit_train_seqs))
            dropped |= hit_train_seqs
            if shared_kmers:
                n_hit += 1
                print(f"  {t:24s} shared {len(shared_kmers)} reduced {k}-mer(s): {sorted(shared_kmers)}")
                print(f"    -> {len(hit_train_seqs)} train seq(s), e.g. {sorted(hit_train_seqs)[:5]}")
        pct = 100 * n_hit / len(eligible) if eligible else 0.0
        print(
            f"xl targets (len>={k}) sharing a reduced {k}-mer with train: "
            f"{n_hit}/{len(eligible)} ({pct:.1f}%)",
        )
        drop_pct = 100 * len(dropped) / len(train) if train else 0.0
        print(
            f"total train seqs dropped at this cutoff (share a reduced {k}-mer "
            f"with >=1 xl target): {len(dropped)}/{len(train)} ({drop_pct:.1f}%)",
        )
        print_dir_breakdown(dropped, seq_to_dir, totals_by_dir)

    print_summary_table(targets, stats)
    print_coverage_section(targets, train, get_index, seq_to_dir, totals_by_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
