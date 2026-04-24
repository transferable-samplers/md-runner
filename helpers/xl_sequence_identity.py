#!/usr/bin/env python3
"""Compute sequence identity between xl.txt target sequences and PDB filename sequences.

Each PDB file is named SEQ.pdb where SEQ is an amino-acid string. For every target
in sequences/xl.txt we run a Needleman-Wunsch global alignment against every PDB
sequence and report identity = matches / alignment_length.

Modes:
  default           top-N PDB hits per target
  --cutoff C        list/write PDBs with max identity to any target >= C
  --exclude-cutoff  remove PDBs with max identity >= cutoff, then show top hits
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

PDB_DIRS = [
    Path("/network/scratch/t/tanc/PDBS_OLIGO"),
    Path("/network/scratch/t/tanc/PDBS_ASSORTED"),
    Path("/network/scratch/t/tanc/ATONG01/transferable-samplers/many-peptides-md/pdbs/train"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = REPO_ROOT / "sequences" / "xl.txt"

_TARGETS: list[str] = []


def read_targets(path: Path) -> list[str]:
    seqs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        seq = parts[-1].upper()
        if seq.isalpha():
            seqs.append(seq)
    return seqs


def collect_pdb_sequences(dirs: list[Path]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for d in dirs:
        if not d.exists():
            print(f"[warn] missing directory: {d}", file=sys.stderr)
            continue
        for p in d.glob("*.pdb"):
            seq = p.stem.upper()
            if not seq.isalpha() or seq in seen:
                continue
            seen.add(seq)
            out.append((seq, p))
    return out


def nw_identity(a: str, b: str, match: int = 1, mismatch: int = -1, gap: int = -1) -> float:
    """Needleman-Wunsch global alignment; return identity = matches / alignment_length."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    score = [[0] * (m + 1) for _ in range(n + 1)]
    tb = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = i * gap
        tb[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = j * gap
        tb[0][j] = 2
    for i in range(1, n + 1):
        ai = a[i - 1]
        row = score[i]
        prev_row = score[i - 1]
        tb_row = tb[i]
        for j in range(1, m + 1):
            diag = prev_row[j - 1] + (match if ai == b[j - 1] else mismatch)
            up = prev_row[j] + gap
            left = row[j - 1] + gap
            if diag >= up and diag >= left:
                row[j] = diag
                tb_row[j] = 0
            elif up >= left:
                row[j] = up
                tb_row[j] = 1
            else:
                row[j] = left
                tb_row[j] = 2
    i, j = n, m
    matches = 0
    length = 0
    while i > 0 or j > 0:
        length += 1
        move = tb[i][j]
        if move == 0:
            if a[i - 1] == b[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return matches / length if length else 0.0


def _init_worker(targets: list[str]) -> None:
    global _TARGETS
    _TARGETS = targets


def _score(seq: str) -> list[float]:
    return [nw_identity(t, seq) for t in _TARGETS]


def score_all(pdb_seqs: list[tuple[str, Path]], targets: list[str], workers: int) -> list[list[float]]:
    """Return identities[i][j] for pdb_seqs[i] vs targets[j]."""
    seqs_only = [s for s, _ in pdb_seqs]
    if workers <= 1:
        _init_worker(targets)
        return [_score(s) for s in seqs_only]
    chunksize = max(1, len(seqs_only) // (workers * 8))
    with Pool(workers, initializer=_init_worker, initargs=(targets,)) as pool:
        return pool.map(_score, seqs_only, chunksize=chunksize)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    ap.add_argument("--top", type=int, default=1, help="top-N matches per target (default: 1)")
    ap.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="redundancy-filter mode: count/write PDBs with max identity >= cutoff; writes drop_sequences.txt",
    )
    ap.add_argument(
        "--exclude-cutoff",
        type=float,
        default=None,
        help="filter out PDBs with max identity >= cutoff, then show top hits",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="parallel workers (default: $SLURM_CPUS_PER_TASK or 1)",
    )
    ap.add_argument("--drop-file", type=Path, default=REPO_ROOT / "drop_sequences.txt", help="output for --cutoff mode")
    args = ap.parse_args()

    targets = read_targets(args.targets)
    pdb_seqs = collect_pdb_sequences(PDB_DIRS)
    print(
        f"loaded {len(targets)} targets and {len(pdb_seqs)} PDB sequences (workers={args.workers})\n",
        file=sys.stderr,
    )

    idents = score_all(pdb_seqs, targets, args.workers)

    if args.cutoff is not None:
        by_dir: dict[str, list[int]] = {}
        drop_seqs: list[tuple[str, Path, float]] = []
        for (seq, path), row in zip(pdb_seqs, idents):
            mx = max(row)
            bucket = by_dir.setdefault(path.parent.name, [0, 0])
            bucket[1] += 1
            if mx >= args.cutoff:
                bucket[0] += 1
                drop_seqs.append((seq, path, mx))
        print(f"cutoff: max identity to any xl target >= {args.cutoff}")
        for name, (drop, total) in sorted(by_dir.items()):
            print(f"  {name:20s} drop {drop:>6}/{total:<6} ({100 * drop / total:.2f}%)")
        print(
            f"  {'TOTAL':20s} drop {len(drop_seqs):>6}/{len(pdb_seqs):<6} ({100 * len(drop_seqs) / len(pdb_seqs):.2f}%)",
        )
        with args.drop_file.open("w") as f:
            for seq, path, ident in sorted(drop_seqs, key=lambda r: -r[2]):
                f.write(f"{seq}\t{ident:.4f}\t{path}\n")
        print(f"wrote {len(drop_seqs)} sequences to {args.drop_file}")
        return 0

    keep_mask = [True] * len(pdb_seqs)
    if args.exclude_cutoff is not None:
        keep_mask = [max(row) < args.exclude_cutoff for row in idents]
        kept = sum(keep_mask)
        print(
            f"excluding PDBs with max identity >= {args.exclude_cutoff}: "
            f"kept {kept}/{len(pdb_seqs)} "
            f"({100 * kept / len(pdb_seqs):.2f}%)\n",
        )

    for j, t in enumerate(targets):
        scored = [(idents[i][j], pdb_seqs[i][0], pdb_seqs[i][1]) for i in range(len(pdb_seqs)) if keep_mask[i]]
        scored.sort(key=lambda r: r[0], reverse=True)
        print(f"{t}")
        for ident, seq, path in scored[: args.top]:
            print(f"  {ident:6.3f}  {seq}  [{path.parent.name}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
