#!/usr/bin/env python3
"""Compute sequence identity and similarity between xl.txt targets and PDB filename sequences.

Uses Biopython's PairwiseAligner (global) for alignment. For each aligned pair:

  identity   = fraction of aligned columns with identical residues
  similarity = fraction of aligned columns where BLOSUM62(x, y) > 0
               (identities + conservative substitutions)

Modes:
  default            top-N PDB hits per target (ranked by --rank-by)
  --cutoff C         list/write PDBs with max identity to any target >= C
  --exclude-cutoff C filter out PDBs with max identity >= C, then show top hits
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

from Bio.Align import PairwiseAligner, substitution_matrices

PDB_DIRS = [
    Path("/network/scratch/t/tanc/PDBS_OLIGO"),
    Path("/network/scratch/t/tanc/PDBS_ASSORTED"),
    Path("/network/scratch/t/tanc/ATONG01/transferable-samplers/many-peptides-md/pdbs/train"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = REPO_ROOT / "sequences" / "xl.txt"

_TARGETS: list[str] = []
_ALIGNER: PairwiseAligner | None = None
_BLOSUM = None


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


def make_aligner(scoring: str) -> PairwiseAligner:
    aln = PairwiseAligner()
    aln.mode = "global"
    if scoring == "simple":
        aln.match_score = 1
        aln.mismatch_score = -1
        aln.open_gap_score = -1
        aln.extend_gap_score = -1
    elif scoring == "blosum62":
        aln.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aln.open_gap_score = -11
        aln.extend_gap_score = -1
    else:
        raise ValueError(f"unknown scoring: {scoring}")
    return aln


def align_pair(a: str, b: str) -> tuple[float, float, float]:
    """Return (identity, similarity, coverage) from best global alignment.

    identity   = matches / alignment_length  (EMBOSS needle convention)
    similarity = positions with BLOSUM62(x,y) > 0 / alignment_length
    coverage   = non-gap aligned positions / len(shorter sequence)
                 (MMseqs2-style coverage of shorter sequence)
    """
    assert _ALIGNER is not None
    assert _BLOSUM is not None
    aln = _ALIGNER.align(a, b)[0]
    s0, s1 = str(aln[0]), str(aln[1])
    length = len(s0)
    matches = 0
    sim_count = 0
    aligned_cols = 0
    for x, y in zip(s0, s1):
        if x == "-" or y == "-":
            continue
        aligned_cols += 1
        if x == y:
            matches += 1
            sim_count += 1
        elif _BLOSUM[x, y] > 0:
            sim_count += 1
    shorter = min(len(a), len(b))
    identity = matches / length
    similarity = sim_count / length
    coverage = aligned_cols / shorter if shorter else 0.0
    return identity, similarity, coverage


def _init_worker(targets: list[str], scoring: str) -> None:
    global _TARGETS, _ALIGNER, _BLOSUM
    _TARGETS = targets
    _ALIGNER = make_aligner(scoring)
    _BLOSUM = substitution_matrices.load("BLOSUM62")


def _score(seq: str) -> list[tuple[float, float, float]]:
    return [align_pair(t, seq) for t in _TARGETS]


def score_all(
    pdb_seqs: list[tuple[str, Path]],
    targets: list[str],
    scoring: str,
    workers: int,
) -> list[list[tuple[float, float, float]]]:
    """Return scores[i][j] = (identity, similarity, coverage)."""
    seqs_only = [s for s, _ in pdb_seqs]
    if workers <= 1:
        _init_worker(targets, scoring)
        return [_score(s) for s in seqs_only]
    chunksize = max(1, len(seqs_only) // (workers * 8))
    with Pool(workers, initializer=_init_worker, initargs=(targets, scoring)) as pool:
        return pool.map(_score, seqs_only, chunksize=chunksize)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    ap.add_argument("--top", type=int, default=1, help="top-N matches per target (default: 1)")
    ap.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="count/write PDBs with max identity to any target >= cutoff",
    )
    ap.add_argument(
        "--sim-cutoff",
        type=float,
        default=None,
        help="additional similarity threshold (drop if max_id >= cutoff "
        "OR max_sim >= sim-cutoff); applies to both --cutoff and "
        "--exclude-cutoff modes",
    )
    ap.add_argument(
        "--exclude-cutoff",
        type=float,
        default=None,
        help="filter out PDBs with max identity >= cutoff, then show top hits",
    )
    ap.add_argument(
        "--scoring",
        choices=["simple", "blosum62"],
        default="blosum62",
        help="alignment scoring (default: blosum62)",
    )
    ap.add_argument(
        "--rank-by",
        choices=["identity", "similarity"],
        default="similarity",
        help="ranking metric for top hits (default: similarity)",
    )
    ap.add_argument(
        "--cov-cutoff",
        type=float,
        default=0.8,
        help="MMseqs2-style coverage threshold on shorter sequence (default: 0.8). "
        "A pair is redundant iff identity >= cutoff AND coverage >= cov-cutoff.",
    )
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--drop-file", type=Path, default=REPO_ROOT / "drop_sequences.txt")
    args = ap.parse_args()

    targets = read_targets(args.targets)
    pdb_seqs = collect_pdb_sequences(PDB_DIRS)
    print(
        f"loaded {len(targets)} targets and {len(pdb_seqs)} PDB sequences "
        f"(scoring={args.scoring}, workers={args.workers})\n",
        file=sys.stderr,
    )

    scores = score_all(pdb_seqs, targets, args.scoring, args.workers)

    def is_redundant(row: list[tuple[float, float, float]], id_thr: float) -> bool:
        """MMseqs2-style: any target with identity>=id_thr AND coverage>=cov_thr,
        OR (optional) any target with similarity>=sim_thr."""
        cov_thr = args.cov_cutoff
        id_cov_hit = any(r[0] >= id_thr and r[2] >= cov_thr for r in row)
        sim_hit = args.sim_cutoff is not None and any(r[1] >= args.sim_cutoff for r in row)
        return id_cov_hit or sim_hit

    def best_pair(row: list[tuple[float, float, float]]) -> tuple[float, float, float]:
        """Per-target pair with max identity; return (id, sim, cov) of that pair."""
        j = max(range(len(row)), key=lambda k: row[k][0])
        return row[j]

    if args.cutoff is not None:
        by_dir: dict[str, list[int]] = {}
        drop_seqs: list[tuple[str, Path, float, float, float]] = []
        for (seq, path), row in zip(pdb_seqs, scores):
            bucket = by_dir.setdefault(path.parent.name, [0, 0])
            bucket[1] += 1
            if is_redundant(row, args.cutoff):
                bucket[0] += 1
                ident, sim, cov = best_pair(row)
                drop_seqs.append((seq, path, ident, sim, cov))
        parts = [f"identity >= {args.cutoff} AND coverage >= {args.cov_cutoff}"]
        if args.sim_cutoff is not None:
            parts.append(f"similarity >= {args.sim_cutoff}")
        print(f"cutoff: ({' OR '.join(parts)}) on any xl target")
        for name, (drop, total) in sorted(by_dir.items()):
            print(f"  {name:20s} drop {drop:>6}/{total:<6} ({100 * drop / total:.2f}%)")
        print(
            f"  {'TOTAL':20s} drop {len(drop_seqs):>6}/{len(pdb_seqs):<6} "
            f"({100 * len(drop_seqs) / len(pdb_seqs):.2f}%)",
        )
        with args.drop_file.open("w") as f:
            f.write("#sequence\tidentity\tsimilarity\tcoverage\tpath\n")
            for seq, path, ident, sim, cov in sorted(drop_seqs, key=lambda r: -r[2]):
                f.write(f"{seq}\t{ident:.4f}\t{sim:.4f}\t{cov:.4f}\t{path}\n")
        print(f"wrote {len(drop_seqs)} sequences to {args.drop_file}")
        return 0

    keep_mask = [True] * len(pdb_seqs)
    if args.exclude_cutoff is not None:
        keep_mask = [not is_redundant(row, args.exclude_cutoff) for row in scores]
        kept = sum(keep_mask)
        parts = [f"identity >= {args.exclude_cutoff} AND coverage >= {args.cov_cutoff}"]
        if args.sim_cutoff is not None:
            parts.append(f"similarity >= {args.sim_cutoff}")
        print(
            f"excluding PDBs with ({' OR '.join(parts)}): "
            f"kept {kept}/{len(pdb_seqs)} ({100 * kept / len(pdb_seqs):.2f}%)\n",
        )

    rank_idx = 0 if args.rank_by == "identity" else 1
    for j, t in enumerate(targets):
        scored = [
            (scores[i][j][0], scores[i][j][1], scores[i][j][2], pdb_seqs[i][0], pdb_seqs[i][1])
            for i in range(len(pdb_seqs))
            if keep_mask[i]
        ]
        scored.sort(key=lambda r: r[rank_idx], reverse=True)
        print(f"{t}")
        for ident, sim, cov, seq, path in scored[: args.top]:
            print(f"  id={ident:6.3f}  sim={sim:6.3f}  cov={cov:6.3f}  {seq}  [{path.parent.name}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
