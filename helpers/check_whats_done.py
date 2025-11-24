import argparse
import sys
from pathlib import Path

"""
check_whats_done.py

Read sequence names from listed files, combine into a set,
and report which sequences do NOT have any matching
.../data/md/{sequence}_*/chunks/chunk_<N>.npz
"""


def read_sequences(path: Path) -> set[str]:
    if not path.exists():
        print(f"Warning: list file not found: {path}", file=sys.stderr)
        return set()
    lines = [line.strip() for line in path.read_text().splitlines()]
    return {line for line in lines if line and not line.startswith("#")}


def main() -> None:
    p = argparse.ArgumentParser(description="Check which sequences reached a specified chunk")
    p.add_argument(
        "--sequence_files",
        nargs="+",
        default=["sequences/example_sequences.txt"],
        help="sequence list files",
    )
    p.add_argument(
        "--data-root",
        default="/network/scratch/t/tanc/md-runner/data/md",
        help="root folder containing md/{sequence}_*/chunks/chunk_<N>.npz",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=49,
        help="which chunk number to check for (default: 49)",
    )
    args = p.parse_args()

    seqs = set()
    for lf in args.sequence_files:
        seqs.update(read_sequences(Path(lf)))

    chunk_filename = f"chunk_{args.chunk}.npz"
    missing = []

    for seq in sorted(seqs):
        pattern = f"{seq}_*/chunks/{chunk_filename}"
        matches = list(Path(args.data_root).glob(pattern))
        if not matches:
            missing.append(seq)

    print(f"Checking for chunk_{args.chunk}")
    print(f"Total sequences (unique): {len(seqs)}")
    print(f"Reached chunk_{args.chunk}: {len(seqs) - len(missing)}")
    print(f"Missing chunk_{args.chunk} ({len(missing)}):")
    for s in missing:
        print(s)


if __name__ == "__main__":
    main()
