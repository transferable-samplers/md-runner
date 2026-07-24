"""Extract subsampled positions from OLIGO trajectory npz files into raw .npy files."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_SRC = Path("/network/archive/t/tanc/OLIGO")
DEFAULT_DST = Path("/network/scratch/t/tanc/OLIGO_SMALL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    parser.add_argument("--key", default="300.0_positions")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    files = sorted(args.src.glob("*.trajectories.npz"))
    files = files[args.shard_index::args.num_shards]
    for f in tqdm(files, desc=f"extracting (shard {args.shard_index}/{args.num_shards})"):
        out_path = args.dst / f"{f.stem.removesuffix('.trajectories')}.npy"
        if out_path.exists():
            continue
        with np.load(f) as data:
            positions = data[args.key][1::args.stride]
        np.save(out_path, positions)


if __name__ == "__main__":
    main()
