import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def collate_chunks_for_sequence(chunks_dir: Path, output_path: Path) -> bool:
    """Collate chunks for a single sequence. Returns True if successful."""
    # Find all chunk files
    chunk_files = sorted(
        chunks_dir.glob("chunk_*.npz"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    if not chunk_files:
        print(f"Warning: No chunk_*.npz files found in {chunks_dir}", file=sys.stderr)
        return False

    # Load and collect all positions and velocities
    positions_list = []
    velocities_list = []

    for chunk_file in chunk_files:
        with np.load(chunk_file) as chunk_data:
            if "positions" not in chunk_data:
                print(f"Warning: {chunk_file} missing 'positions' key", file=sys.stderr)
                return False
            if "velocities" not in chunk_data:
                print(f"Warning: {chunk_file} missing 'velocities' key", file=sys.stderr)
                return False

            positions = chunk_data["positions"]
            velocities = chunk_data["velocities"]

        if positions.shape[0] != velocities.shape[0]:
            print(
                f"Warning: {chunk_file} positions/velocities length mismatch "
                f"({positions.shape[0]} vs {velocities.shape[0]})",
                file=sys.stderr,
            )
            return False

        positions_list.append(positions)
        velocities_list.append(velocities)

    if not positions_list:
        print(f"Warning: No valid chunk data found in {chunks_dir}", file=sys.stderr)
        return False

    # Concatenate into contiguous arrays
    combined_positions = np.concatenate(positions_list, axis=0)
    combined_velocities = np.concatenate(velocities_list, axis=0)

    # Save combined arrays
    np.savez_compressed(
        output_path,
        positions=combined_positions,
        velocities=combined_velocities,
    )

    print(
        f"Saved {len(chunk_files)} chunks -> {output_path} (shape: {combined_positions.shape})",
    )
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Combine chunk files into contiguous arrays for all sequences")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root input directory containing {seq}_info/chunks/ directories",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save combined arrays",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for parallel processing (default: 8)",
    )
    args = p.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all {seq}_info directories
    seq_info_dirs = sorted(input_dir.glob("*"))
    if not seq_info_dirs:
        print(f"Error: No *_info directories found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(seq_info_dirs)} sequence directories")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_sequence(seq_info_dir: Path) -> bool:
        """Process a single sequence. Returns True if successful."""
        chunks_dir = seq_info_dir / "chunks"
        if not chunks_dir.exists():
            print(f"Warning: {chunks_dir} does not exist, skipping", file=sys.stderr)
            return False

        seq_name = seq_info_dir.name.split("_")[0]
        output_path = output_dir / f"{seq_name}.npz"

        return collate_chunks_for_sequence(chunks_dir, output_path)

    # Process sequences in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_seq = {
            executor.submit(process_sequence, seq_info_dir): seq_info_dir for seq_info_dir in seq_info_dirs
        }

        # Process completed tasks
        for future in as_completed(future_to_seq):
            seq_info_dir = future_to_seq[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"Error processing {seq_info_dir}: {exc}", file=sys.stderr)
                failed += 1

    print(f"\nSummary: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
