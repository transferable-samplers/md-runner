import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def downsample_file(input_path: Path, output_path: Path, factor: int) -> bool:
    """Downsample a single NPZ file. Returns True if successful."""
    try:
        with np.load(input_path) as data:
            # Check for required keys
            if "positions" not in data:
                print(f"Warning: {input_path} missing 'positions' key", file=sys.stderr)
                return False
            if "velocities" not in data:
                print(f"Warning: {input_path} missing 'velocities' key", file=sys.stderr)
                return False

            if data["positions"].shape[0] != data["velocities"].shape[0]:
                print(f"Warning: {input_path} positions/velocities length mismatch", file=sys.stderr)
                return False

            # Downsample along the first axis (time dimension)
            positions = data["positions"][::factor]
            velocities = data["velocities"][::factor]
            original_shape = data["positions"].shape
            new_shape = positions.shape

        # Save downsampled arrays
        np.savez_compressed(
            output_path,
            positions=positions,
            velocities=velocities,
        )

        print(f"Downsampled {input_path.name}: {original_shape} -> {new_shape} (factor: {factor})")
        return True

    except Exception as e:
        print(f"Error processing {input_path}: {e}", file=sys.stderr)
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Downsample NPZ arrays by a factor")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing .npz files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to save downsampled arrays",
    )
    p.add_argument(
        "--factor",
        type=int,
        required=True,
        help="Downsampling factor (e.g., 2 means keep every 2nd frame, 10 means keep every 10th frame)",
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
    factor = args.factor

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    if factor < 1:
        print(f"Error: Downsampling factor must be >= 1, got {factor}", file=sys.stderr)
        sys.exit(1)

    # Find all .npz files
    npz_files = sorted(input_dir.glob("*.npz"))

    if not npz_files:
        print(f"Error: No .npz files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(npz_files)} .npz files to downsample")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_file(npz_file: Path) -> bool:
        """Process a single file. Returns True if successful."""
        output_path = output_dir / npz_file.name
        return downsample_file(npz_file, output_path, factor)

    # Process files in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, npz_file): npz_file for npz_file in npz_files}

        # Process completed tasks
        for future in as_completed(future_to_file):
            npz_file = future_to_file[future]
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"Error processing {npz_file}: {exc}", file=sys.stderr)
                failed += 1

    print(f"\nSummary: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
