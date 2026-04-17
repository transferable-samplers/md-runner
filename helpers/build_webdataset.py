import argparse
import io
import os
import random
import tarfile
import time

import numpy as np


def load_sequences(sequence_file: str) -> list[str]:
    """Load sequence names from a text file (one per line)."""
    with open(sequence_file) as f:
        sequences = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return sequences


def get_trajectory_length(seq_name: str, downsampled_dir: str, trajectory_file_suffix: str) -> int:
    """Get the length of a trajectory from its npz file."""
    seq_path = os.path.join(downsampled_dir, f"{seq_name}{trajectory_file_suffix}.npz")
    with np.load(seq_path) as data:
        length = len(data["positions"])
        return length


def generate_tar_frame_index_files(
    num_tarfiles: int,
    seq_names: list[str],
    downsampled_dir: str,
    output_dir: str,
    tar_frame_index_prefix: str,
    samples_per_seq_per_tar: int,
    trajectory_file_suffix: str,
    seed: int = 42,
):
    """Generate tar frame index files with random frame indices for each sequence per tar.
    Ensures no frame index overlap across different tar files for the same sequence.
    """
    # Get trajectory length from first sequence (assuming all are the same)
    trajectory_length = get_trajectory_length(seq_names[0], downsampled_dir, trajectory_file_suffix)

    # Verify we have enough frames
    total_samples_needed = num_tarfiles * samples_per_seq_per_tar
    assert total_samples_needed <= trajectory_length, (
        f"Not enough frames! Need {total_samples_needed} but only have {trajectory_length}"
    )

    print(f"Generating {num_tarfiles} tar frame index files...")
    print(f"Trajectory length: {trajectory_length}, Total samples per seq: {total_samples_needed}")

    rng = np.random.default_rng(seed)

    # Pre-generate shuffled indices for each sequence (no overlap across tars)
    all_indices = {}
    for seq_name in seq_names:
        # Shuffle all available indices for this sequence
        shuffled = rng.permutation(trajectory_length)[:total_samples_needed]
        all_indices[seq_name] = shuffled

    # Now split into tar files
    for tar_index in range(num_tarfiles):
        tar_frame_index_path = os.path.join(output_dir, f"{tar_frame_index_prefix}_{tar_index}.npz")

        # Shape: (num_sequences, samples_per_seq_per_tar)
        indices = np.zeros((len(seq_names), samples_per_seq_per_tar), dtype=np.int64)

        start_idx = tar_index * samples_per_seq_per_tar
        end_idx = start_idx + samples_per_seq_per_tar

        for seq_idx, seq_name in enumerate(seq_names):
            # Take the next chunk of pre-shuffled indices for this sequence
            indices[seq_idx] = all_indices[seq_name][start_idx:end_idx]

        # Save tar frame index file
        np.savez_compressed(tar_frame_index_path, arrays=indices)

        if (tar_index + 1) % 100 == 0:
            print(f"  Generated {tar_index + 1}/{num_tarfiles} tar frame index files...")

    print(f"Finished generating {num_tarfiles} tar frame index files!")


def process_sequence(
    seq_index,
    batch_tars,
    tar_frame_indices,
    seq_name,
    downsampled_dir,
    samples_per_seq_per_tar,
    trajectory_file_suffix,
):
    """Process a single sequence to extract samples for the batch of tarfiles."""
    # Load the downsampled trajectory for this sequence
    downsampled_npz_path = os.path.join(downsampled_dir, f"{seq_name}{trajectory_file_suffix}.npz")

    if not os.path.exists(downsampled_npz_path):
        raise FileNotFoundError(f"Downsampled trajectory not found for {seq_name}: {downsampled_npz_path}")

    data = np.load(downsampled_npz_path)
    downsampled_array = data["positions"].copy()
    data.close()

    # Collect all frame indices and filenames needed for this sequence across all batch tars
    required_indexes = []
    filenames = []

    for tar_index in batch_tars:
        indices_for_tar = tar_frame_indices[tar_index][seq_index]  # Shape: (samples_per_seq_per_tar,)
        assert len(indices_for_tar) == samples_per_seq_per_tar

        for frame_index in indices_for_tar:
            assert frame_index < len(downsampled_array), (
                f"Index {frame_index} out of bounds for {seq_name} (length: {len(downsampled_array)})"
            )

            sample_filename = f"{seq_name}_{str(frame_index).zfill(8)}.bin"
            required_indexes.append(frame_index)
            filenames.append(sample_filename)

    # Extract all required samples at once
    samples = downsampled_array[required_indexes]

    # Organize samples by tar file
    sequence_batch_data = {}
    idx = 0
    for tar_index in batch_tars:
        sequence_batch_data[tar_index] = []
        for _ in range(samples_per_seq_per_tar):
            sequence_batch_data[tar_index].append((filenames[idx], samples[idx]))
            idx += 1

    return seq_index, sequence_batch_data


def build_tar(tar_index, samples_for_tar, tar_filename):
    """Build a single tar file with shuffled samples."""
    # Shuffle samples within this tar
    seed = hash((tar_index, time.time_ns()))
    rng = random.Random(seed)
    rng.shuffle(samples_for_tar)

    with tarfile.open(tar_filename, "w") as tar:
        for sample_filename, sample in samples_for_tar:
            assert sample.dtype == np.float32
            assert len(sample.shape) == 2 and sample.shape[1] == 3
            assert sample.shape[0] < 600

            raw_bytes = sample.astype(np.float32).tobytes()
            buffer = io.BytesIO(raw_bytes)

            tarinfo = tarfile.TarInfo(name=sample_filename)
            tarinfo.size = len(raw_bytes)
            tar.addfile(tarinfo, buffer)

    return tar_index


def process_batch(batch_tars, args, seq_names, num_sequences, num_integer_for_tarfile, num_samples_per_tar):
    """Process a batch of tar files."""
    # Load tar frame indices for this batch
    tar_frame_indices = {}
    for tar_index in batch_tars:
        tar_frame_index_path = os.path.join(args.output_dir, f"{args.tar_frame_index_prefix}_{tar_index}.npz")
        if not os.path.exists(tar_frame_index_path):
            raise FileNotFoundError(f"Missing tar frame index file: {tar_frame_index_path}")
        tar_frame_indices[tar_index] = np.load(tar_frame_index_path)["arrays"]

    # Load sequences sequentially
    batch_data = []
    for i in range(num_sequences):
        seq_index, sequence_batch_data = process_sequence(
            i,
            batch_tars,
            tar_frame_indices,
            seq_names[i],
            args.downsampled_dir,
            args.samples_per_seq_per_tar,
            args.trajectory_file_suffix,
        )
        batch_data.append(sequence_batch_data)

    # Build tar files sequentially
    for tar_index in batch_tars:
        tar_code = str(tar_index).zfill(num_integer_for_tarfile)
        tar_filename = os.path.join(args.output_dir, f"{tar_code}.tar")

        if os.path.exists(tar_filename):
            print(f"Skipping existing tar {tar_index}")
            continue

        # Collect all samples for this tar file
        samples_to_write = []
        for seq_index in range(num_sequences):
            samples_to_write.extend(batch_data[seq_index][tar_index])

        assert len(samples_to_write) == num_samples_per_tar, (
            f"Expected {num_samples_per_tar} samples but got {len(samples_to_write)}"
        )

        build_tar(tar_index, samples_to_write, tar_filename)
        print(f"Finished building tar {tar_index}")


def main():
    parser = argparse.ArgumentParser(description="Build webdataset from downsampled trajectories")
    parser.add_argument(
        "--sequence-file",
        type=str,
        required=True,
        help="Path to text file with sequence names (one per line)",
    )
    parser.add_argument(
        "--downsampled-dir",
        type=str,
        required=True,
        help="Directory containing downsampled trajectory files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for webdataset tar files",
    )
    parser.add_argument(
        "--tar-frame-index-prefix",
        type=str,
        default="tar_frame_index",
        help="Prefix for tar frame index files (default: 'tar_frame_index')",
    )
    parser.add_argument(
        "--samples-per-seq-per-tar",
        type=int,
        default=16,
        help="Number of samples per sequence per tar file (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of tar files to process per batch (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generating tar frame indices (default: 42)",
    )
    parser.add_argument(
        "--trajectory-file-suffix",
        type=str,
        default="_downsampled",
        help="Suffix for trajectory files (default: '_downsampled')",
    )

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    seq_names = load_sequences(args.sequence_file)
    num_sequences = len(seq_names)
    print(f"Loaded {num_sequences} sequences from {args.sequence_file}")

    # Check for existing tar frame index files
    tar_frame_index_files = sorted(
        [
            f
            for f in os.listdir(args.output_dir)
            if f.startswith(f"{args.tar_frame_index_prefix}_") and f.endswith(".npz")
        ],
    )
    existing_num_tarfiles = len(tar_frame_index_files)

    # Auto-determine from trajectory length
    print("No tar frame index files found. Determining number of tarfiles from trajectory length...")
    trajectory_length = get_trajectory_length(seq_names[0], args.downsampled_dir, args.trajectory_file_suffix)
    num_tarfiles = trajectory_length // args.samples_per_seq_per_tar
    print(f"Auto-determined {num_tarfiles} tarfiles based on trajectory length ({trajectory_length})")

    # Generate or validate tar frame index files
    if existing_num_tarfiles > 0:
        # Validate all expected files exist
        missing_files = []
        for tar_index in range(num_tarfiles):
            tar_frame_index_path = os.path.join(args.output_dir, f"{args.tar_frame_index_prefix}_{tar_index}.npz")
            if not os.path.exists(tar_frame_index_path):
                missing_files.append(tar_index)

        if missing_files:
            raise ValueError(
                f"Partially created tar frame index files detected! "
                f"Found {existing_num_tarfiles} files but {len(missing_files)} are missing. "
                f"Please delete all tar frame index files and regenerate.",
            )
        print(f"All {num_tarfiles} tar frame index files already exist")
    else:
        generate_tar_frame_index_files(
            num_tarfiles,
            seq_names,
            args.downsampled_dir,
            args.output_dir,
            args.tar_frame_index_prefix,
            args.samples_per_seq_per_tar,
            args.trajectory_file_suffix,
            args.seed,
        )

    num_integer_for_tarfile = len(str(num_tarfiles))
    num_samples_per_tar = args.samples_per_seq_per_tar * num_sequences

    # Find tar files that need to be created
    tars_to_do = []
    for tar_index in range(num_tarfiles):
        tar_code = str(tar_index).zfill(num_integer_for_tarfile)
        tar_filename = os.path.join(args.output_dir, f"{tar_code}.tar")
        if not os.path.exists(tar_filename):
            tars_to_do.append(tar_index)

    print(f"Found {len(tars_to_do)} tar files to process (out of {num_tarfiles} total)")

    # Process in batches
    for batch_index in range(0, len(tars_to_do), args.batch_size):
        start_time = time.time()
        batch_tars = tars_to_do[batch_index : batch_index + args.batch_size]

        print(f"Processing batch {batch_index // args.batch_size + 1} with {len(batch_tars)} tarfiles", flush=True)

        process_batch(batch_tars, args, seq_names, num_sequences, num_integer_for_tarfile, num_samples_per_tar)

        elapsed_time = time.time() - start_time
        print(f"Batch {batch_index // args.batch_size + 1} processed in {elapsed_time:.2f} seconds", flush=True)

    print(f"\nCompleted processing all {len(tars_to_do)} tar files!")


if __name__ == "__main__":
    main()
