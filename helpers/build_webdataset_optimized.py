import argparse
import io
import os
import random
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np


def load_sequences(sequence_file: str) -> list[str]:
    with open(sequence_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def get_trajectory_length(seq_name: str, downsampled_dir: str, suffix: str) -> int:
    path = os.path.join(downsampled_dir, f"{seq_name}{suffix}.npz")
    with np.load(path) as z:
        return len(z["positions"])


def generate_tar_frame_index_files(
    num_tarfiles: int,
    seq_names: list[str],
    downsampled_dir: str,
    output_dir: str,
    tar_frame_index_prefix: str,
    samples_per_seq_per_tar: int,
    suffix: str,
    seed: int = 42,
) -> None:
    """Generate per-tar index files: arrays shape (num_sequences, samples_per_seq_per_tar)."""
    trajectory_length = get_trajectory_length(seq_names[0], downsampled_dir, suffix)

    total_needed = num_tarfiles * samples_per_seq_per_tar
    if total_needed > trajectory_length:
        raise ValueError(f"Not enough frames: need {total_needed}, have {trajectory_length}")

    rng = np.random.default_rng(seed)

    # For each sequence: a unique set of indices across all tarfiles (no overlap)
    all_indices: dict[str, np.ndarray] = {}
    for seq in seq_names:
        all_indices[seq] = rng.permutation(trajectory_length)[:total_needed]

    for tar_i in range(num_tarfiles):
        out_path = os.path.join(output_dir, f"{tar_frame_index_prefix}_{tar_i}.npz")
        start = tar_i * samples_per_seq_per_tar
        end = start + samples_per_seq_per_tar

        idx = np.zeros((len(seq_names), samples_per_seq_per_tar), dtype=np.int64)
        for s_i, seq in enumerate(seq_names):
            idx[s_i] = all_indices[seq][start:end]

        np.savez_compressed(out_path, arrays=idx)

        if (tar_i + 1) % 100 == 0:
            print(f"  Generated {tar_i + 1}/{num_tarfiles} index files...")

    print(f"Finished generating {num_tarfiles} tar frame index files.")


def process_sequence_for_batch(
    seq_index: int,
    batch_tars: list[int],
    tar_frame_indices: dict[int, np.ndarray],
    seq_name: str,
    downsampled_dir: str,
    samples_per_seq_per_tar: int,
    suffix: str,
) -> tuple[int, dict[int, list[tuple[str, np.ndarray]]]]:
    """Load one sequence once, extract all samples needed for this batch of tarfiles."""
    path = os.path.join(downsampled_dir, f"{seq_name}{suffix}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trajectory: {path}")

    with np.load(path) as z:
        positions = z["positions"].copy()  # copy so file can close safely

    required = []
    filenames = []

    for tar_i in batch_tars:
        idxs = tar_frame_indices[tar_i][seq_index]
        if len(idxs) != samples_per_seq_per_tar:
            raise ValueError(f"Bad index shape for tar {tar_i}, seq {seq_index}")

        for frame_idx in idxs:
            fi = int(frame_idx)
            if fi < 0 or fi >= len(positions):
                raise IndexError(f"{seq_name}: frame {fi} out of bounds (len={len(positions)})")

            required.append(fi)
            filenames.append(f"{seq_name}_{fi:08d}.bin")

    samples = positions[required]

    out: dict[int, list[tuple[str, np.ndarray]]] = {}
    k = 0
    for tar_i in batch_tars:
        out[tar_i] = []
        for _ in range(samples_per_seq_per_tar):
            out[tar_i].append((filenames[k], samples[k]))
            k += 1

    return seq_index, out


def build_tar(tar_index: int, samples_for_tar: list[tuple[str, np.ndarray]], tar_filename: str) -> int:
    # Deterministic-ish shuffle per tar (doesn't need to be cryptographic)
    rng = random.Random(hash((tar_index, os.getpid(), time.time_ns())))
    rng.shuffle(samples_for_tar)

    with tarfile.open(tar_filename, "w") as tar:
        for fname, sample in samples_for_tar:
            # Optional sanity checks (remove if you want max speed)
            assert sample.dtype == np.float32
            assert sample.ndim == 2 and sample.shape[1] == 3

            raw = sample.astype(np.float32, copy=False).tobytes()
            info = tarfile.TarInfo(name=fname)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

    return tar_index


def process_batch(
    batch_tars: list[int],
    args,
    seq_names: list[str],
    num_sequences: int,
    num_integer_for_tarfile: int,
    num_samples_per_tar: int,
) -> None:
    # Load tar frame indices (and close files immediately)
    tar_frame_indices: dict[int, np.ndarray] = {}
    for tar_i in batch_tars:
        idx_path = os.path.join(args.output_dir, f"{args.tar_frame_index_prefix}_{tar_i}.npz")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Missing tar index file: {idx_path}")
        with np.load(idx_path) as z:
            tar_frame_indices[tar_i] = z["arrays"]

    # For this batch: gather per-sequence extracted samples
    batch_data: list[dict[int, list[tuple[str, np.ndarray]]]] = [None] * num_sequences  # type: ignore

    with ThreadPoolExecutor(max_workers=args.max_workers_load) as ex:
        futures = [
            ex.submit(
                process_sequence_for_batch,
                s_i,
                batch_tars,
                tar_frame_indices,
                seq_names[s_i],
                args.downsampled_dir,
                args.samples_per_seq_per_tar,
                args.trajectory_file_suffix,
            )
            for s_i in range(num_sequences)
        ]
        for fut in as_completed(futures):
            s_i, seq_out = fut.result()
            batch_data[s_i] = seq_out

    # Build tar files in parallel (processes), because tar writing can benefit from multiple cores
    with ProcessPoolExecutor(max_workers=args.max_workers_tar) as ex:
        futures = []
        for tar_i in batch_tars:
            tar_code = str(tar_i).zfill(num_integer_for_tarfile)
            tar_path = os.path.join(args.output_dir, f"{tar_code}.tar")

            if os.path.exists(tar_path):
                print(f"Skipping existing tar {tar_i}")
                continue

            samples_for_tar: list[tuple[str, np.ndarray]] = []
            for s_i in range(num_sequences):
                samples_for_tar.extend(batch_data[s_i][tar_i])

            if len(samples_for_tar) != num_samples_per_tar:
                raise ValueError(f"Tar {tar_i}: expected {num_samples_per_tar}, got {len(samples_for_tar)}")

            futures.append(ex.submit(build_tar, tar_i, samples_for_tar, tar_path))

        for fut in as_completed(futures):
            tar_i = fut.result()
            print(f"Finished building tar {tar_i}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build webdataset from downsampled trajectories")
    p.add_argument("--sequence-file", required=True)
    p.add_argument("--downsampled-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tar-frame-index-prefix", default="tar_frame_index")
    p.add_argument("--samples-per-seq-per-tar", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-workers-load", type=int, default=32)
    p.add_argument("--max-workers-tar", type=int, default=8)
    p.add_argument("--num-tarfiles", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory-file-suffix", default="_downsampled")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seq_names = load_sequences(args.sequence_file)
    num_sequences = len(seq_names)
    if num_sequences == 0:
        raise ValueError("No sequences loaded.")
    print(f"Loaded {num_sequences} sequences")

    # Determine trajectory length and validate equal lengths
    lengths = [get_trajectory_length(s, args.downsampled_dir, args.trajectory_file_suffix) for s in seq_names]
    min_len, max_len = min(lengths), max(lengths)
    if min_len != max_len:
        raise ValueError(f"Sequence length mismatch: min={min_len}, max={max_len}")
    trajectory_length = min_len

    # Existing index files?
    existing = sorted(
        f for f in os.listdir(args.output_dir) if f.startswith(f"{args.tar_frame_index_prefix}_") and f.endswith(".npz")
    )
    existing_num = len(existing)

    # Choose num_tarfiles
    if args.num_tarfiles is not None:
        num_tarfiles = args.num_tarfiles
    elif existing_num > 0:
        num_tarfiles = existing_num
    else:
        num_tarfiles = trajectory_length // args.samples_per_seq_per_tar
        print(f"Auto-determined num_tarfiles={num_tarfiles}")

    total_needed = num_tarfiles * args.samples_per_seq_per_tar
    if total_needed > trajectory_length:
        raise ValueError(f"Not enough frames: need {total_needed}, have {trajectory_length}")

    # Ensure index files exist (generate if missing)
    if existing_num > 0:
        # Validate completeness
        missing = []
        for tar_i in range(num_tarfiles):
            path = os.path.join(args.output_dir, f"{args.tar_frame_index_prefix}_{tar_i}.npz")
            if not os.path.exists(path):
                missing.append(tar_i)
        if missing:
            raise ValueError(f"Partial tar index set detected; missing {len(missing)} files (e.g. {missing[:5]})")
        print(f"Using existing {num_tarfiles} tar frame index files.")
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

    # Determine tars to do
    tars_to_do = []
    for tar_i in range(num_tarfiles):
        tar_code = str(tar_i).zfill(num_integer_for_tarfile)
        tar_path = os.path.join(args.output_dir, f"{tar_code}.tar")
        if not os.path.exists(tar_path):
            tars_to_do.append(tar_i)

    print(f"Need to build {len(tars_to_do)}/{num_tarfiles} tarfiles")

    for start in range(0, len(tars_to_do), args.batch_size):
        batch = tars_to_do[start : start + args.batch_size]
        print(f"Batch {start // args.batch_size + 1}: {len(batch)} tarfiles")
        t0 = time.time()
        process_batch(batch, args, seq_names, num_sequences, num_integer_for_tarfile, num_samples_per_tar)
        print(f"Batch done in {time.time() - t0:.2f}s")

    print("Done.")


if __name__ == "__main__":
    main()
