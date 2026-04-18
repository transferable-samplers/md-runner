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


def load_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an REMD trajectory npz. Returns (temperatures[T], positions[T, F, N, 3]).
    Drops frame 0 per replica (initial state, makes F an odd number like 1251 -> 1250).
    """
    with np.load(path) as z:
        temps = np.asarray(z["temperatures"], dtype=np.float32)
        key_by_temp: dict[float, str] = {}
        for k in z.files:
            if k.endswith("_positions"):
                key_by_temp[float(k[: -len("_positions")])] = k
        positions_list = []
        for t in temps:
            match_t = min(key_by_temp.keys(), key=lambda x: abs(x - float(t)))
            positions_list.append(z[key_by_temp[match_t]][1:])  # drop frame 0
        positions = np.stack(positions_list, axis=0)  # (T, F, N, 3)
    return temps, positions


def sanitize_key(seq_name: str) -> str:
    # The logical sequence is the peptide — letters before the first underscore.
    return seq_name.split("_", 1)[0]


def per_sequence_replica_indices(
    seq_index: int,
    T: int,
    F: int,
    num_tarfiles: int,
    samples_per_replica_per_tar: int,
    seed: int,
) -> np.ndarray:
    """Per-replica permutation of F frames; slice [:, tar_i, :] to get that tar's indices.
    Returns shape (T, num_tarfiles, samples_per_replica_per_tar). Deterministic from
    (seed, seq_index): rebuilding with the same seed gives the same assignment.
    """
    total_needed = num_tarfiles * samples_per_replica_per_tar
    if total_needed > F:
        raise ValueError(f"Not enough frames per replica: need {total_needed}, have {F}")
    rng = np.random.default_rng(np.random.SeedSequence([seed, seq_index]))
    out = np.empty((T, num_tarfiles, samples_per_replica_per_tar), dtype=np.int64)
    for r in range(T):
        out[r] = rng.permutation(F)[:total_needed].reshape(num_tarfiles, samples_per_replica_per_tar)
    return out


def process_sequence_for_batch(
    seq_index: int,
    batch_tars: list[int],
    seq_name: str,
    downsampled_dir: str,
    samples_per_replica_per_tar: int,
    num_tarfiles: int,
    F_expected: int,
    suffix: str,
    seed: int,
) -> tuple[int, dict[int, list[tuple[str, np.ndarray, float]]]]:
    """Load one sequence once, extract all samples needed for this batch of tarfiles."""
    path = os.path.join(downsampled_dir, f"{seq_name}{suffix}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trajectory: {path}")

    temps, positions = load_trajectory(path)  # (T,), (T, F, N, 3)
    T, F, _, _ = positions.shape
    if F != F_expected:
        raise ValueError(f"{seq_name}: F={F} != expected {F_expected}")

    indices = per_sequence_replica_indices(
        seq_index, T, F, num_tarfiles, samples_per_replica_per_tar, seed,
    )
    key_stem = sanitize_key(seq_name)

    out: dict[int, list[tuple[str, np.ndarray, float]]] = {}
    for tar_i in batch_tars:
        samples: list[tuple[str, np.ndarray, float]] = []
        for r in range(T):
            t_val = float(temps[r])
            for f_within in indices[r, tar_i]:
                fi = int(f_within)
                flat = r * F + fi
                pos = positions[r, fi]
                key = f"{key_stem}_{flat:08d}"
                samples.append((key, pos, t_val))
        out[tar_i] = samples

    return seq_index, out


def build_tar(tar_index: int, samples_for_tar: list[tuple[str, np.ndarray, float]], tar_filename: str, seed: int) -> int:
    # Deterministic shuffle per tar
    rng = random.Random(seed + tar_index)
    rng.shuffle(samples_for_tar)

    with tarfile.open(tar_filename, "w") as tar:
        for key, pos, temp in samples_for_tar:
            if pos.dtype != np.float32:
                raise ValueError(f"{key}: positions dtype {pos.dtype} != float32")
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError(f"{key}: positions shape {pos.shape} invalid")

            # Layout: [temp float32][positions float32 ...]. Decoder:
            #   data = np.frombuffer(b, dtype=np.float32)
            #   temp, pos = float(data[0]), data[1:].reshape(-1, 3)
            raw = np.float32(temp).tobytes() + pos.tobytes()
            info = tarfile.TarInfo(name=f"{key}.bin")
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))

    return tar_index


def process_batch(
    batch_tars: list[int],
    args,
    seq_names: list[str],
    num_sequences: int,
    F: int,
    num_integer_for_tarfile: int,
) -> None:
    batch_data: list[dict[int, list[tuple[str, np.ndarray, float]]]] = [None] * num_sequences  # type: ignore

    with ThreadPoolExecutor(max_workers=args.max_workers_load) as ex:
        futures = [
            ex.submit(
                process_sequence_for_batch,
                s_i,
                batch_tars,
                seq_names[s_i],
                args.downsampled_dir,
                args.samples_per_replica_per_tar,
                args.num_tarfiles,
                F,
                args.trajectory_file_suffix,
                args.seed,
            )
            for s_i in range(num_sequences)
        ]
        for fut in as_completed(futures):
            s_i, seq_out = fut.result()
            batch_data[s_i] = seq_out

    with ProcessPoolExecutor(max_workers=args.max_workers_tar) as ex:
        futures = []
        for tar_i in batch_tars:
            tar_code = str(tar_i).zfill(num_integer_for_tarfile)
            tar_path = os.path.join(args.output_dir, f"{tar_code}.tar")

            samples_for_tar: list[tuple[str, np.ndarray, float]] = []
            for s_i in range(num_sequences):
                samples_for_tar.extend(batch_data[s_i][tar_i])

            futures.append(ex.submit(build_tar, tar_i, samples_for_tar, tar_path, args.seed))

        for fut in as_completed(futures):
            tar_i = fut.result()
            print(f"Finished building tar {tar_i}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build REMD webdataset; single .bin with [temp][pos] per sample")
    p.add_argument("--sequence-file", required=True)
    p.add_argument("--downsampled-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-replica-per-tar", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-workers-load", type=int, default=8)
    p.add_argument("--max-workers-tar", type=int, default=3)
    p.add_argument("--num-tarfiles", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trajectory-file-suffix", default=".trajectories")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seq_names = load_sequences(args.sequence_file)
    num_sequences = len(seq_names)
    if num_sequences == 0:
        raise ValueError("No sequences loaded.")
    print(f"Loaded {num_sequences} sequences")

    # Probe F from first sequence. F (frames/replica) must be uniform across sequences;
    # T (replica count) is allowed to vary.
    path = os.path.join(args.downsampled_dir, f"{seq_names[0]}{args.trajectory_file_suffix}.npz")
    temps0, pos0 = load_trajectory(path)
    T0, F = pos0.shape[:2]
    print(f"Reference from seq[0]: T={T0}, F={F}")

    if args.num_tarfiles is not None:
        num_tarfiles = args.num_tarfiles
        if num_tarfiles * args.samples_per_replica_per_tar > F:
            raise ValueError(
                f"Not enough frames per replica: need "
                f"{num_tarfiles * args.samples_per_replica_per_tar}, have {F}"
            )
    else:
        num_tarfiles = F // args.samples_per_replica_per_tar
        print(f"Auto-determined num_tarfiles={num_tarfiles}")
    args.num_tarfiles = num_tarfiles

    num_integer_for_tarfile = len(str(num_tarfiles))

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
        process_batch(batch, args, seq_names, num_sequences, F, num_integer_for_tarfile)
        print(f"Batch done in {time.time() - t0:.2f}s")

    print("Done.")


if __name__ == "__main__":
    main()
