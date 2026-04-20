import argparse
import io
import os
import random
import tarfile
import time

import numpy as np
from tqdm import tqdm


def load_sequences(sequence_file: str) -> list[str]:
    with open(sequence_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def sanitize_key(seq_name: str) -> str:
    # Keep full seq_name so different runs of the same peptide don't collide.
    return seq_name


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build REMD webdataset from pre-shuffled flat .npy files.",
    )
    p.add_argument("--sequence-file", required=True)
    p.add_argument("--converted-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--samples-per-replica-per-tar", type=int, default=1)
    p.add_argument("--num-tarfiles", type=int, default=None)
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of tars buffered simultaneously. Memory ~ batch_size * num_seqs * T * K * (N*3+1)*4 bytes.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    K = args.samples_per_replica_per_tar
    os.makedirs(args.output_dir, exist_ok=True)

    seq_names = load_sequences(args.sequence_file)
    num_sequences = len(seq_names)
    if num_sequences == 0:
        raise ValueError("No sequences loaded.")
    print(f"Loaded {num_sequences} sequences")

    first_pos_path = os.path.join(args.converted_dir, f"{seq_names[0]}.pos.npy")
    pos0 = np.load(first_pos_path, mmap_mode="r")
    T0, F, N0, _ = pos0.shape
    del pos0
    print(f"Reference from seq[0]: T={T0}, F={F}, N={N0}")

    if args.num_tarfiles is not None:
        num_tarfiles = args.num_tarfiles
    else:
        num_tarfiles = F // K
        print(f"Auto-determined num_tarfiles={num_tarfiles}")
    if num_tarfiles * K > F:
        raise ValueError(f"Not enough frames per replica: need {num_tarfiles * K}, have {F}")

    width = len(str(num_tarfiles))

    tars_to_do = [
        tar_i
        for tar_i in range(num_tarfiles)
        if not os.path.exists(os.path.join(args.output_dir, f"{tar_i:0{width}d}.tar"))
    ]
    print(f"Need to build {len(tars_to_do)}/{num_tarfiles} tarfiles")
    if not tars_to_do:
        print("Nothing to do.")
        return

    seq_order = list(range(num_sequences))
    random.Random(args.seed).shuffle(seq_order)

    t_all = time.time()
    for g_start in range(0, len(tars_to_do), args.batch_size):
        group = tars_to_do[g_start : g_start + args.batch_size]
        t_group = time.time()
        print(f"Group {g_start // args.batch_size + 1}: {len(group)} tars (indices {group[0]}..{group[-1]})")

        buffers: dict[int, list[tuple[str, bytes]]] = {tar_i: [] for tar_i in group}

        for s_i in tqdm(seq_order, desc="seqs", unit="seq", dynamic_ncols=True):
            seq_name = seq_names[s_i]
            pos_path = os.path.join(args.converted_dir, f"{seq_name}.pos.npy")
            temps_path = os.path.join(args.converted_dir, f"{seq_name}.temps.npy")
            pos = np.load(pos_path, mmap_mode="r")
            temps = np.load(temps_path)
            T = pos.shape[0]
            if pos.shape[1] != F:
                raise ValueError(f"{seq_name}: F={pos.shape[1]} != expected {F}")
            key_stem = sanitize_key(seq_name)

            for tar_i in group:
                buf = buffers[tar_i]
                chunk = np.ascontiguousarray(pos[:, tar_i * K : (tar_i + 1) * K])
                for r in range(T):
                    t_bytes = np.float32(float(temps[r])).tobytes()
                    for k in range(K):
                        frame = chunk[r, k]  # (N, 3)
                        flat = r * F + tar_i * K + k
                        key = f"{key_stem}_{flat:08d}"
                        buf.append((key, t_bytes + frame.tobytes()))
            del pos, temps

        for tar_i in tqdm(group, desc="tars", unit="tar", dynamic_ncols=True):
            samples = buffers.pop(tar_i)
            random.Random(args.seed + tar_i).shuffle(samples)
            final_path = os.path.join(args.output_dir, f"{tar_i:0{width}d}.tar")
            tmp_path = final_path + ".tmp"
            try:
                with tarfile.open(tmp_path, "w") as tar:
                    for key, raw in samples:
                        info = tarfile.TarInfo(name=f"{key}.bin")
                        info.size = len(raw)
                        tar.addfile(info, io.BytesIO(raw))
                os.replace(tmp_path, final_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

        print(f"Group done in {time.time() - t_group:.2f}s")

    print(f"All done in {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
