import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def load_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load an REMD trajectory npz. Returns (temperatures[T], positions[T, F, N, 3]).
    Drops frame 0 per replica (initial state)."""
    with np.load(path) as z:
        temps = np.asarray(z["temperatures"], dtype=np.float32)
        key_by_temp: dict[float, str] = {}
        for k in z.files:
            if k.endswith("_positions"):
                key_by_temp[float(k[: -len("_positions")])] = k
        positions_list = []
        for t in temps:
            match_t = min(key_by_temp.keys(), key=lambda x: abs(x - float(t)))
            positions_list.append(np.asarray(z[key_by_temp[match_t]][1:], dtype=np.float32))
        positions = np.stack(positions_list, axis=0)  # (T, F, N, 3)
    return temps, positions


def _seed_for(global_seed: int, seq_name: str) -> np.random.SeedSequence:
    # Stable, order-independent seed derived from (global_seed, seq_name).
    h = int.from_bytes(hashlib.sha256(seq_name.encode("utf-8")).digest()[:4], "big")
    return np.random.SeedSequence([int(global_seed), h])


def convert_one(
    seq_name: str,
    src_dir: str,
    dst_dir: str,
    suffix: str,
    global_seed: int,
) -> str:
    pos_out = os.path.join(dst_dir, f"{seq_name}.pos.npy")
    temps_out = os.path.join(dst_dir, f"{seq_name}.temps.npy")
    if os.path.exists(pos_out) and os.path.exists(temps_out):
        return f"skip {seq_name}"

    src = os.path.join(src_dir, f"{seq_name}{suffix}.npz")
    if not os.path.exists(src):
        raise FileNotFoundError(src)

    temps, positions = load_trajectory(src)  # (T,), (T, F, N, 3)
    T, F, N, _ = positions.shape

    ss = _seed_for(global_seed, seq_name)
    child_seeds = ss.spawn(T)
    for r in range(T):
        rng = np.random.default_rng(child_seeds[r])
        perm = rng.permutation(F)
        positions[r] = positions[r, perm]

    tmp_pos = pos_out + ".tmp"
    tmp_temps = temps_out + ".tmp"
    np.save(tmp_pos, positions)
    np.save(tmp_temps, temps)
    os.replace(tmp_pos, pos_out)
    os.replace(tmp_temps, temps_out)
    return f"done {seq_name} T={T} F={F} N={N}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert REMD .npz trajectories to pre-shuffled flat .npy pairs.",
    )
    p.add_argument("--src-dir", required=True)
    p.add_argument("--dst-dir", required=True)
    p.add_argument("--sequence-file", default=None, help="If provided, convert only listed sequences")
    p.add_argument("--trajectory-file-suffix", default=".trajectories")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    if args.sequence_file:
        with open(args.sequence_file) as f:
            seq_names = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        suffix_full = f"{args.trajectory_file_suffix}.npz"
        seq_names = sorted(f[: -len(suffix_full)] for f in os.listdir(args.src_dir) if f.endswith(suffix_full))
    if not seq_names:
        raise ValueError("No sequences to convert.")
    print(f"Converting {len(seq_names)} sequences from {args.src_dir} -> {args.dst_dir}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                convert_one,
                name,
                args.src_dir,
                args.dst_dir,
                args.trajectory_file_suffix,
                args.seed,
            )
            for name in seq_names
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="convert", unit="seq"):
            msg = fut.result()
            if msg.startswith("done"):
                tqdm.write(msg)
    print("Done.")


if __name__ == "__main__":
    main()
