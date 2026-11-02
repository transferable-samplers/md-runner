"""Rewrite the many-peptides-md HF dataset: drop velocities, subsample train.

For every .npz under trajectories/ and trajectories_subsampled/:
  * the `velocities` array is removed (all other keys kept unchanged);
  * files under trajectories/train/ additionally keep every `--stride`-th frame
    (default 10: 1 ps -> 10 ps spacing, 200k -> 20k frames).
Each output keeps the compression of its source file (train files are
savez_compressed, val/test/subsampled are plain savez).

Integrity chain (recorded per file in <work-dir>/manifest.jsonl):
  1. sha256 of the downloaded source == LFS sha256 of the pinned source revision
     (proves we read the genuine original);
  2. content digest (dtype, shape, raw bytes) of every kept array, computed from
     the decoded original *before* writing the new file;
  3. sha256 of the new file as written. After each commit the remote LFS sha256
     is checked against it. helpers/verify_many_peptides_hf.py re-downloads the
     new files and re-checks (2) and (3) independently.

Streams: workers download -> process -> write to a staging dir -> delete the
source; the main process uploads the staging dir every `--shard-size` files as
one commit, and only then appends those files to the manifest (resume-safe).

    python helpers/slim_many_peptides_hf.py --work-dir /network/scratch/.../hf_slim --dry-run --limit 4
"""

import argparse
import hashlib
import json
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "transferable-samplers/many-peptides-md"
PREFIXES = ("trajectories/", "trajectories_subsampled/")

# Expected source layout per subset: (exact key set, number of frames).
# Anything else aborts the run rather than silently writing an odd file.
TRAJ_KEYS = {"sequence", "positions", "velocities"}
SUBSAMPLED_KEYS = TRAJ_KEYS | {"tica_projection", "tica_mean", "tica_dim"}


def expected_source(path_in_repo: str) -> tuple[set[str], int]:
    if path_in_repo.startswith("trajectories/train/"):
        return TRAJ_KEYS, 200_000  # 200 ns @ 1 ps
    if path_in_repo.startswith(("trajectories/val/", "trajectories/test/")):
        return TRAJ_KEYS, 500_000  # 5 us @ 10 ps
    if path_in_repo.startswith("trajectories_subsampled/"):
        return SUBSAMPLED_KEYS, 10_000  # 5 us @ 500 ps
    raise ValueError(f"unexpected path {path_in_repo}")


def check_source(path_in_repo: str, d) -> None:
    keys, n_frames = expected_source(path_in_repo)
    if set(d.files) != keys:
        raise RuntimeError(f"{path_in_repo}: source keys {sorted(d.files)} != expected {sorted(keys)}")
    pos, vel = d["positions"], d["velocities"]
    if pos.ndim != 3 or pos.shape[2] != 3 or pos.dtype != np.float32:
        raise RuntimeError(f"{path_in_repo}: positions {pos.shape} {pos.dtype}, expected (T, N, 3) float32")
    if pos.shape[0] != n_frames:
        raise RuntimeError(f"{path_in_repo}: {pos.shape[0]} frames != expected {n_frames}")
    if vel.shape != pos.shape or vel.dtype != pos.dtype:
        raise RuntimeError(f"{path_in_repo}: velocities {vel.shape} {vel.dtype} != positions {pos.shape} {pos.dtype}")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(a: np.ndarray) -> str:
    """Content hash independent of file container / compression."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def process_one(path_in_repo: str, expected_sha: str, work_dir: str, stride: int, revision: str) -> dict:
    work = Path(work_dir)
    dl_dir = work / "download" / hashlib.md5(path_in_repo.encode()).hexdigest()[:2]
    src = Path(
        hf_hub_download(REPO_ID, path_in_repo, repo_type="dataset", revision=revision, local_dir=dl_dir)
    )
    src_sha = file_sha256(src)
    if src_sha != expected_sha:
        raise RuntimeError(f"{path_in_repo}: source sha256 {src_sha} != LFS {expected_sha}")
    src_size = src.stat().st_size
    with zipfile.ZipFile(src) as z:
        compressed = any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist())

    is_train = path_in_repo.startswith("trajectories/train/")
    out_arrays = {}
    with np.load(src, allow_pickle=False) as d:
        check_source(path_in_repo, d)
        orig_keys = list(d.files)
        orig_shapes = {}
        for k in orig_keys:
            if k == "velocities":
                orig_shapes[k] = list(d[k].shape)  # decodes velocities once; also proves it was readable
                continue
            a = d[k]
            orig_shapes[k] = list(a.shape)
            if k == "positions" and is_train:
                a = np.ascontiguousarray(a[::stride])
            out_arrays[k] = a
    n_out = expected_source(path_in_repo)[1] // (stride if is_train else 1)
    if out_arrays["positions"].shape[0] != n_out:
        raise RuntimeError(f"{path_in_repo}: kept {out_arrays['positions'].shape[0]} frames != expected {n_out}")
    digests = {k: array_digest(v) for k, v in out_arrays.items()}

    dst = work / "staging" / path_in_repo
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **out_arrays)

    # round-trip check of the written file against digests from the original
    with np.load(tmp, allow_pickle=False) as d:
        if set(d.files) != set(out_arrays):
            raise RuntimeError(f"{path_in_repo}: written keys {d.files}")
        for k in out_arrays:
            if array_digest(d[k]) != digests[k]:
                raise RuntimeError(f"{path_in_repo}: digest mismatch for {k}")
    tmp.rename(dst)
    src.unlink()

    return {
        "path": path_in_repo,
        "source_revision": revision,
        "source_sha256": src_sha,
        "source_size": src_size,
        "source_keys": orig_keys,
        "source_shapes": orig_shapes,
        "stride": stride if is_train else 1,
        "compressed": compressed,
        "kept_digests": digests,
        "kept_shapes": {k: list(v.shape) for k, v in out_arrays.items()},
        "new_sha256": file_sha256(dst),
        "new_size": dst.stat().st_size,
    }


def lfs_sha_map(api: HfApi, revision: str) -> dict[str, str]:
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    return {s.rfilename: s.lfs.sha256 for s in info.siblings if s.lfs is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=500, help="files per upload commit")
    p.add_argument("--limit", type=int, default=None, help="only process the first N pending files")
    p.add_argument("--target-revision", default="main", help="branch to commit the new files to")
    p.add_argument("--dry-run", action="store_true", help="process but do not upload")
    args = p.parse_args()

    api = HfApi()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Pin the source revision so all downloads come from the original data,
    # never from files we have already overwritten.
    rev_file = args.work_dir / "source_revision.txt"
    if not rev_file.exists():
        rev_file.write_text(api.dataset_info(REPO_ID).sha)
    revision = rev_file.read_text().strip()
    print(f"source revision {revision}", flush=True)

    src_sha = lfs_sha_map(api, revision)
    files = sorted(f for f in src_sha if f.endswith(".npz") and f.startswith(PREFIXES))

    manifest = args.work_dir / ("manifest.dryrun.jsonl" if args.dry_run else "manifest.jsonl")
    done = set()
    if manifest.exists():
        done = {json.loads(line)["path"] for line in manifest.read_text().splitlines() if line}
    pending = [f for f in files if f not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(files)} npz files, {len(done)} done, {len(pending)} to process", flush=True)

    staging = args.work_dir / "staging"
    shutil.rmtree(staging, ignore_errors=True)  # partial outputs from a killed run

    def commit(records):
        if not args.dry_run:
            api.upload_folder(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=args.target_revision,
                folder_path=staging,
                commit_message=f"Drop velocities, subsample train x{args.stride} ({len(records)} files)",
            )
            # confirm the remote holds exactly the bytes we wrote
            remote = lfs_sha_map(api, args.target_revision)
            bad = [r["path"] for r in records if remote.get(r["path"]) != r["new_sha256"]]
            if bad:
                raise RuntimeError(f"remote sha256 mismatch after commit: {bad[:5]} ...")
        with manifest.open("a") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        shutil.rmtree(staging, ignore_errors=True)
        print(f"{'(dry-run) ' if args.dry_run else ''}committed {len(records)} files", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        # chunked submission bounds staging/download disk use to ~shard_size files
        for start in range(0, len(pending), args.shard_size):
            chunk = pending[start : start + args.shard_size]
            futs = [ex.submit(process_one, f, src_sha[f], str(args.work_dir), args.stride, revision) for f in chunk]
            records = []
            for fut in as_completed(futs):
                r = fut.result()
                records.append(r)
                print(f"  {r['path']}: {r['source_shapes']['positions'][0]}->{r['kept_shapes']['positions'][0]} "
                      f"frames, {r['source_size'] / 1e6:.0f}->{r['new_size'] / 1e6:.1f} MB", flush=True)
            commit(records)


if __name__ == "__main__":
    main()
