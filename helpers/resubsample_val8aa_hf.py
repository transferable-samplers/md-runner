"""Fix trajectories_subsampled/val/8AA/*_subsampled.npz on the no-velocities branch.

The 2025-11-17 "fixed tica data for 8AA" upload replaced these 10 files with 50k-frame
(100 ps) positions instead of the documented 10k frames (500 ps). Rebuild them:
  positions       = trajectories/val/8AA/<SEQ>.npz positions[::50]   (10 ps -> 500 ps)
  sequence, tica_* = unchanged from the current (fixed-TICA) subsampled file
Saved uncompressed like the other subsampled files; no velocities.

Checks, all must pass before anything is uploaded:
  * new positions == current 50k positions[::5]
  * new positions == positions of the ORIGINAL subsampled file (rev 4d232a6, 2025-08-15)
After the commit the remote LFS sha256 must equal the written file, and the file is
re-downloaded and its array digests re-checked. Records are appended to manifest.jsonl
(later lines supersede earlier ones for the same path).

    python helpers/resubsample_val8aa_hf.py --work-dir /network/scratch/t/tanc/hf_slim_many_peptides [--dry-run]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from slim_many_peptides_hf import REPO_ID, array_digest, file_sha256, lfs_sha_map  # noqa: E402

ORIGINAL_REV = "4d232a6"  # last revision with the original 10k-frame val subsampled files
PREFIX = "trajectories_subsampled/val/8AA/"
FULL_STRIDE = 50  # 10 ps -> 500 ps
N_FRAMES = 10_000


def dl(path, rev, d):
    return Path(hf_hub_download(REPO_ID, path, repo_type="dataset", revision=rev, local_dir=d))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--branch", default="no-velocities")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    api = HfApi()
    branch_sha = api.dataset_info(REPO_ID, revision=args.branch).sha
    work = args.work_dir / "fix_val8aa"
    shutil.rmtree(work, ignore_errors=True)
    staging = work / "staging"
    files = sorted(f for f in api.list_repo_files(REPO_ID, repo_type="dataset", revision=branch_sha)
                   if f.startswith(PREFIX) and f.endswith(".npz"))
    assert len(files) == 10, files
    print(f"branch {args.branch} @ {branch_sha}; {len(files)} files", flush=True)

    records = []
    for sub_path in files:
        seq = Path(sub_path).name.removesuffix("_subsampled.npz")
        full_path = f"trajectories/val/8AA/{seq}.npz"
        with np.load(dl(full_path, branch_sha, work / "dl"), allow_pickle=False) as d:
            full = d["positions"]
            if full.shape[0] != 500_000:
                raise RuntimeError(f"{full_path}: {full.shape[0]} frames")
            pos = np.ascontiguousarray(full[::FULL_STRIDE])
        cur_file = dl(sub_path, branch_sha, work / "dl")
        with np.load(cur_file, allow_pickle=False) as d:
            cur = {k: d[k] for k in d.files}
        with np.load(dl(sub_path, ORIGINAL_REV, work / "orig"), allow_pickle=False) as d:
            orig_pos = d["positions"]
            orig_seq = d["sequence"]

        if set(cur) != {"sequence", "positions", "tica_projection", "tica_mean", "tica_dim"}:
            raise RuntimeError(f"{sub_path}: unexpected current keys {sorted(cur)}")
        if pos.shape[0] != N_FRAMES:
            raise RuntimeError(f"{sub_path}: {pos.shape[0]} frames")
        if not np.array_equal(pos, cur["positions"][::5]):
            raise RuntimeError(f"{sub_path}: full[::50] != current 50k[::5]")
        if not np.array_equal(pos, orig_pos):
            raise RuntimeError(f"{sub_path}: full[::50] != original {ORIGINAL_REV} positions")
        if str(cur["sequence"]) != seq or str(orig_seq) != seq:
            raise RuntimeError(f"{sub_path}: sequence mismatch {cur['sequence']} / {orig_seq}")

        out = {k: cur[k] for k in ("sequence", "tica_projection", "tica_mean", "tica_dim")}
        out["positions"] = pos
        digests = {k: array_digest(v) for k, v in out.items()}
        dst = staging / sub_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **out)
        with np.load(dst, allow_pickle=False) as d:
            assert set(d.files) == set(out) and all(array_digest(d[k]) == digests[k] for k in out)

        records.append({
            "path": sub_path,
            "fix": "resubsampled from trajectories/val/8AA positions[::50]; tica/sequence kept",
            "source_revision": branch_sha,
            "source_sha256": file_sha256(cur_file),
            "source_size": cur_file.stat().st_size,
            "source_keys": sorted(cur),
            "source_shapes": {k: list(v.shape) for k, v in cur.items()},
            "full_trajectory_path": full_path,
            "stride": 1,
            "expected_frames": N_FRAMES,
            "compressed": False,
            "kept_digests": digests,
            "kept_shapes": {k: list(v.shape) for k, v in out.items()},
            "new_sha256": file_sha256(dst),
            "new_size": dst.stat().st_size,
        })
        print(f"  {sub_path}: {cur['positions'].shape[0]}->{pos.shape[0]} frames, matches {ORIGINAL_REV} original",
              flush=True)

    if args.dry_run:
        print("(dry-run) all checks passed, nothing uploaded")
        return

    api.upload_folder(repo_id=REPO_ID, repo_type="dataset", revision=args.branch, folder_path=staging,
                      commit_message="resubsampling val/8AA subsampled files to 10k frames from trajectories/")
    remote = lfs_sha_map(api, args.branch)
    bad = [r["path"] for r in records if remote.get(r["path"]) != r["new_sha256"]]
    if bad:
        raise RuntimeError(f"remote sha256 mismatch: {bad}")
    for r in records:  # independent re-download check
        with np.load(dl(r["path"], args.branch, work / "verify"), allow_pickle=False) as d:
            if {k: array_digest(d[k]) for k in d.files} != r["kept_digests"]:
                raise RuntimeError(f"{r['path']}: re-downloaded content mismatch")
    with (args.work_dir / "manifest.jsonl").open("a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"committed and verified {len(records)} files; manifest updated")


if __name__ == "__main__":
    main()
