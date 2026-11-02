"""Independently verify the slimmed many-peptides-md files against the manifest.

Run this before deleting/squashing the original data. For every manifest entry:
  * the file at --revision exists and its LFS sha256 == manifest new_sha256;
  * every manifest path is present, and every .npz under the target prefixes on
    the remote is in the manifest (nothing missed);
With --redownload (recommended before deletion; ~0.6 TB) additionally:
  * the downloaded bytes hash to new_sha256;
  * keys == original keys minus `velocities`;
  * each array's content digest == the digest computed from the ORIGINAL file;
  * positions frame count == ceil(original frames / stride).

    python helpers/verify_many_peptides_hf.py --manifest <work-dir>/manifest.jsonl --redownload --workers 16
"""

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from slim_many_peptides_hf import PREFIXES, REPO_ID, array_digest, expected_source, file_sha256, lfs_sha_map  # noqa: E402


def check_content(rec: dict, revision: str, tmp_root: str, stride: int) -> list[str]:
    errs = []
    with tempfile.TemporaryDirectory(dir=tmp_root) as td:
        f = Path(hf_hub_download(REPO_ID, rec["path"], repo_type="dataset", revision=revision, local_dir=td))
        if file_sha256(f) != rec["new_sha256"]:
            errs.append("downloaded sha256 mismatch")
        with np.load(f, allow_pickle=False) as d:
            # absolute expectations, independent of what the manifest says
            src_keys, src_frames = expected_source(rec["path"])
            expected_keys = src_keys - {"velocities"}
            if set(d.files) != expected_keys:
                errs.append(f"keys {sorted(d.files)} != {sorted(expected_keys)}")
            if set(rec["source_keys"]) != src_keys:
                errs.append(f"manifest source keys {rec['source_keys']} != {sorted(src_keys)}")
            if rec["source_shapes"]["positions"][0] != src_frames:
                errs.append(f"manifest source frames {rec['source_shapes']['positions'][0]} != {src_frames}")
            is_train = rec["path"].startswith("trajectories/train/")
            if rec["stride"] != (stride if is_train else 1):
                errs.append(f"manifest stride {rec['stride']} unexpected")
            n_exp = src_frames // (stride if is_train else 1)
            if "positions" in d.files and d["positions"].shape[0] != n_exp:
                errs.append(f"positions frames {d['positions'].shape[0]} != {n_exp}")
            for k, dig in rec["kept_digests"].items():
                if k in d.files and array_digest(d[k]) != dig:
                    errs.append(f"content digest mismatch: {k}")
    return errs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--redownload", action="store_true")
    p.add_argument("--stride", type=int, default=10, help="expected train subsampling stride")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tmp-dir", default=None, help="where to put temporary downloads")
    args = p.parse_args()

    recs = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    by_path = {r["path"]: r for r in recs}
    remote = lfs_sha_map(HfApi(), args.revision)
    failures = {}

    remote_npz = {f for f in remote if f.endswith(".npz") and f.startswith(PREFIXES)}
    for f in sorted(remote_npz - by_path.keys()):
        failures[f] = ["on remote but not in manifest (not processed)"]
    for path, r in by_path.items():
        if path not in remote:
            failures[path] = ["missing on remote"]
        elif remote[path] != r["new_sha256"]:
            failures[path] = ["remote LFS sha256 != manifest new_sha256"]
    print(f"metadata check: {len(by_path)} manifest entries, {len(failures)} failures", flush=True)

    if args.redownload:
        todo = [r for r in recs if r["path"] not in failures]
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(check_content, r, args.revision, args.tmp_dir, args.stride): r["path"] for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                path = futs[fut]
                try:
                    errs = fut.result()
                except Exception as e:  # noqa: BLE001
                    errs = [f"exception: {e!r}"]
                if errs:
                    failures[path] = errs
                    print(f"FAIL {path}: {errs}", flush=True)
                if i % 500 == 0:
                    print(f"  content-checked {i}/{len(todo)}", flush=True)

    if failures:
        for path, errs in sorted(failures.items())[:50]:
            print(f"FAIL {path}: {errs}")
        print(f"\n{len(failures)} FAILURES")
        sys.exit(1)
    print(f"\nALL OK: {len(by_path)} files verified" + (" (content re-downloaded)" if args.redownload else ""))


if __name__ == "__main__":
    main()
