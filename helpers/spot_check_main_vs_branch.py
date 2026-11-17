"""Compare every target .npz between the original main (OLD) and no-velocities (NEW).

    python helpers/spot_check_main_vs_branch.py          # all files (~11 TB of downloads)
    python helpers/spot_check_main_vs_branch.py 10       # 10 random files from each of the 19 directories
"""

import random
import sys
import tempfile

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "transferable-samplers/many-peptides-md"
OLD = "1af9336878122eb1d62894fe2fb3ff4b801a3216"  # main before the rewrite (2025-12-15)
NEW = "a106995"                                   # no-velocities after the val/8AA fix

files = [f for f in list_repo_files(REPO, repo_type="dataset", revision=NEW)
         if f.endswith(".npz") and f.startswith(("trajectories/", "trajectories_subsampled/"))]
if len(sys.argv) > 1:  # N random files from each directory, e.g. trajectories/train/4AA
    n = int(sys.argv[1])
    by_dir = {}
    for f in files:
        by_dir.setdefault("/".join(f.split("/")[:3]), []).append(f)
    files = [f for group in by_dir.values() for f in random.sample(group, min(n, len(group)))]

for path in files:
    if path.startswith("trajectories/train/"):
        stride = 10  # 200k -> 20k frames
    elif path.startswith("trajectories_subsampled/val/8AA/"):
        stride = 5   # 50k on main -> 10k
    else:
        stride = 1

    with tempfile.TemporaryDirectory() as tmp:
        old = np.load(hf_hub_download(REPO, path, repo_type="dataset", revision=OLD, local_dir=tmp + "/old"))
        new = np.load(hf_hub_download(REPO, path, repo_type="dataset", revision=NEW, local_dir=tmp + "/new"))

        assert set(new.files) == set(old.files) - {"velocities"}, path
        for key in new.files:
            expected = old[key][::stride] if key == "positions" else old[key]
            assert new[key].dtype == expected.dtype, (path, key)
            assert np.array_equal(new[key], expected), (path, key)

    print("ok", path, flush=True)

print(f"all {len(files)} files ok")
