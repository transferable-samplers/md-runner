"""Upload a local folder of files to a Hugging Face Hub dataset repo."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_LOCAL_PATH = Path("/network/scratch/t/tanc/OLIGO_SMALL")
DEFAULT_REPO_ID = "transferable-samplers/oligo-prelim"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-path", type=Path, default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    if args.local_path.is_dir():
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=args.local_path,
            num_workers=args.num_workers,
        )
    else:
        api.upload_file(
            path_or_fileobj=args.local_path,
            path_in_repo=args.local_path.name,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
        )


if __name__ == "__main__":
    main()
