from huggingface_hub import HfApi

api = HfApi()  # uses your saved login, or pass token="hf_..."
REPO = "transferable-samplers/many-peptides-md"

# api.super_squash_history(REPO, branch="main", repo_type="dataset")
# api.delete_branch(REPO, branch="no-velocities", repo_type="dataset")
# api.delete_branch(REPO, branch="fixing-tica", repo_type="dataset")

# files = list(api.list_lfs_files(REPO, repo_type="dataset"))
# print(sum(f.size for f in files) / 1e9, "GB")
