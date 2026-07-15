from huggingface_hub import list_repo_files
from huggingface_hub.utils import RepositoryNotFoundError

MODEL_IDS = [
    "lmsys/vicuna-13b-v0",
    "lmsys/vicuna-13b-v1.1",
    "lmsys/vicuna-13b-v1.3",
    "lmsys/vicuna-13b-v1.5",
]

for mid in MODEL_IDS:
    try:
        files = list_repo_files(mid, repo_type="model")
        bin_files = [f for f in files if f.endswith('.bin')]
        print(f"OK: {mid} -> {len(bin_files)} bin files, total files: {len(files)}")
        if bin_files:
            for f in bin_files[:5]:
                print(f"  - {f}")
    except RepositoryNotFoundError:
        print(f"NOT FOUND: {mid}")
    except Exception as e:
        print(f"ERROR {mid}: {type(e).__name__}: {e}")
