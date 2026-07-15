"""
Resume-capable download script for Vicuna 13B weights (bin files only).
Uses v1.1 as the closest available version to v0.
"""
import os
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

LOCAL_DIR = r".\MiniGPT-4\vicuna_13b"
MODEL_ID = "lmsys/vicuna-13b-v1.1"

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    existing = set(os.listdir(LOCAL_DIR))
    bin_files = [f for f in existing if f.endswith('.bin')]
    if bin_files:
        total = sum(os.path.getsize(os.path.join(LOCAL_DIR, f)) for f in bin_files)
        print(f"Existing: {bin_files}, total={total/1024**3:.2f} GB")
        if total >= 24 * 1024**3:
            print("Weights look complete (~24GB+). Done.")
            return 0
    
    print(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
    print("Only downloading .bin weight files (preserving existing config/tokenizer)")
    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.bin"],
        )
        print("Download complete!")
        return 0
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
