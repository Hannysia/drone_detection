import argparse
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

from src.config import REPO_ID, EMBEDDINGS_DIR, HF_TOKEN

SPLIT = "train"

def download_embeddings(embedding_type: str) -> None:
    if not REPO_ID:
        print("Error: REPO_ID is not defined in your config/env!")
        return

    target_dir = EMBEDDINGS_DIR / embedding_type
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{SPLIT}_{embedding_type}_embeddings.pt"
    hf_file_path = f"embeddings/{file_name}"
    
    final_local_path = target_dir / file_name

    print(f"\nDownloading from Hugging Face: {hf_file_path}")
    
    try:
        cached_file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_file_path,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        shutil.copy(cached_file_path, final_local_path)
        
        print(f"[SUCCESS] File saved directly to: {final_local_path.resolve()}")
        
    except Exception as e:
        print(f"[ERROR] Failed to download {file_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download train embeddings directly as a flat file."
    )
    parser.add_argument(
        "-t", "--type",
        choices=["positives", "negatives"],
        required=True,
        help="Type of embeddings to download: 'positives' or 'negatives'"
    )

    args = parser.parse_args()
    download_embeddings(embedding_type=args.type)