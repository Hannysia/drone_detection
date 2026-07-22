import os
import sys
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Setup paths to allow importing from 'src'
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

REPO_ID = os.getenv("REPO_ID")
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "./embeddings"))

# Constant - we only ever work with the 'train' split
SPLIT = "train"


def download_embeddings(embedding_type: str):
    if not REPO_ID:
        print("Error: REPO_ID is not defined in your .env file!")
        return

    # Target directory (e.g., E:/bachelor_data/embeddings/positives)
    target_dir = EMBEDDINGS_DIR / embedding_type
    target_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{SPLIT}_{embedding_type}_embeddings.pt"
    # Remote path inside the Hugging Face repository
    hf_file_path = f"embeddings/{file_name}"
    
    # Final clean local path for the file
    final_local_path = target_dir / file_name

    print(f"\nDownloading from Hugging Face: {hf_file_path}")
    
    try:
        # 1. Download to the default global system cache (outside of the project)
        cached_file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_file_path,
            repo_type="dataset"
        )
        
        # 2. Copy only the raw .pt file to the final destination
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