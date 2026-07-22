import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

# ================= CONFIGURATION =================
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("REPO_ID")
EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
TEMP_DIR = Path(os.getenv("TEMP_DIR"))
SUBDIRS_TO_PACK = ["RAW_POSITIVES"] 

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SPLITS
# =================================================

os.makedirs(TEMP_DIR, exist_ok=True)

def create_zip_archive(split_name, source_dir, output_zip_path):
    print(f"\n[{split_name.upper()}] Archiving files...")
    
    file_count = 0
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for subdir in SUBDIRS_TO_PACK:
            target_path = source_dir / subdir
            if not target_path.exists():
                print(f"Directory {target_path} does not exist, skipping.")
                continue
                
            for file_path in target_path.glob('*'):
                if file_path.suffix in ['.jpg', '.txt']:
                    arcname = os.path.join(subdir, file_path.name)
                    zipf.write(file_path, arcname)
                    file_count += 1
                    
    print(f"[{split_name.upper()}] Archive created: {output_zip_path} ({file_count} files packed)")
    return file_count > 0

def upload_to_hf(file_path, repo_id, token, path_in_repo):
    """
    Uploads a single file to Hugging Face dataset repository.
    """
    api = HfApi()
    print(f"Uploading {file_path.name} to Hugging Face dataset '{repo_id}'...")
    
    try:
        response = api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        print(f"Successfully uploaded: {response}")
    except Exception as e:
        print(f"Error during upload: {e}")

if __name__ == "__main__":
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN not found in environment variables. Please check your .env file.")

    for split in SPLITS:
        split_source_dir = EXTRACTED_DATA_DIR / split
        
        if not split_source_dir.exists():
            print(f"Directory for split '{split}' not found at {split_source_dir}, skipping.")
            continue
            
        zip_file_name = f"{split}_data.zip"
        local_zip_path = TEMP_DIR / zip_file_name
        
        success = create_zip_archive(split, split_source_dir, local_zip_path)
        
        if success:
            path_in_hf_repo = f"temp_archives/{zip_file_name}"
            upload_to_hf(local_zip_path, REPO_ID, HF_TOKEN, path_in_hf_repo)
            
            try:
                os.remove(local_zip_path)
                print(f"Removed local temporary file: {local_zip_path}")
            except OSError as e:
                print(f"Could not remove temporary local file {local_zip_path}: {e}")
                
    print("\nAll tasks completed.")