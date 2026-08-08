import logging
import sys
from huggingface_hub import HfApi
from pathlib import Path

from src.config import (
    HF_TOKEN, REPO_ID, EXTRACTED_DATA_DIR, TEMP_DIR, SUBDIRS_TO_PACK, SPLITS
)
from src.utils.io import create_zip_archive

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def upload_to_hf(file_path: Path, repo_id: str, token: str, path_in_repo: str) -> None:
    """Uploads a single file to Hugging Face dataset repository."""
    api = HfApi(token=token)
    log.info(f"Uploading {file_path.name} to Hugging Face dataset '{repo_id}'...")
    
    try:
        response = api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        log.info(f"Successfully uploaded: {response.filename}")
    except Exception as e:
        log.error(f"Error during upload: {e}")

def main() -> None:
    if not HF_TOKEN:
        log.error("Error: HF_TOKEN not found in environment variables. Check .env")
        sys.exit(1)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_source_dir = EXTRACTED_DATA_DIR / split
        
        if not split_source_dir.exists():
            log.warning(f"Directory for split '{split}' not found at {split_source_dir}, skipping.")
            continue
            
        zip_file_name = f"{split}_data.zip"
        local_zip_path = TEMP_DIR / zip_file_name
        
        success = create_zip_archive(
            split_name=split,
            source_dir=split_source_dir,
            output_zip_path=local_zip_path,
            subdirs=SUBDIRS_TO_PACK
        )
        
        if success:
            path_in_hf_repo = f"temp_archives/{zip_file_name}"
            upload_to_hf(local_zip_path, REPO_ID, HF_TOKEN, path_in_hf_repo)
            
            try:
                local_zip_path.unlink()
                log.info(f"Removed local temporary file: {local_zip_path}")
            except OSError as e:
                log.error(f"Could not remove temporary local file {local_zip_path}: {e}")
                
    log.info("\nAll backup tasks completed.")

if __name__ == "__main__":
    main()