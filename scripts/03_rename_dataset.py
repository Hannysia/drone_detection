import logging
import sys

from src.config import CLASSIFIED_DIR, RENAMED_DIR, VIDEO_EXTENSIONS
from src.processing.dataset import rename_dataset_videos_safely

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    if not CLASSIFIED_DIR.exists():
        log.error(f"Error: Source directory does not exist: {CLASSIFIED_DIR}")
        sys.exit(1)
        
    log.info("Starting dataset normalization...")
    log.info(f"Source: {CLASSIFIED_DIR}")
    log.info(f"Destination: {RENAMED_DIR}\n")
    
    rename_dataset_videos_safely(
        source_dir=CLASSIFIED_DIR,
        dest_dir=RENAMED_DIR,
        video_extensions=VIDEO_EXTENSIONS
    )
    
    log.info("\nDataset normalization finished successfully.")

if __name__ == "__main__":
    main()