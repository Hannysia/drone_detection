import logging
import sys

from src.config import EXTRACTED_DATA_DIR, FORCE_CLASS_ID, SPLITS
from src.processing.dataset import unify_labels_logic

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    if not EXTRACTED_DATA_DIR.exists():
        log.error(f"Error: Base dataset directory does not exist: {EXTRACTED_DATA_DIR}")
        sys.exit(1)
        
    unify_labels_logic(
        base_path=EXTRACTED_DATA_DIR,
        splits=SPLITS,
        force_class_id=FORCE_CLASS_ID
    )

if __name__ == "__main__":
    main()