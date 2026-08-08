import argparse
import logging
import sys

from src.config import EXTRACTED_DATA_DIR, TRASH_DIR
from src.processing.dataset import list_trash_folders_logic, restore_from_trash_logic

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def main() -> None:
    if not EXTRACTED_DATA_DIR or not TRASH_DIR:
        log.error("Error: Environment variables not loaded. Check .env")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Restore files from trash back to the dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Split: train, val or test"
    )
    parser.add_argument(
        "-t", "--type",
        choices=["positives", "negatives"],
        help="Data type: positives or negatives"
    )
    parser.add_argument(
        "--stage",
        choices=["kcenter", "uniform", "iou"],
        default="kcenter",
        help=(
            "Which stage to restore:\n"
            "  kcenter — k-Center deduplication (train)\n"
            "  uniform — uniform sampling (val/test)\n"
            "  iou     — IoU filter"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what will be restored without actually changing files."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show all available trash folders and exit."
    )

    args = parser.parse_args()

    if args.list:
        list_trash_folders_logic(TRASH_DIR)
    else:
        if not args.split or not args.type:
            parser.error("--split and --type are required unless --list is specified.")
        
        restore_from_trash_logic(
            split=args.split,
            embedding_type=args.type,
            stage=args.stage,
            trash_base_dir=TRASH_DIR,
            extracted_data_dir=EXTRACTED_DATA_DIR,
            dry_run=args.dry_run
        )

if __name__ == "__main__":
    main()