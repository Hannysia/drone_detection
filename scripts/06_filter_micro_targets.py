import logging
import os
import shutil
import sys
from PIL import Image
from tqdm.auto import tqdm

from src.config import MIN_BBOX_AREA, EXTRACTED_DATA_DIR, MT_TRASH_DIR
from src.utils.io import move_to_trash
from src.processing.filters import filter_label_file

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

SPLITS_TO_CLEAN = ['train']

def run_micro_target_filtering() -> None:
    log.info("Stage 4: Dataset Cleaning — Micro Target Filtering")
    log.info(f"Filtering out bounding boxes with absolute area < {MIN_BBOX_AREA} px²")

    for split in SPLITS_TO_CLEAN:
        pos_dir = EXTRACTED_DATA_DIR / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            log.warning(f"[{split.upper()}] Directory not found: {pos_dir}, skipping.")
            continue

        all_images = sorted(list(pos_dir.glob("*.jpg")))
        total_images = len(all_images)
        
        split_trash_dir = MT_TRASH_DIR / split / "micro_targets"
        if split_trash_dir.exists():
            shutil.rmtree(split_trash_dir)
        split_trash_dir.mkdir(parents=True, exist_ok=True)

        removed_frames_count = 0
        modified_labels_count = 0
        total_deleted_bboxes = 0

        for img_path in tqdm(all_images, desc=f"Filtering {split.upper()} split"):
            lbl_path = img_path.with_suffix(".txt")

            if not lbl_path.exists():
                continue

            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                log.error(f"Failed to read image header {img_path.name}: {e}")
                continue

            preserved_annotations, deleted_bboxes = filter_label_file(
                lbl_path, img_w, img_h, MIN_BBOX_AREA
            )
            total_deleted_bboxes += deleted_bboxes

            if deleted_bboxes > 0:
                if len(preserved_annotations) == 0:
                    try:
                        move_to_trash(lbl_path, split_trash_dir, is_positive=True)
                        removed_frames_count += 1
                    except OSError as e:
                        log.error(e)
                else:
                    try:
                        with open(lbl_path, 'w', encoding='utf-8') as f:
                            f.writelines(preserved_annotations)
                        modified_labels_count += 1
                    except Exception as e:
                        log.error(f"Failed to overwrite {lbl_path.name}: {e}")

        log.info(f"\n[{split.upper()}] Cleaning Summary:")
        log.info(f"  - Total deleted micro bounding boxes: {total_deleted_bboxes}")
        log.info(f"  - Modified labels (kept images, removed tiny boxes): {modified_labels_count}")
        log.info(f"  - Removed empty frames (moved to trash): {removed_frames_count}/{total_images}")
        log.info(f"  - Remaining active frames: {total_images - removed_frames_count}")

if __name__ == "__main__":
    if not EXTRACTED_DATA_DIR or not MT_TRASH_DIR:
        log.error("Error: Required environment variables not loaded. Ensure .env is present.")
        sys.exit(1)
        
    run_micro_target_filtering()