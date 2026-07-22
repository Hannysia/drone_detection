import os
import sys
import glob
import shutil
import logging
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))

load_dotenv()

# Configuration
# BBoxes with absolute area less than 256 pixels² (e.g. 16x16px) will be filtered out
MIN_BBOX_AREA = 256.0  
SPLITS_TO_CLEAN = ["train"] # Only clean training split to protect validation/test natural distributions

EXTRACTED_DATA_PATH = Path(os.getenv("EXTRACTED_DATA_PATH"))
TRASH_DIR = Path(os.getenv("MT_TRASH_DIR"))

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def filter_label_file(lbl_path: Path, img_w: int, img_h: int, min_area: float) -> tuple[list[str], int]:
    """
    Parses a YOLO label file and filters out annotations smaller than the min_area.
    Returns:
        - List of preserved annotation strings
        - Number of deleted annotations
    """
    preserved_lines = []
    deleted_count = 0
    
    if not lbl_path.exists():
        return preserved_lines, deleted_count

    try:
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            class_id, cx, cy, nw, nh = map(float, parts)
            
            # Compute absolute dimensions
            pixel_w = nw * img_w
            pixel_h = nh * img_h
            pixel_area = pixel_w * pixel_h
            
            # Filter check
            if pixel_area >= min_area:
                preserved_lines.append(f"{int(class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            else:
                deleted_count += 1
                
    except Exception as e:
        log.error(f"Error parsing label file {lbl_path.name}: {e}")
        
    return preserved_lines, deleted_count


def move_frame_to_trash(lbl_path: Path, trash_dir: Path):
    """Moves both the text label and corresponding image to the micro-targets trash directory."""
    try:
        shutil.move(str(lbl_path), trash_dir / lbl_path.name)
        img_path = lbl_path.with_suffix(".jpg")
        if img_path.exists():
            shutil.move(str(img_path), trash_dir / img_path.name)
    except OSError as e:
        log.error(f"Failed to move {lbl_path.name} to trash: {e}")


def run_micro_target_filtering():
    log.info("Stage 4: Dataset Cleaning — Micro Target Filtering")
    log.info(f"Filtering out bounding boxes with absolute area < {MIN_BBOX_AREA} px² (approx. 16x16px)")

    for split in SPLITS_TO_CLEAN:
        pos_dir = EXTRACTED_DATA_PATH / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            log.warning(f"[{split.upper()}] Directory not found: {pos_dir}, skipping.")
            continue

        all_images = sorted(glob.glob(str(pos_dir / "*.jpg")))
        total_images = len(all_images)
        
        # Create dedicated trash folder for this stage
        split_trash_dir = TRASH_DIR / split / "micro_targets"
        if split_trash_dir.exists():
            shutil.rmtree(split_trash_dir)
        split_trash_dir.mkdir(parents=True, exist_ok=True)

        removed_frames_count = 0
        modified_labels_count = 0
        total_deleted_bboxes = 0

        for img_path_str in tqdm(all_images, desc=f"Filtering {split.upper()} split"):
            img_path = Path(img_path_str)
            lbl_path = img_path.with_suffix(".txt")

            if not lbl_path.exists():
                continue

            # 1. Read dimensions quickly using PIL header scanner
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception as e:
                log.error(f"Failed to read image header {img_path.name}: {e}")
                continue

            # 2. Parse and filter annotations
            preserved_annotations, deleted_bboxes = filter_label_file(lbl_path, img_w, img_h, MIN_BBOX_AREA)
            total_deleted_bboxes += deleted_bboxes

            if deleted_bboxes > 0:
                if len(preserved_annotations) == 0:
                    # Case A: Frame contained ONLY micro targets. Move entire frame to trash.
                    move_frame_to_trash(lbl_path, split_trash_dir)
                    removed_frames_count += 1
                else:
                    # Case B: Some labels preserved. Rewrite .txt with only valid bboxes.
                    try:
                        with open(lbl_path, 'w') as f:
                            f.writelines(preserved_annotations)
                        modified_labels_count += 1
                    except Exception as e:
                        log.error(f"Failed to overwrite {lbl_path.name}: {e}")

        log.info(f"\n[{split.upper()}] Cleaning Summary:")
        log.info(f"  - Total deleted micro bounding boxes: {total_deleted_bboxes}")
        log.info(f"  - Modified labels (removed tiny boxes, kept images): {modified_labels_count}")
        log.info(f"  - Removed empty frames (moved to trash): {removed_frames_count}/{total_images}")
        log.info(f"  - Remaining active frames: {total_images - removed_frames_count}")


if __name__ == "__main__":
    # Environment safeguard checks
    if not os.getenv("EXTRACTED_DATA_PATH") or not os.getenv("TRASH_DIR"):
        log.error("Error: Required environment variables not loaded. Ensure .env is present.")
        sys.exit(1)
        
    run_micro_target_filtering()