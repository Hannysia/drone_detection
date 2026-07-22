import os
import sys
import glob
import shutil
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config import IOU_THRESHOLD, SPLITS

EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
TRASH_DIR = Path(os.getenv("TRASH_DIR"))


logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def calculate_iou(box1, box2):
    """Calculates IoU for two YOLO boxes (cx, cy, w, h normalized)."""
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    ix1, iy1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    ix2, iy2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    iarea = (ix2 - ix1) * (iy2 - iy1)
    b1_area = box1[2] * box1[3]
    b2_area = box2[2] * box2[3]
    return iarea / (b1_area + b2_area - iarea + 1e-7)

def read_bbox(label_path):
    """Reads first bbox from YOLO label file. Returns [cx, cy, w, h]."""
    with open(label_path, 'r') as f:
        parts = f.readline().strip().split()
    return list(map(float, parts[1:5]))

def move_to_trash(label_path, trash_dir):
    """Moves label and corresponding image to trash."""
    label_path = Path(label_path)
    shutil.move(str(label_path), trash_dir / label_path.name)
    img_path = label_path.with_suffix('.jpg')
    if img_path.exists():
        shutil.move(str(img_path), trash_dir / img_path.name)

def run_iou_deduplication():
    log.info(f"Stage 1: Coarse IoU Deduplication")
    log.info(f"IoU threshold: {IOU_THRESHOLD}")

    for split in SPLITS:
        pos_dir = EXTRACTED_DATA_DIR / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            log.info(f"[{split.upper()}] Not found, skipping.")
            continue

        all_labels = sorted(glob.glob(str(pos_dir / "*.txt")))
        total_frames = len(all_labels)

        video_groups = {}
        for lbl in all_labels:
            v_id = "_f".join(os.path.basename(lbl).split("_f")[:-1])
            video_groups.setdefault(v_id, []).append(lbl)
        for v_id in video_groups:
            video_groups[v_id].sort()

        split_trash = TRASH_DIR / split
        if split_trash.exists():
            shutil.rmtree(split_trash)
        split_trash.mkdir(parents=True)

        moved_count = 0

        for video_name, labels in video_groups.items():
            anchor_box = read_bbox(labels[0])

            for lbl_path in labels[1:]:
                curr_box = read_bbox(lbl_path)

                if calculate_iou(anchor_box, curr_box) > IOU_THRESHOLD:
                    move_to_trash(lbl_path, split_trash)
                    moved_count += 1
                else:
                    anchor_box = curr_box

        log.info(
            f"[{split.upper()}] "
            f"Filtered: {moved_count}/{total_frames} frames moved to trash. "
            f"Remaining: {total_frames - moved_count}."
        )

if __name__ == "__main__":
    run_iou_deduplication()