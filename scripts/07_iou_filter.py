import logging
import shutil

from src.config import IOU_THRESHOLD, SPLITS, EXTRACTED_DATA_DIR, TRASH_DIR
from src.processing.deduplication import group_labels_by_video, deduplicate_video_group

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def run_iou_deduplication() -> None:
    log.info("Stage 1: Coarse IoU Deduplication")
    log.info(f"IoU threshold: {IOU_THRESHOLD}")

    for split in SPLITS:
        pos_dir = EXTRACTED_DATA_DIR / split / "RAW_POSITIVES"
        if not pos_dir.exists():
            log.info(f"[{split.upper()}] Not found, skipping.")
            continue

        all_labels = sorted(list(pos_dir.glob("*.txt")))
        total_frames = len(all_labels)

        if total_frames == 0:
            log.info(f"[{split.upper()}] No labels found, skipping.")
            continue

        video_groups = group_labels_by_video(all_labels)

        split_trash = TRASH_DIR / split
        if split_trash.exists():
            shutil.rmtree(split_trash)
        split_trash.mkdir(parents=True)

        total_moved = 0
        for video_name, labels in video_groups.items():
            total_moved += deduplicate_video_group(labels, IOU_THRESHOLD, split_trash)

        log.info(
            f"[{split.upper()}] "
            f"Filtered: {total_moved}/{total_frames} frames moved to trash. "
            f"Remaining: {total_frames - total_moved}."
        )

if __name__ == "__main__":
    run_iou_deduplication()