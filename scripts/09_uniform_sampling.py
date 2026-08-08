import argparse
import logging
import shutil
import sys
from tqdm.auto import tqdm

from src.config import EXTRACTED_DATA_DIR, TRASH_DIR, SAMPLE_EVERY, CAP_PER_VIDEO
from src.utils.io import move_to_trash
from src.processing.deduplication import group_files_by_video
from src.processing.sampling import select_indices

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

def uniform_sample(split: str, embedding_type: str, dry_run: bool = False) -> None:
    is_positive = (embedding_type == "positives")
    subdir_name = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"
    search_pattern = "*.txt" if is_positive else "*.jpg"

    target_dir = EXTRACTED_DATA_DIR / split / subdir_name
    if not target_dir.exists():
        log.error(f"Directory not found: {target_dir}")
        return

    every_n = SAMPLE_EVERY.get((split, embedding_type), 1)
    cap = CAP_PER_VIDEO.get((split, embedding_type), None)

    all_files = sorted(list(target_dir.glob(search_pattern)))
    total = len(all_files)

    if total == 0:
        log.info(f"No files found in {target_dir}")
        return

    video_groups = group_files_by_video(all_files)

    keep_total = 0
    for files in video_groups.values():
        keep_idx = select_indices(len(files), every_n, cap)
        keep_total += len(keep_idx)

    log.info(f"\nUniform sampling: {split.upper()} / {embedding_type.upper()}")
    log.info(f"Directory: {target_dir}")
    log.info(f"Total files: {total}")
    log.info(f"Unique videos: {len(video_groups)}")

    if cap:
        log.info(f"  Cap на відео:  {cap} кадрів")
    if every_n and every_n > 1:
        log.info(f"Keeping every {every_n}th frame")
    log.info(f"Expected to remain: ~{keep_total} frames")
    log.info(f"Expected to go to the trash: ~{total - keep_total} frames")

    if dry_run:
        log.info("\n[DRY RUN] Changes are not being applied.")
        largest_vid = max(video_groups, key=lambda v: len(video_groups[v]))
        largest_files = video_groups[largest_vid]
        keep_idx = select_indices(len(largest_files), every_n, cap)
        log.info(f"\nExample for the largest video '{largest_vid}'"
                 f" ({len(largest_files)} frames → {len(keep_idx)} frames remain):")
        for i, f in enumerate(largest_files[:12]):
            action = "KEEP " if i in keep_idx else "trash"
            log.info(f"  [{i:3d}] {action} — {f.name}")
        if len(largest_files) > 12:
            log.info(f"  ... and another {len(largest_files) - 12} frames")
        return

    split_trash = TRASH_DIR / split / f"uniform_sample_{embedding_type}"
    if split_trash.exists():
        shutil.rmtree(split_trash)
    split_trash.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    errors = 0

    for video_name, files in tqdm(video_groups.items(), desc=f"Sampling {split}/{embedding_type}"):
        keep_idx = select_indices(len(files), every_n, cap)

        for i, file_path in enumerate(files):
            if i in keep_idx:
                continue

            try:
                move_to_trash(file_path, split_trash, is_positive=is_positive)
                moved_count += 1
            except OSError as e:
                log.error(e)
                errors += 1

    remaining = total - moved_count
    log.info(f"\n[{split.upper()}] Sampling complete.")
    log.info(f"  Moved to trash: {moved_count}/{total}")
    log.info(f"  Remaining:         {remaining}")
    if errors:
        log.warning(f"  Errors: {errors}")

if __name__ == "__main__":
    if not EXTRACTED_DATA_DIR or not TRASH_DIR:
        log.error("Error: Environment variables not loaded.")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Uniform sampling of val/test samples from cap in the video."
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        required=True,
        help="Sample: val or test"
    )
    parser.add_argument(
        "-t", "--type",
        choices=["positives", "negatives"],
        required=True,
        help="Data type: positives or negatives"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what will be done without making any actual changes"
    )

    args = parser.parse_args()
    uniform_sample(
        split=args.split,
        embedding_type=args.type,
        dry_run=args.dry_run,
    )