"""
Uniform sampling for val and test datasets
==============================================
Retains every Nth frame within each video.
Preserves the natural distribution of the data without distortion.

Usage:
    python scripts\05_uniform_sampling.py --split val --type positives
    python scripts\05_uniform_sampling.py --split val --type negatives
    python scripts\05_uniform_sampling.py --split test --type positives
    python scripts\05_uniform_sampling.py --split test --type negatives
    python scripts\05_uniform_sampling.py --split val --type positives --dry-run
"""

import os
import re
import sys
import glob
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH"))
TRASH_DIR          = Path(os.getenv("TRASH_DIR"))

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# ================= КОНФІГУРАЦІЯ =================
# Для позитивів: кожен N-й кадр залишається
# None = не застосовується
SAMPLE_EVERY = {
    ("val",  "positives"): 4,    # 6035 → ~1500
    ("val",  "negatives"): 1,    # не ріжемо — тільки cap
    ("test", "positives"): 3,    # 6748 → ~2250
    ("test", "negatives"): 1,    # не ріжемо — тільки cap
}

# Для негативів: максимум кадрів з одного відео
# None = без обмеження
CAP_PER_VIDEO = {
    ("val",  "positives"): None,
    ("val",  "negatives"): 25,   # p75=24, захищає короткі відео
    ("test", "positives"): None,
    ("test", "negatives"): 45,   # p75=42, захищає короткі відео
}
# =================================================

def select_indices(n_frames: int, every_n: int, cap: int | None) -> set:
    """
    Повертає множину індексів кадрів які треба ЗАЛИШИТИ.

    Логіка:
    1. Якщо є cap і кадрів більше ніж cap — рівномірно відбираємо cap індексів
    2. Якщо є every_n — залишаємо кожен every_n-й кадр
    3. Якщо ні cap ні every_n — залишаємо всі
    """
    if n_frames == 0:
        return set()

    # Застосовуємо cap якщо є і відео довше за cap
    if cap is not None and n_frames > cap:
        indices = np.linspace(0, n_frames - 1, cap, dtype=int)
        return set(indices.tolist())

    # Застосовуємо every_n якщо є
    if every_n is not None and every_n > 1:
        return {i for i in range(n_frames) if i % every_n == 0}

    # Залишаємо всі
    return set(range(n_frames))

def uniform_sample(split: str, embedding_type: str, dry_run: bool = False):
    is_positive  = (embedding_type == "positives")
    subdir_name  = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"
    search_pattern = "*.txt" if is_positive else "*.jpg"

    target_dir = EXTRACTED_DATA_DIR / split / subdir_name
    if not target_dir.exists():
        log.error(f"Directory not found: {target_dir}")
        return

    every_n = SAMPLE_EVERY.get((split, embedding_type), 1)
    cap     = CAP_PER_VIDEO.get((split, embedding_type), None)

    all_files = sorted(glob.glob(str(target_dir / search_pattern)))
    total     = len(all_files)

    if total == 0:
        log.info(f"No files found in {target_dir}")
        return

    video_groups = {}
    for filepath_str in all_files:
        filepath = Path(filepath_str)
        v_id = re.sub(r'_f\d+$', '', filepath.stem)
        video_groups.setdefault(v_id, []).append(filepath)

    for v_id in video_groups:
        video_groups[v_id].sort()

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
        # Показуємо приклад для найбільшого відео
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

    # Створюємо папку trash
    split_trash = TRASH_DIR / split / f"uniform_sample_{embedding_type}"
    if split_trash.exists():
        shutil.rmtree(split_trash)
    split_trash.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    errors      = 0

    for video_name, files in tqdm(video_groups.items(),
                                   desc=f"Sampling {split}/{embedding_type}"):
        keep_idx = select_indices(len(files), every_n, cap)

        for i, file_path in enumerate(files):
            if i in keep_idx:
                continue

            try:
                img_path = file_path.with_suffix(".jpg")
                if img_path.exists():
                    shutil.move(str(img_path), split_trash / img_path.name)
                    moved_count += 1

                if is_positive:
                    txt_path = file_path.with_suffix(".txt")
                    if txt_path.exists():
                        shutil.move(str(txt_path), split_trash / txt_path.name)

            except OSError as e:
                log.error(f"Error: {file_path.name}: {e}")
                errors += 1

    remaining = total - moved_count
    log.info(f"\n[{split.upper()}] Sampling complete.")
    log.info(f"  Moved to trash: {moved_count}/{total}")
    log.info(f"  Remaining:         {remaining}")
    if errors:
        log.warning(f"  Errors: {errors}")


if __name__ == "__main__":
    if not os.getenv("EXTRACTED_DATA_PATH") or not os.getenv("TRASH_DIR"):
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