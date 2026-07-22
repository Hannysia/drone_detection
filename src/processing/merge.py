import os
import shutil
from pathlib import Path

# ================= CONFIGURATION =================
# Source directory (your temporary staging folder)
SOURCE_DIR = Path(r"E:\backup\extracted_dataset_new")

# Destination directory (your main working dataset folder)
DEST_DIR   = Path(r"E:\backup\extracted_dataset")

SPLITS     = ["train", "val", "test"]
SUBDIR     = "RAW_NEGATIVES"
# =================================================


def merge_datasets():
    if not SOURCE_DIR.exists():
        print(f"Error: Source directory does not exist: {SOURCE_DIR}")
        return

    print("=" * 80)
    print("STARTING DATASET MERGE PROCESS")
    print("=" * 80)
    print(f"Moving files from: {SOURCE_DIR}")
    print(f"Moving files into: {DEST_DIR}")
    print("-" * 80)

    total_moved = 0

    for split in SPLITS:
        src_split_dir = SOURCE_DIR / split / SUBDIR
        dest_split_dir = DEST_DIR / split / SUBDIR

        if not src_split_dir.exists():
            print(f"No files found for split '{split}' in staging directory. Skipping...")
            continue

        # Create destination directory if it doesn't exist yet
        dest_split_dir.mkdir(parents=True, exist_ok=True)

        # Find all files in the staging subdir
        files_to_move = list(src_split_dir.glob("*"))
        if not files_to_move:
            print(f"Split '{split}' subdirectory is empty. Skipping...")
            continue

        print(f"Moving {len(files_to_move):,} files for split: {split.upper()}...")
        
        split_moved_count = 0
        for file_path in files_to_move:
            dest_file_path = dest_split_dir / file_path.name
            
            # Safe move: if file already exists in destination, it will be skipped/warned
            if dest_file_path.exists():
                print(f"  ⚠️ Warning: File already exists in destination, skipping: {file_path.name}")
                continue

            try:
                shutil.move(str(file_path), str(dest_file_path))
                split_moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving file {file_path.name}: {e}")

        print(f"  └─ Success: Transferred {split_moved_count:,} files for {split.upper()}")
        total_moved += split_moved_count

    print("\n" + "=" * 80)
    print(f"MERGE COMPLETED. Successfully transferred {total_moved:,} files.")
    print("=" * 80)


if __name__ == "__main__":
    merge_datasets()