import logging
from pathlib import Path
import shutil

from tqdm.auto import tqdm

log = logging.getLogger(__name__)

TRASH_FOLDER_NAMES = {
    "kcenter": "local_kcenter_{type}",
    "uniform": "uniform_sample_{type}",
    "iou": "iou_dedup_{type}",
}


def list_trash_folders_logic(trash_dir: Path) -> None:
    """Lists all available trash folders and file counts."""
    if not trash_dir.exists():
        log.error(f"TRASH_DIR does not exist: {trash_dir}")
        return

    log.info(f"\nAll trash folders in {trash_dir}:")
    log.info("=" * 60)

    found = False
    for split_dir in sorted(trash_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        for trash_subdir in sorted(split_dir.iterdir()):
            if not trash_subdir.is_dir():
                continue
            jpg_count = len(list(trash_subdir.glob("*.jpg")))
            txt_count = len(list(trash_subdir.glob("*.txt")))
            log.info(f"  {split_dir.name}/{trash_subdir.name}/")
            log.info(f"    .jpg: {jpg_count}  .txt: {txt_count}")
            found = True

    if not found:
        log.info("  (empty)")


def restore_from_trash_logic(
    split: str,
    embedding_type: str,
    stage: str,
    trash_base_dir: Path,
    extracted_data_dir: Path,
    dry_run: bool = False,
) -> None:
    """Moves back deleted files from a specific trash stage to the working directory."""
    is_positive = (embedding_type == "positives")
    subdir_name = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"

    folder_template = TRASH_FOLDER_NAMES.get(stage)
    if folder_template is None:
        log.error(f"Unknown stage: {stage}. Available: {list(TRASH_FOLDER_NAMES.keys())}")
        return

    trash_folder_name = folder_template.replace("{type}", embedding_type)
    trash_dir = trash_base_dir / split / trash_folder_name
    target_data_dir = extracted_data_dir / split / subdir_name

    if not trash_dir.exists():
        log.error(f"Trash directory not found: {trash_dir}")
        return

    if not target_data_dir.exists():
        log.error(f"Target directory not found: {target_data_dir}")
        return

    trash_files = sorted(list(trash_dir.glob("*.jpg")))
    total = len(trash_files)

    if total == 0:
        log.info(f"Trash is empty: {trash_dir}")
        return

    log.info(f"\nRestore: {split.upper()} / {embedding_type.upper()} / stage={stage}")
    log.info(f"  Source:      {trash_dir}")
    log.info(f"  Destination: {target_data_dir}")
    log.info(f"  Files .jpg:  {total}")

    if dry_run:
        log.info("\n[DRY RUN] Changes are not being applied. First 10 files:")
        for f in trash_files[:10]:
            log.info(f"  {f.name} → {target_data_dir / f.name}")
        if total > 10:
            log.info(f"  ... and {total - 10} more files")
        return

    restored_jpg = 0
    restored_txt = 0
    errors = 0

    for jpg_path in tqdm(trash_files, desc=f"Restoring {split}/{embedding_type}"):
        try:
            dest_jpg = target_data_dir / jpg_path.name
            shutil.move(str(jpg_path), dest_jpg)
            restored_jpg += 1

            if is_positive:
                txt_path = trash_dir / jpg_path.with_suffix(".txt").name
                if txt_path.exists():
                    shutil.move(str(txt_path), target_data_dir / txt_path.name)
                    restored_txt += 1

        except OSError as e:
            log.error(f"Error: {jpg_path.name}: {e}")
            errors += 1

    log.info(f"\n[{split.upper()}] Restore completed.")
    log.info(f"  Restored .jpg: {restored_jpg}/{total}")
    if is_positive:
        log.info(f"  Restored .txt: {restored_txt}")
    if errors:
        log.warning(f"  Errors: {errors}")


def unify_labels_logic(base_path: Path, splits: list[str], force_class_id: str) -> None:
    """Goes through all label files and forces the category class ID to force_class_id."""
    log.info(f"Starting label unification at: {base_path}")
    total_updated = 0

    for split in splits:
        label_dir = base_path / split / "RAW_POSITIVES"

        if not label_dir.exists():
            log.warning(f"Warning: Directory {label_dir} not found. Skipping {split}...")
            continue

        log.info(f"Processing split: {split}...")

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        parts[0] = force_class_id
                        updated_lines.append(" ".join(parts) + "\n")

                with open(label_file, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)

                total_updated += 1
            except Exception as e:
                log.error(f"Failed to overwrite label {label_file.name}: {e}")

    log.info("\n" + "=" * 50)
    log.info("UNIFICATION COMPLETE")
    log.info(f"Total files updated to Class ID {force_class_id}: {total_updated}")
    log.info("=" * 50)