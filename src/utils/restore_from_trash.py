"""
Відновлення кадрів зі смітника назад у датасет
===============================================
Підтримує всі типи папок trash:
  - local_kcenter_{type}     ← k-Center дедуплікація (train)
  - uniform_sample_{type}    ← рівномірне семплування (val/test)
  - iou_dedup_{type}         ← IoU фільтр (будь-який split)

Використання:
  # Відновити після k-Center (train)
  python restore_from_trash.py --split train -t negatives --stage kcenter

  # Відновити після рівномірного семплування (val/test)
  python restore_from_trash.py --split val -t positives --stage uniform
  python restore_from_trash.py --split test -t negatives --stage uniform

  # Dry-run (без реальних змін)
  python restore_from_trash.py --split val -t positives --stage uniform --dry-run

  # Показати всі доступні папки trash
  python restore_from_trash.py --list
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f".env завантажено з {env_path}")
    else:
        load_dotenv()
except ImportError:
    pass

EXTRACTED_DATA_DIR = Path(os.getenv("EXTRACTED_DATA_PATH", ""))
TRASH_DIR          = Path(os.getenv("TRASH_DIR", ""))

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Мапи назв папок trash для кожного stage
TRASH_FOLDER_NAMES = {
    "kcenter": "local_kcenter_{type}",
    "uniform": "uniform_sample_{type}",
    "iou":     "iou_dedup_{type}",
}


def list_trash_folders():
    """Показує всі наявні папки trash з кількістю файлів."""
    if not TRASH_DIR.exists():
        log.error(f"TRASH_DIR не існує: {TRASH_DIR}")
        return

    log.info(f"\nВсі папки trash в {TRASH_DIR}:")
    log.info("=" * 60)

    found = False
    for split_dir in sorted(TRASH_DIR.iterdir()):
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
        log.info("  (порожньо)")


def restore_from_trash(split: str, embedding_type: str,
                       stage: str, dry_run: bool = False):
    is_positive = (embedding_type == "positives")
    subdir_name = "RAW_POSITIVES" if is_positive else "RAW_NEGATIVES"

    # Визначаємо папку trash
    folder_template = TRASH_FOLDER_NAMES.get(stage)
    if folder_template is None:
        log.error(f"Невідомий stage: {stage}. Доступні: {list(TRASH_FOLDER_NAMES.keys())}")
        return

    trash_folder_name = folder_template.replace("{type}", embedding_type)
    trash_dir         = TRASH_DIR / split / trash_folder_name
    target_data_dir   = EXTRACTED_DATA_DIR / split / subdir_name

    # Перевіряємо наявність папок
    if not trash_dir.exists():
        log.error(f"Смітник не знайдено: {trash_dir}")
        log.info(f"Підказка: запусти --list щоб побачити всі доступні папки")
        return

    if not target_data_dir.exists():
        log.error(f"Цільова директорія не знайдена: {target_data_dir}")
        return

    # Збираємо файли
    trash_files = sorted(trash_dir.glob("*.jpg"))
    total       = len(trash_files)

    if total == 0:
        log.info(f"Смітник порожній: {trash_dir}")
        return

    log.info(f"\nВідновлення: {split.upper()} / {embedding_type.upper()} / stage={stage}")
    log.info(f"  Смітник:     {trash_dir}")
    log.info(f"  Призначення: {target_data_dir}")
    log.info(f"  Файлів .jpg: {total}")

    if dry_run:
        log.info("\n[DRY RUN] Зміни не застосовуються. Перші 10 файлів:")
        for f in trash_files[:10]:
            log.info(f"  {f.name} → {target_data_dir / f.name}")
        if total > 10:
            log.info(f"  ... і ще {total - 10} файлів")
        return

    restored_jpg = 0
    restored_txt = 0
    errors       = 0

    for jpg_path in tqdm(trash_files, desc=f"Відновлення {split}/{embedding_type}"):
        try:
            dest_jpg = target_data_dir / jpg_path.name
            shutil.move(str(jpg_path), dest_jpg)
            restored_jpg += 1

            if is_positive:
                txt_path = trash_dir / jpg_path.with_suffix(".txt").name
                if txt_path.exists():
                    shutil.move(str(txt_path),
                                target_data_dir / txt_path.name)
                    restored_txt += 1

        except OSError as e:
            log.error(f"Помилка: {jpg_path.name}: {e}")
            errors += 1

    log.info(f"\n[{split.upper()}] Відновлення завершено.")
    log.info(f"  Відновлено .jpg: {restored_jpg}/{total}")
    if is_positive:
        log.info(f"  Відновлено .txt: {restored_txt}")
    if errors:
        log.warning(f"  Помилок: {errors}")

    remaining = list(trash_dir.glob("*"))
    if not remaining:
        log.info(f"  Смітник очищено: {trash_dir}")
    else:
        log.warning(f"  Залишилось у смітнику: {len(remaining)} файлів")


if __name__ == "__main__":
    if not os.getenv("EXTRACTED_DATA_PATH") or not os.getenv("TRASH_DIR"):
        log.error("Помилка: змінні середовища не завантажені. Перевір .env")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Відновлення кадрів зі смітника назад у датасет.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Вибірка: train, val або test"
    )
    parser.add_argument(
        "-t", "--type",
        choices=["positives", "negatives"],
        help="Тип даних: positives або negatives"
    )
    parser.add_argument(
        "--stage",
        choices=["kcenter", "uniform", "iou"],
        default="kcenter",
        help=(
            "Який етап відновлювати:\n"
            "  kcenter — k-Center дедуплікація (train)\n"
            "  uniform — рівномірне семплування (val/test)\n"
            "  iou     — IoU фільтр"
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показати що буде відновлено без реальних змін"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показати всі доступні папки trash"
    )

    args = parser.parse_args()

    if args.list:
        list_trash_folders()
    else:
        if not args.split or not args.type:
            parser.error("--split і --type обов'язкові якщо не використовується --list")
        restore_from_trash(
            split=args.split,
            embedding_type=args.type,
            stage=args.stage,
            dry_run=args.dry_run,
        )