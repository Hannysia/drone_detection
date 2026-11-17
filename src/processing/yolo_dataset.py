from collections import Counter
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import yaml

from src.config import MANIFESTS_DIR, SPLITS
from src.processing.manifests import CATEGORY_DIRS, read_manifest

log = logging.getLogger(__name__)


def safe_symlink(src: Path, dst: Path) -> None:
    """Creates a symbolic link, safely unlinking any existing file or link at destination first."""
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except OSError:
            return
    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass


def prepare_yolo_dataset(
    IMG_DATASET_PATH: Path,
    working_dir: Path,
    run_name: str,
    use_negatives: bool,
    *,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
) -> Path:
    root = IMG_DATASET_PATH.resolve()
    output = working_dir / f"yolo_dataset_{run_name}"
    yaml_path = working_dir / f"dataset_{run_name}.yaml"

    if output.exists() or output.is_symlink() or yaml_path.exists() or yaml_path.is_symlink():
        raise FileExistsError(f"Dataset output already exists for {run_name}")

    rows = read_manifest(manifest_path)
    working_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()

    with TemporaryDirectory(dir=working_dir) as tmp:
        staged = Path(tmp) / "dataset"
        for split in SPLITS:
            for kind in ("images", "labels"):
                (staged / kind / split).mkdir(parents=True)

        for row in rows:
            split, category = row["split"], row["category"]
            if row["excluded_at"]:
                continue
            if split == "train" and category == "negative" and not use_negatives:
                continue

            image = root / split / CATEGORY_DIRS[category] / row["filename"]
            files = [("images", image)]
            if category == "positive":
                files.append(("labels", image.with_suffix(".txt")))

            for kind, source in files:
                if not source.is_file():
                    raise FileNotFoundError(f"Missing selected file: {source}")
                destination = staged / kind / split / source.name
                try:
                    destination.symlink_to(source)
                except OSError as exc:
                    if getattr(exc, "winerror", None) != 1314:
                        raise
                    shutil.copy2(source, destination)

            counts[split, category] += 1

        config = {"path": str(output.resolve()), "names": {0: "drone"}}
        config.update({split: f"images/{split}" for split in SPLITS})

        staged_yaml = Path(tmp) / "dataset.yaml"
        staged_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
        staged.rename(output)
        try:
            os.link(staged_yaml, yaml_path)
        except OSError:
            shutil.rmtree(output)
            raise

    for split in SPLITS:
        log.info(
            "%s: %d positives, %d negatives",
            split,
            counts[split, "positive"],
            counts[split, "negative"],
        )

    return yaml_path