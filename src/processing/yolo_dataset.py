from collections import Counter
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import yaml

from src.config import MANIFESTS_DIR, SPLITS
from src.processing.manifests import CATEGORY_DIRS, read_manifest
from src.utils.io import check_extraction_source

log = logging.getLogger(__name__)


def _validate_run_name(run_name: str) -> str:
    if (
        not run_name
        or run_name != run_name.strip()
        or run_name in {".", ".."}
        or "/" in run_name
        or "\\" in run_name
    ):
        raise ValueError(f"Invalid run name: {run_name!r}")
    return run_name
def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
def cleanup_yolo_dataset(working_dir: Path, run_name: str) -> None:
    validated_name = _validate_run_name(run_name)
    _remove_existing_path(working_dir / f"yolo_dataset_{validated_name}")
    _remove_existing_path(working_dir / f"dataset_{validated_name}.yaml")


def prepare_yolo_dataset(
    IMG_DATASET_PATH: Path,
    working_dir: Path,
    run_name: str,
    use_negatives: bool,
    *,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    validated_name = _validate_run_name(run_name)
    root = IMG_DATASET_PATH.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if source_path is not None:
        expected_source = json.loads(source_path.read_text(encoding="utf-8"))
        check_extraction_source(root, expected_source)
    output = working_dir / f"yolo_dataset_{validated_name}"
    yaml_path = working_dir / f"dataset_{validated_name}.yaml"
    existing = (
        output.exists()
        or output.is_symlink()
        or yaml_path.exists()
        or yaml_path.is_symlink()
    )
    if existing and not overwrite:
        raise FileExistsError(f"Dataset output already exists for {validated_name}")


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

        if not sum(counts.values()):
            raise ValueError(f"No frames selected from manifest: {manifest_path}")

        config = {"path": str(output.resolve()), "names": {0: "drone"}}
        config.update({split: f"images/{split}" for split in SPLITS})

        staged_yaml = Path(tmp) / "dataset.yaml"
        staged_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
        if overwrite:
            cleanup_yolo_dataset(working_dir, validated_name)

        staged.rename(output)
        try:
            os.link(staged_yaml, yaml_path)
        except OSError:
            shutil.rmtree(output, ignore_errors=True)
            raise

    log.info(
        "Prepared YOLO dataset %s (use_negatives=%s) from %s",
        validated_name,
        use_negatives,
        root,
    )
    for split in SPLITS:
        log.info(
            "%s: %d positives, %d negatives",
            split,
            counts[split, "positive"],
            counts[split, "negative"],
        )

    return yaml_path