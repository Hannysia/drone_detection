
"""Create an inventory of the historical training dataset."""

import csv
from pathlib import Path

from src.config import (
    EXTRACTED_DATA_DIR,
    HISTORICAL_MANIFEST,
    SPLITS,
)


CATEGORY_DIRS = {
    "positive": "RAW_POSITIVES",
    "negative": "RAW_NEGATIVES",
}

FIELDNAMES = ("filename", "split", "category")


def collect_historical_rows(
    dataset_root: Path,
) -> list[dict[str, str]]:
    """Collect frame metadata without modifying the dataset."""

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}"
        )

    rows = []

    for split in SPLITS:
        filenames_in_split = set()

        for category, folder_name in CATEGORY_DIRS.items():
            folder = dataset_root / split / folder_name

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Missing directory: {folder}"
                )

            for image in sorted(folder.glob("*.jpg")):
                if image.name in filenames_in_split:
                    raise ValueError(
                        f"Duplicate filename in {split}: {image.name}"
                    )

                filenames_in_split.add(image.name)

                if (
                    category == "positive"
                    and not image.with_suffix(".txt").is_file()
                ):
                    raise FileNotFoundError(
                        f"Missing label for positive frame: {image}"
                    )

                rows.append({
                    "filename": image.name,
                    "split": split,
                    "category": category,
                })

    if not rows:
        raise ValueError("No JPG images found")

    return rows


def save_historical_manifest(
    dataset_root: Path,
    output_path: Path,
) -> int:
    """Save the historical dataset manifest as CSV."""

    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()

    if output_path.is_relative_to(dataset_root):
        raise ValueError(
            "Manifest output must be outside the dataset directory"
        )

    rows = collect_historical_rows(dataset_root)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


if __name__ == "__main__":
    print(f"Dataset: {EXTRACTED_DATA_DIR}")
    print(f"Manifest: {HISTORICAL_MANIFEST}")

    count = save_historical_manifest(
        dataset_root=EXTRACTED_DATA_DIR,
        output_path=HISTORICAL_MANIFEST,
    )

    print(f"Saved {count} frames")
