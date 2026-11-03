import csv
import json
import re
from collections import Counter
from pathlib import Path

from src.config import (
    EXTRACTED_DATA_DIR,
    FRAME_BASELINE_DIR,
    MANIFESTS_DIR,
    REPO_ID,
    SPLITS,
)
from src.processing.historical_manifest import (
    CATEGORY_DIRS,
    FIELDNAMES,
    collect_historical_rows,
)
from src.utils.io import check_extraction_source, get_extraction_paths

def save_baseline_manifest(
    dataset_root: Path,
    output_path: Path,
) -> Counter[tuple[str, str]]:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()

    if output_path.is_relative_to(dataset_root):
        raise ValueError("Manifest output must be outside the dataset directory")

    if output_path.exists():
        raise FileExistsError(f"Manifest already exists: {output_path}")

    rows = collect_historical_rows(dataset_root)

    for split in SPLITS:
        for category, folder_name in CATEGORY_DIRS.items():
            folder = dataset_root / split / folder_name

            for label in sorted(folder.glob("*.txt")):
                if not label.with_suffix(".jpg").is_file():
                    raise FileNotFoundError(f"Missing image for label: {label}")

                if category == "negative":
                    raise ValueError(f"Unexpected label for negative frame: {label}")

                if not label.read_text(encoding="utf-8").strip():
                    raise ValueError(f"Empty positive label: {label}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return Counter((row["split"], row["category"]) for row in rows)


def main() -> None:
    source_path = MANIFESTS_DIR / "sources" / "hf.json"

    with source_path.open(encoding="utf-8") as file:
        source = json.load(file)

    if source["repo_id"] != REPO_ID or source["repo_type"] != "dataset":
        raise ValueError("Source does not match the configured HF dataset")

    version = source["annotation_version"]
    revision = source["revision"]

    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("Invalid annotation version")

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected an immutable HF commit SHA")

    dataset_root, _, _ = get_extraction_paths(
        baseline_dir=FRAME_BASELINE_DIR / version,
        historical_dir=EXTRACTED_DATA_DIR,
    )
    check_extraction_source(dataset_root, source)
    output_path = MANIFESTS_DIR / "baseline" / version / "main.csv"

    print(f"Dataset: {dataset_root}")
    print(f"Manifest: {output_path}")

    counts = save_baseline_manifest(dataset_root, output_path)

    print("split,positive,negative,total")

    for split in SPLITS:
        positive = counts[split, "positive"]
        negative = counts[split, "negative"]
        print(f"{split},{positive},{negative},{positive + negative}")

    print(f"Saved {sum(counts.values())} frames")


if __name__ == "__main__":
    main()