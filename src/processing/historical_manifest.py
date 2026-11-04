from pathlib import Path

from src.config import EXTRACTED_DATA_DIR, HISTORICAL_MANIFEST
from src.processing.manifests import (
    CATEGORY_DIRS,
    FIELDNAMES,
    collect_frame_rows as collect_historical_rows,
    write_manifest,
)


def save_historical_manifest(
    dataset_root: Path,
    output_path: Path,
) -> int:
    dataset_root = dataset_root.resolve()
    output_path = output_path.resolve()

    if output_path.is_relative_to(dataset_root):
        raise ValueError(
            "Manifest output must be outside the dataset directory"
        )

    rows = collect_historical_rows(dataset_root)

    return write_manifest(
        rows,
        output_path,
        include_excluded=False,
    )


if __name__ == "__main__":
    print(f"Dataset: {EXTRACTED_DATA_DIR}")
    print(f"Manifest: {HISTORICAL_MANIFEST}")

    count = save_historical_manifest(
        dataset_root=EXTRACTED_DATA_DIR,
        output_path=HISTORICAL_MANIFEST,
    )

    print(f"Saved {count} frames")