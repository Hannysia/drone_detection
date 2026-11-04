import csv
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.config import SPLITS


CATEGORY_DIRS = {
    "positive": "RAW_POSITIVES",
    "negative": "RAW_NEGATIVES",
}

FIELDNAMES = ("filename", "split", "category")
SELECTION_FIELDNAMES = FIELDNAMES + ("excluded_at",)
EXCLUSION_STAGES = {"", "iou", "embeddings"}


def _validate_fields(fields: Iterable[str]) -> None:
    fields = tuple(fields)

    if len(fields) != len(set(fields)) or set(fields) not in (
        set(FIELDNAMES),
        set(SELECTION_FIELDNAMES),
    ):
        raise ValueError(f"Invalid manifest columns: {fields}")


def _normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized = []
    seen = set()

    for number, row in enumerate(rows, start=2):
        _validate_fields(row)

        if any(not isinstance(value, str) for value in row.values()):
            raise ValueError(f"Invalid value in manifest row {number}")

        filename = row["filename"]

        if (
            len(filename) <= 4
            or not filename.endswith(".jpg")
            or filename != filename.strip()
            or any(
                char in '<>:"/\\|?*' or ord(char) < 32
                for char in filename
            )
        ):
            raise ValueError(
                f"Invalid filename in manifest row {number}: {filename!r}"
            )

        if row["split"] not in SPLITS or row["category"] not in CATEGORY_DIRS:
            raise ValueError(
                f"Invalid split or category in manifest row {number}"
            )

        excluded_at = row.get("excluded_at", "")

        if excluded_at not in EXCLUSION_STAGES:
            raise ValueError(
                f"Invalid excluded_at in manifest row {number}"
            )

        key = (row["split"], filename)

        if key in seen:
            raise ValueError(f"Duplicate manifest key: {key}")

        seen.add(key)

        normalized.append({
            "filename": filename,
            "split": row["split"],
            "category": row["category"],
            "excluded_at": excluded_at,
        })

    if not normalized:
        raise ValueError("Manifest has no rows")

    return normalized


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _validate_fields(reader.fieldnames or ())
        return _normalize_rows(list(reader))


def write_manifest(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    include_excluded: bool = True,
) -> int:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Manifest already exists: {output_path}")

    rows = _normalize_rows(rows)

    if not include_excluded and any(row["excluded_at"] for row in rows):
        raise ValueError("Cannot discard exclusion decisions")

    categories = tuple(CATEGORY_DIRS)

    rows.sort(key=lambda row: (
        SPLITS.index(row["split"]),
        categories.index(row["category"]),
        row["filename"],
    ))

    fields = SELECTION_FIELDNAMES if include_excluded else FIELDNAMES
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)

            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in fields}
                for row in rows
            )

            stream.flush()
            os.fsync(stream.fileno())

        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return len(rows)


def collect_frame_rows(dataset_root: Path) -> list[dict[str, str]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    rows = []

    for split in SPLITS:
        filenames_in_split = set()

        for category, folder_name in CATEGORY_DIRS.items():
            folder = dataset_root / split / folder_name

            if not folder.is_dir():
                raise FileNotFoundError(f"Missing directory: {folder}")

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