import csv
import json
import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from src.config import MANIFESTS_DIR, SPLITS
from src.utils.io import check_extraction_source, parse_frame_name


CATEGORY_DIRS = {
    "positive": "RAW_POSITIVES",
    "negative": "RAW_NEGATIVES",
}

FIELDNAMES = ("filename", "split", "category")
SELECTION_FIELDNAMES = FIELDNAMES + ("excluded_at",)
EXCLUSION_STAGES = {"", "iou", "embeddings", "temporal"}

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


def _load_validated_split_rows(
    dataset_root: Path,
    *,
    split: str,
    manifest_path: Path,
    source_path: Path | None,
    allowed_splits: tuple[str, ...],
) -> tuple[Path, list[dict[str, str]]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split!r}")
    if split not in allowed_splits:
        raise ValueError(
            f"Split {split!r} is not allowed; expected one of {allowed_splits}"
        )

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if source_path is not None:
        expected_source = json.loads(Path(source_path).read_text(encoding="utf-8"))
        check_extraction_source(root, expected_source)

    return root, read_manifest(Path(manifest_path))


def select_manifest_frames(
    dataset_root: Path,
    *,
    split: str = "val",
    category: str | None = None,
    manifest_path: Path = MANIFESTS_DIR / "temporal" / "v1" / "main.csv",
    source_path: Path | None = None,
    allowed_splits: tuple[str, ...] = ("val", "test"),
) -> list[tuple[Path, str]]:
    """Returns active frame paths and categories for a split filtered by manifest."""
    if category is not None and category not in CATEGORY_DIRS:
        raise ValueError(f"Unknown category: {category!r}")

    root, rows = _load_validated_split_rows(
        dataset_root,
        split=split,
        manifest_path=manifest_path,
        source_path=source_path,
        allowed_splits=allowed_splits,
    )
    selected = []
    for row in rows:
        if row["split"] != split or row["excluded_at"]:
            continue
        if category is not None and row["category"] != category:
            continue
        image_path = root / split / CATEGORY_DIRS[row["category"]] / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing manifest image: {image_path}")
        selected.append((image_path, row["category"]))

    if not selected:
        raise ValueError(
            f"No active frames for split={split!r}, category={category!r} in {manifest_path}"
        )
    return selected


def load_baseline_video_sequences(
    dataset_root: Path,
    *,
    split: str = "val",
    manifest_path: Path = MANIFESTS_DIR / "baseline" / "v1" / "main.csv",
    source_path: Path | None = None,
    allowed_splits: tuple[str, ...] = ("val", "test"),
    require_positive_labels: bool = True,
) -> dict[str, list[dict[str, object]]]:
    """Loads chronologically ordered retained frames grouped by video_id with gap deltas."""
    root, rows = _load_validated_split_rows(
        dataset_root,
        split=split,
        manifest_path=manifest_path,
        source_path=source_path,
        allowed_splits=allowed_splits,
    )
    grouped: dict[str, list[tuple[int, dict[str, object]]]] = {}
    seen_frames: dict[str, set[int]] = {}

    for row in rows:
        if row["split"] != split or row["excluded_at"]:
            continue
        video_id, frame_num = parse_frame_name(row["filename"])
        seen = seen_frames.setdefault(video_id, set())
        if frame_num in seen:
            raise ValueError(f"Duplicate frame number {frame_num} for video {video_id}")
        seen.add(frame_num)

        image_path = root / split / CATEGORY_DIRS[row["category"]] / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing baseline frame: {image_path}")

        label_path: Path | None = None
        if row["category"] == "positive":
            candidate = image_path.with_suffix(".txt")
            if require_positive_labels and not candidate.is_file():
                raise FileNotFoundError(f"Missing label for positive baseline frame: {candidate}")
            if candidate.is_file():
                label_path = candidate

        grouped.setdefault(video_id, []).append((
            frame_num,
            {
                "video_id": video_id,
                "filename": row["filename"],
                "image_path": image_path,
                "label_path": label_path,
                "category": row["category"],
                "source_frame": frame_num,
            },
        ))

    if not grouped:
        raise ValueError(f"No active video frames for split={split!r} in {manifest_path}")

    sequences: dict[str, list[dict[str, object]]] = {}
    for video_id in sorted(grouped):
        items = sorted(grouped[video_id], key=lambda pair: pair[0])
        seq_frames: list[dict[str, object]] = []
        prev_source: int | None = None
        for mot_frame, (source_frame, item) in enumerate(items, start=1):
            dt_frames = 1 if prev_source is None else source_frame - prev_source
            prev_source = source_frame
            seq_frames.append({
                **item,
                "mot_frame": mot_frame,
                "dt_frames": dt_frames,
            })
        sequences[video_id] = seq_frames

    return sequences