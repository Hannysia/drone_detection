import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_STORED, ZipFile

from src.processing.manifests import CATEGORY_DIRS, read_manifest
from src.utils.io import check_extraction_source


def export_baseline(
    dataset_root: Path,
    manifest_path: Path,
    source_path: Path,
    output_path: Path,
) -> int:
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Archive already exists: {output_path}")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    check_extraction_source(dataset_root, source)
    rows = read_manifest(manifest_path)

    if any(row["excluded_at"] for row in rows):
        raise ValueError("Expected a baseline without exclusion decisions")

    files = [
        (manifest_path, "manifest.csv"),
        (dataset_root / "source.json", "source.json"),
    ]

    for row in rows:
        relative = Path(row["split"]) / CATEGORY_DIRS[row["category"]] / row["filename"]
        files.append((dataset_root / relative, relative.as_posix()))

        if row["category"] == "positive":
            label = relative.with_suffix(".txt")
            files.append((dataset_root / label, label.as_posix()))

    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing export file: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None

    try:
        with NamedTemporaryFile(
            dir=output_path.parent,
            prefix=".dataset-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)

        with ZipFile(temporary_path, "w", compression=ZIP_STORED) as archive:
            for path, name in files:
                archive.write(path, name)

        os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return len(rows)


def main() -> None:
    from src.config import FRAME_BASELINE_DIR, MANIFESTS_DIR

    source_path = MANIFESTS_DIR / "sources" / "hf.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    version = source["annotation_version"]
    output = FRAME_BASELINE_DIR.parent / "exports" / f"drone_dataset_{version}.zip"

    count = export_baseline(
        dataset_root=FRAME_BASELINE_DIR / version,
        manifest_path=MANIFESTS_DIR / "baseline" / version / "main.csv",
        source_path=source_path,
        output_path=output,
    )
    print(f"Exported {count} frames to {output}")


if __name__ == "__main__":
    main()