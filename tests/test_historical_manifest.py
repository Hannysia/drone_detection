
import csv

import pytest

from src.processing.historical_manifest import (
    collect_historical_rows,
    save_historical_manifest,
)


def make_dataset(tmp_path):
    root = tmp_path / "dataset"

    for split in ("train", "val", "test"):
        for category in ("RAW_POSITIVES", "RAW_NEGATIVES"):
            (root / split / category).mkdir(parents=True)

    positive = (
        root / "train" / "RAW_POSITIVES" / "video_f0001.jpg"
    )
    positive.write_bytes(b"positive jpeg placeholder")
    positive.with_suffix(".txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"
    )

    negative = (
        root / "test" / "RAW_NEGATIVES"
        / "no_drone_f0001.jpg"
    )
    negative.write_bytes(b"negative jpeg placeholder")

    return root, positive, negative


def test_manifest_contains_three_fields_and_preserves_files(tmp_path):
    root, _, _ = make_dataset(tmp_path)
    output = tmp_path / "manifests" / "historical" / "final.csv"

    before = {
        p.relative_to(root): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }

    assert save_historical_manifest(root, output) == 2

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [
        {
            "filename": "video_f0001.jpg",
            "split": "train",
            "category": "positive",
        },
        {
            "filename": "no_drone_f0001.jpg",
            "split": "test",
            "category": "negative",
        },
    ]

    after = {
        p.relative_to(root): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }

    assert before == after


def test_missing_positive_label_is_reported(tmp_path):
    root, positive, _ = make_dataset(tmp_path)
    positive.with_suffix(".txt").unlink()

    with pytest.raises(FileNotFoundError, match="Missing label"):
        collect_historical_rows(root)


def test_duplicate_filename_is_reported(tmp_path):
    root, positive, _ = make_dataset(tmp_path)

    duplicate = (
        root / "train" / "RAW_NEGATIVES" / positive.name
    )
    duplicate.write_bytes(b"duplicate")

    with pytest.raises(ValueError, match="Duplicate filename"):
        collect_historical_rows(root)


def test_existing_manifest_is_not_overwritten(tmp_path):
    root, _, _ = make_dataset(tmp_path)
    output = tmp_path / "final.csv"
    output.write_text("previous manifest", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_historical_manifest(root, output)

    assert output.read_text(encoding="utf-8") == "previous manifest"
