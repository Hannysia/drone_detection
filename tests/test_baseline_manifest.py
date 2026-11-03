import csv
import json

import pytest

from src.processing import baseline_manifest
from src.processing.baseline_manifest import save_baseline_manifest


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"

    for split in ("train", "val", "test"):
        for category in ("RAW_POSITIVES", "RAW_NEGATIVES"):
            (root / split / category).mkdir(parents=True)

    positive = root / "train" / "RAW_POSITIVES" / "demo_f0003.jpg"
    positive.write_bytes(b"positive")
    positive.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n")

    negative = root / "val" / "RAW_NEGATIVES" / "other_f0005.jpg"
    negative.write_bytes(b"negative")

    return root, positive, negative


def test_manifest_preserves_files_and_reports_counts(dataset, tmp_path):
    root, _, _ = dataset
    output = tmp_path / "manifests" / "main.csv"

    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    counts = save_baseline_manifest(root, output)

    assert counts["train", "positive"] == 1
    assert counts["val", "negative"] == 1
    assert counts["test", "positive"] == 0
    assert sum(counts.values()) == 2

    with output.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["filename", "split", "category"]
        assert list(reader) == [
            {
                "filename": "demo_f0003.jpg",
                "split": "train",
                "category": "positive",
            },
            {
                "filename": "other_f0005.jpg",
                "split": "val",
                "category": "negative",
            },
        ]

    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    assert after == before


def test_existing_manifest_is_not_overwritten(dataset, tmp_path):
    root, _, _ = dataset
    output = tmp_path / "main.csv"
    output.write_text("previous")

    with pytest.raises(FileExistsError):
        save_baseline_manifest(root, output)

    assert output.read_text() == "previous"


def test_output_inside_dataset_is_rejected(dataset):
    root, _, _ = dataset

    with pytest.raises(ValueError, match="outside"):
        save_baseline_manifest(root, root / "main.csv")


@pytest.mark.parametrize(
    ("problem", "error", "message"),
    [
        ("missing_label", FileNotFoundError, "Missing label"),
        ("empty_label", ValueError, "Empty positive label"),
        ("negative_label", ValueError, "Unexpected label"),
        ("orphan_label", FileNotFoundError, "Missing image"),
        ("duplicate_image", ValueError, "Duplicate filename"),
        ("missing_directory", FileNotFoundError, "Missing directory"),
    ],
)
def test_invalid_dataset_does_not_create_manifest(
    dataset, tmp_path, problem, error, message,
):
    root, positive, negative = dataset
    output = tmp_path / "main.csv"

    if problem == "missing_label":
        positive.with_suffix(".txt").unlink()
    elif problem == "empty_label":
        positive.with_suffix(".txt").write_text(" \n")
    elif problem == "negative_label":
        negative.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2")
    elif problem == "orphan_label":
        (positive.parent / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2")
    elif problem == "duplicate_image":
        (root / "train" / "RAW_NEGATIVES" / positive.name).write_bytes(
            b"duplicate"
        )
    elif problem == "missing_directory":
        (root / "test" / "RAW_NEGATIVES").rmdir()

    with pytest.raises(error, match=message):
        save_baseline_manifest(root, output)

    assert not output.exists()


def test_empty_dataset_is_rejected(dataset, tmp_path):
    root, positive, negative = dataset
    positive.with_suffix(".txt").unlink()
    positive.unlink()
    negative.unlink()
    output = tmp_path / "main.csv"

    with pytest.raises(ValueError, match="No JPG"):
        save_baseline_manifest(root, output)

    assert not output.exists()


def test_main_uses_pinned_workspace_and_preserves_old_baseline(
    dataset, tmp_path, monkeypatch, capsys,
):
    root, _, _ = dataset
    version = "v1"
    revision = "a" * 40
    frames = tmp_path / "frames"
    workspace = frames / version
    workspace.parent.mkdir(parents=True)
    root.rename(workspace)

    manifests = tmp_path / "manifests"
    source = manifests / "sources" / "hf.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({
        "repo_id": "test/dataset",
        "repo_type": "dataset",
        "annotation_version": version,
        "revision": revision,
    }))

    (workspace / "source.json").write_text(source.read_text())

    old = manifests / "baseline" / "main.csv"
    old.parent.mkdir()
    old.write_text("old baseline")

    monkeypatch.setattr(baseline_manifest, "FRAME_BASELINE_DIR", frames)
    monkeypatch.setattr(
        baseline_manifest,
        "EXTRACTED_DATA_DIR",
        tmp_path / "historical",
    )
    monkeypatch.setattr(baseline_manifest, "MANIFESTS_DIR", manifests)
    monkeypatch.setattr(baseline_manifest, "REPO_ID", "test/dataset")

    baseline_manifest.main()

    output = manifests / "baseline" / version  / "main.csv"

    assert output.is_file()
    assert old.read_text() == "old baseline"

    printed = capsys.readouterr().out
    assert "train,1,0,1" in printed
    assert "val,0,1,1" in printed
    assert "Saved 2 frames" in printed