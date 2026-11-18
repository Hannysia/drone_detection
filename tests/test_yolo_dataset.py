import json

import pytest
import yaml

from src.processing.manifests import CATEGORY_DIRS, write_manifest
from src.processing.yolo_dataset import cleanup_yolo_dataset, prepare_yolo_dataset


@pytest.fixture
def yolo_bundle(tmp_path):
    root = tmp_path / "curated"
    root.mkdir()
    source = {
        "repo_id": "test/dataset",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "annotation_version": "v1",
    }
    source_path = tmp_path / "hf.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    (root / "source.json").write_bytes(source_path.read_bytes())

    rows = [
        {"filename": "tr_pos.jpg", "split": "train", "category": "positive", "excluded_at": ""},
        {"filename": "tr_neg.jpg", "split": "train", "category": "negative", "excluded_at": ""},
        {"filename": "tr_excl.jpg", "split": "train", "category": "positive", "excluded_at": "temporal"},
        {"filename": "va_pos.jpg", "split": "val", "category": "positive", "excluded_at": ""},
        {"filename": "va_neg.jpg", "split": "val", "category": "negative", "excluded_at": ""},
        {"filename": "te_pos.jpg", "split": "test", "category": "positive", "excluded_at": ""},
        {"filename": "te_neg.jpg", "split": "test", "category": "negative", "excluded_at": ""},
    ]

    for row in rows:
        folder = root / row["split"] / CATEGORY_DIRS[row["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        image = folder / row["filename"]
        image.write_bytes(b"jpg")
        if row["category"] == "positive":
            image.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    manifest_path = tmp_path / "main.csv"
    write_manifest(rows, manifest_path)
    return root, manifest_path, source_path


def test_prepare_excludes_train_negatives_when_disabled_and_keeps_val_test(yolo_bundle, tmp_path):
    root, manifest_path, source_path = yolo_bundle
    working = tmp_path / "working"

    yaml_path = prepare_yolo_dataset(
        root,
        working,
        "positives_only",
        False,
        manifest_path=manifest_path,
        source_path=source_path,
    )

    output = working / "yolo_dataset_positives_only"
    assert sorted(p.name for p in (output / "images" / "train").iterdir()) == ["tr_pos.jpg"]
    assert sorted(p.name for p in (output / "labels" / "train").iterdir()) == ["tr_pos.txt"]
    assert sorted(p.name for p in (output / "images" / "val").iterdir()) == ["va_neg.jpg", "va_pos.jpg"]
    assert sorted(p.name for p in (output / "labels" / "val").iterdir()) == ["va_pos.txt"]
    assert sorted(p.name for p in (output / "images" / "test").iterdir()) == ["te_neg.jpg", "te_pos.jpg"]
    assert sorted(p.name for p in (output / "labels" / "test").iterdir()) == ["te_pos.txt"]

    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert parsed == {
        "path": str(output.resolve()),
        "names": {0: "drone"},
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }


def test_prepare_includes_train_negatives_when_enabled(yolo_bundle, tmp_path):
    root, manifest_path, source_path = yolo_bundle
    working = tmp_path / "working"

    prepare_yolo_dataset(
        root,
        working,
        "with_negatives",
        True,
        manifest_path=manifest_path,
        source_path=source_path,
    )

    output = working / "yolo_dataset_with_negatives"
    assert sorted(p.name for p in (output / "images" / "train").iterdir()) == ["tr_neg.jpg", "tr_pos.jpg"]
    assert sorted(p.name for p in (output / "labels" / "train").iterdir()) == ["tr_pos.txt"]


def test_prepare_validates_source_json(yolo_bundle, tmp_path):
    root, manifest_path, source_path = yolo_bundle
    working = tmp_path / "working"

    (root / "source.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source mismatch"):
        prepare_yolo_dataset(
            root,
            working,
            "run",
            True,
            manifest_path=manifest_path,
            source_path=source_path,
        )

    (root / "source.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing extraction source"):
        prepare_yolo_dataset(
            root,
            working,
            "run",
            True,
            manifest_path=manifest_path,
            source_path=source_path,
        )


def test_prepare_overwrite_and_reject_existing(yolo_bundle, tmp_path):
    root, manifest_path, source_path = yolo_bundle
    working = tmp_path / "working"

    yaml_path = prepare_yolo_dataset(
        root,
        working,
        "run",
        False,
        manifest_path=manifest_path,
        source_path=source_path,
    )

    with pytest.raises(FileExistsError):
        prepare_yolo_dataset(
            root,
            working,
            "run",
            True,
            manifest_path=manifest_path,
            source_path=source_path,
            overwrite=False,
        )

    prepare_yolo_dataset(
        root,
        working,
        "run",
        True,
        manifest_path=manifest_path,
        source_path=source_path,
        overwrite=True,
    )
    output = working / "yolo_dataset_run"
    assert yaml_path.is_file()
    assert sorted(p.name for p in (output / "images" / "train").iterdir()) == ["tr_neg.jpg", "tr_pos.jpg"]


def test_prepare_missing_file_raises_and_leaves_no_partial_output(yolo_bundle, tmp_path):
    root, manifest_path, source_path = yolo_bundle
    working = tmp_path / "working"
    (root / "train" / "RAW_POSITIVES" / "tr_pos.txt").unlink()

    with pytest.raises(FileNotFoundError, match="Missing selected file"):
        prepare_yolo_dataset(
            root,
            working,
            "broken",
            True,
            manifest_path=manifest_path,
            source_path=source_path,
        )

    assert not (working / "yolo_dataset_broken").exists()
    assert not (working / "dataset_broken.yaml").exists()


@pytest.mark.parametrize("bad_name", ["", " ", "../escape", "a/b", "a\\b", "..", "."])
def test_invalid_run_name_rejected(yolo_bundle, tmp_path, bad_name):
    root, manifest_path, source_path = yolo_bundle
    with pytest.raises(ValueError, match="Invalid run name"):
        prepare_yolo_dataset(
            root,
            tmp_path / "working",
            bad_name,
            True,
            manifest_path=manifest_path,
            source_path=source_path,
        )
    with pytest.raises(ValueError, match="Invalid run name"):
        cleanup_yolo_dataset(tmp_path / "working", bad_name)


def test_cleanup_yolo_dataset_removes_artifacts(yolo_bundle, tmp_path):
    root, manifest_path, source_path = yolo_bundle
    working = tmp_path / "working"

    yaml_path = prepare_yolo_dataset(
        root,
        working,
        "temp_run",
        True,
        manifest_path=manifest_path,
        source_path=source_path,
    )
    output = working / "yolo_dataset_temp_run"
    (output / "labels" / "train.cache").write_bytes(b"cache")

    cleanup_yolo_dataset(working, "temp_run")

    assert not output.exists()
    assert not yaml_path.exists()