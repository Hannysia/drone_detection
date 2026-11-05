import json
from zipfile import ZipFile

import pytest

from src.processing import dataset_export
from src.processing.manifests import write_manifest


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "frames"
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
        {"filename": "a.jpg", "split": "train", "category": "positive"},
        {"filename": "b.jpg", "split": "val", "category": "negative"},
        {"filename": "c.jpg", "split": "test", "category": "positive"},
    ]
    for row in rows:
        folder = root / row["split"] / dataset_export.CATEGORY_DIRS[row["category"]]
        folder.mkdir(parents=True)
        image = folder / row["filename"]
        image.write_bytes(b"test image")
        if row["category"] == "positive":
            image.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1\n")
    manifest = tmp_path / "main.csv"
    write_manifest(rows, manifest, include_excluded=False)
    return root, manifest, source_path, tmp_path / "export.zip"


def test_exports_exact_manifest_files_and_preserves_inputs(bundle):
    root, manifest, source, output = bundle
    (root / "unlisted.jpg").write_bytes(b"extra")
    (root / ".state").mkdir()
    (root / ".state" / "processed_videos.log").write_text("demo")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    assert dataset_export.export_baseline(*bundle) == 3

    with ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "manifest.csv", "source.json",
            "train/RAW_POSITIVES/a.jpg", "train/RAW_POSITIVES/a.txt",
            "val/RAW_NEGATIVES/b.jpg",
            "test/RAW_POSITIVES/c.jpg", "test/RAW_POSITIVES/c.txt",
        }
        assert archive.read("manifest.csv") == manifest.read_bytes()
        assert archive.read("source.json") == source.read_bytes()
        assert archive.testzip() is None
        for name in archive.namelist():
            if name not in {"manifest.csv", "source.json"}:
                assert archive.read(name) == (root / name).read_bytes()
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("missing", [
    "train/RAW_POSITIVES/a.jpg",
    "train/RAW_POSITIVES/a.txt",
    "source.json",
])
def test_missing_input_fails_without_archive(bundle, missing):
    (bundle[0] / missing).unlink()
    with pytest.raises(FileNotFoundError):
        dataset_export.export_baseline(*bundle)
    assert not bundle[3].exists()


def test_source_mismatch_fails(bundle):
    (bundle[0] / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="source mismatch"):
        dataset_export.export_baseline(*bundle)
    assert not bundle[3].exists()


def test_existing_archive_is_not_overwritten(bundle):
    bundle[3].write_bytes(b"existing archive")
    with pytest.raises(FileExistsError):
        dataset_export.export_baseline(*bundle)
    assert bundle[3].read_bytes() == b"existing archive"


def test_rejects_exclusion_decisions(bundle):
    bundle[1].write_text(
        "filename,split,category,excluded_at\n"
        "a.jpg,train,positive,embeddings\n"
    )
    with pytest.raises(ValueError, match="without exclusion"):
        dataset_export.export_baseline(*bundle)
    assert not bundle[3].exists()


def test_interrupted_export_cleans_up_and_can_be_retried(bundle, monkeypatch):
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("disk full")

        patch.setattr(dataset_export.ZipFile, "write", fail)
        with pytest.raises(OSError, match="disk full"):
            dataset_export.export_baseline(*bundle)

    assert not bundle[3].exists()
    assert not list(bundle[3].parent.glob(".dataset-*.tmp"))
    assert dataset_export.export_baseline(*bundle) == 3