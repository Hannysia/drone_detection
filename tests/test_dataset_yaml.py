from pathlib import Path

import pytest
import yaml

from src.processing.dataset import prepare_yolo_dataset


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "frames"
    manifest = tmp_path / "main.csv"
    lines = ["filename,split,category,excluded_at"]

    for split in ("train", "val", "test"):
        for category, folder in (
            ("positive", "RAW_POSITIVES"),
            ("negative", "RAW_NEGATIVES"),
        ):
            directory = root / split / folder
            directory.mkdir(parents=True)

            for name in ("keep", "excluded", "unlisted"):
                image = directory / f"{category}_{name}.jpg"
                image.touch()
                if category == "positive":
                    image.with_suffix(".txt").write_text("0 0.5 0.5 0.1 0.1")
                if name != "unlisted":
                    stage = "temporal" if name == "excluded" else ""
                    lines.append(f"{image.name},{split},{category},{stage}")

    manifest.write_text("\n".join(lines), encoding="utf-8")
    return root, tmp_path / "work", manifest


@pytest.mark.parametrize("negatives", [False, True])
def test_exact_files_by_split(inputs, negatives):
    root, work, manifest = inputs
    path = prepare_yolo_dataset(root, work, "example", negatives, manifest_path=manifest)
    config = yaml.safe_load(path.read_text())
    dataset = Path(config["path"])

    assert dataset == (work / "yolo_dataset_example").resolve()

    for split in ("train", "val", "test"):
        assert config[split] == f"images/{split}"
        expected = {"positive_keep.jpg"}
        if split != "train" or negatives:
            expected.add("negative_keep.jpg")

        assert {p.name for p in (dataset / config[split]).iterdir()} == expected
        assert {p.name for p in (dataset / "labels" / split).iterdir()} == {
            "positive_keep.txt"
        }

    with pytest.raises(FileExistsError):
        prepare_yolo_dataset(root, work, "example", negatives, manifest_path=manifest)
    assert path.is_file()


@pytest.mark.parametrize("suffix", [".jpg", ".txt"])
def test_missing_selected_file_allows_retry(inputs, suffix):
    root, work, manifest = inputs
    missing = root / "test" / "RAW_POSITIVES" / f"positive_keep{suffix}"
    missing.unlink()

    with pytest.raises(FileNotFoundError) as error:
        prepare_yolo_dataset(root, work, "example", True, manifest_path=manifest)

    assert str(missing) in str(error.value)
    missing.touch()
    assert prepare_yolo_dataset(root, work, "example", True, manifest_path=manifest).is_file()



def test_windows_symlink_privilege_fallback(inputs, monkeypatch):
    root, work, manifest = inputs

    def denied(self, target):
        error = OSError("Symlink privilege missing")
        error.winerror = 1314
        raise error

    monkeypatch.setattr(Path, "symlink_to", denied)
    prepare_yolo_dataset(root, work, "example", True, manifest_path=manifest)

    source = root / "train" / "RAW_POSITIVES" / "positive_keep.txt"
    copied = work / "yolo_dataset_example" / "labels" / "train" / source.name
    assert not copied.is_symlink()
    assert copied.read_bytes() == source.read_bytes()