import yaml
import pytest
from src.processing.dataset import prepare_yolo_dataset


def test_prepare_yolo_dataset_writes_readable_yaml(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"

    path = prepare_yolo_dataset(source, work, "example", use_negatives=False)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["path"] == str((work / "yolo_dataset_example").resolve())
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["test"] == "images/test"


def test_existing_run_is_preserved(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    work = tmp_path / "work"

    yaml_path = prepare_yolo_dataset(source, work, "example", use_negatives=True)
    previous = work / "yolo_dataset_example" / "images" / "train" / "old.jpg"
    previous.write_bytes(b"previous frame")
    yaml_path.write_bytes(b"previous configuration")

    with pytest.raises(FileExistsError):
        prepare_yolo_dataset(source, work, "example", use_negatives=False)

    assert previous.read_bytes() == b"previous frame"
    assert yaml_path.read_bytes() == b"previous configuration"