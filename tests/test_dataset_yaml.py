import yaml

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