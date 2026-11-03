import json

import pytest

from src.utils.io import check_extraction_source


@pytest.fixture
def source():
    return {
        "repo_id": "test/dataset",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "annotation_version": "v1",
    }


def test_initialize_and_resume_preserve_source_and_data(tmp_path, source):
    root = tmp_path / "v1"
    check_extraction_source(root, source, initialize=True)
    source_path = root / "source.json"
    original = source_path.read_bytes()
    image = root / "keep.jpg"
    image.write_bytes(b"frame")

    check_extraction_source(root, source, initialize=True)
    check_extraction_source(root, source)

    assert json.loads(original) == source
    assert source_path.read_bytes() == original
    assert image.read_bytes() == b"frame"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "other/dataset"),
        ("repo_type", "model"),
        ("revision", "b" * 40),
        ("annotation_version", "v2"),
    ],
)
@pytest.mark.parametrize("initialize", [False, True])
def test_changed_source_is_rejected(tmp_path, source, field, value, initialize):
    check_extraction_source(tmp_path, source, initialize=True)
    original = (tmp_path / "source.json").read_bytes()
    changed = {**source, field: value}

    with pytest.raises(ValueError, match="source mismatch"):
        check_extraction_source(tmp_path, changed, initialize=initialize)

    assert (tmp_path / "source.json").read_bytes() == original


def test_read_only_check_does_not_initialize_workspace(tmp_path, source):
    root = tmp_path / "v1"

    with pytest.raises(FileNotFoundError, match="Missing extraction source"):
        check_extraction_source(root, source)

    assert not root.exists()


def test_legacy_non_empty_workspace_is_not_adopted(tmp_path, source):
    existing = tmp_path / "old.jpg"
    existing.write_bytes(b"old frame")

    with pytest.raises(ValueError, match="non-empty workspace"):
        check_extraction_source(tmp_path, source, initialize=True)

    assert existing.read_bytes() == b"old frame"
    assert not (tmp_path / "source.json").exists()


def test_interrupted_write_does_not_publish_partial_source(
    tmp_path, source, monkeypatch,
):
    def fail_dump(data, file, **kwargs):
        file.write("{")
        raise OSError("injected write failure")

    with monkeypatch.context() as changes:
        changes.setattr(json, "dump", fail_dump)

        with pytest.raises(OSError, match="injected write failure"):
            check_extraction_source(tmp_path, source, initialize=True)

    assert list(tmp_path.iterdir()) == []

    check_extraction_source(tmp_path, source, initialize=True)
    assert json.loads((tmp_path / "source.json").read_text()) == source