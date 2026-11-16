import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from src.processing.dataset import copy_split_videos, ensure_empty_split_output_dir


def test_ensure_empty_split_output_dir_accepts_missing_and_empty(tmp_path):
    missing = tmp_path / "missing_splits"
    ensure_empty_split_output_dir(missing)

    empty = tmp_path / "empty_splits"
    empty.mkdir()
    ensure_empty_split_output_dir(empty)


def test_ensure_empty_split_output_dir_rejects_non_empty_and_file(tmp_path):
    non_empty = tmp_path / "splits"
    (non_empty / "train").mkdir(parents=True)
    (non_empty / "train" / "old.mp4").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="not empty"):
        ensure_empty_split_output_dir(non_empty)

    file_path = tmp_path / "splits_file"
    file_path.write_text("not a dir", encoding="utf-8")

    with pytest.raises(FileExistsError, match="must be a directory"):
        ensure_empty_split_output_dir(file_path)


def test_ensure_empty_split_output_dir_rejects_symlink(tmp_path, monkeypatch):
    target = tmp_path / "split_link"
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == target)

    with pytest.raises(FileExistsError, match="must be a directory"):
        ensure_empty_split_output_dir(target)


def test_copy_split_videos_writes_splits_into_new_or_empty_dir(tmp_path):
    source = tmp_path / "renamed"
    source.mkdir()
    (source / "a.mp4").write_bytes(b"video-a")
    (source / "b.mp4").write_bytes(b"video-b")
    (source / "c.mp4").write_bytes(b"video-c")

    output = tmp_path / "output"
    splits = {
        "train": pd.DataFrame({"filename": ["a.mp4"]}),
        "val": pd.DataFrame({"filename": ["b.mp4"]}),
        "test": pd.DataFrame({"filename": ["c.mp4"]}),
    }

    copy_split_videos(source, output, splits)

    assert (output / "train" / "a.mp4").read_bytes() == b"video-a"
    assert (output / "val" / "b.mp4").read_bytes() == b"video-b"
    assert (output / "test" / "c.mp4").read_bytes() == b"video-c"


def test_copy_split_videos_leaves_no_partial_files_on_failure(tmp_path):
    source = tmp_path / "renamed"
    source.mkdir()
    (source / "a.mp4").write_bytes(b"video-a")

    output = tmp_path / "output"
    splits = {
        "train": pd.DataFrame({"filename": ["a.mp4"]}),
        "val": pd.DataFrame({"filename": ["missing.mp4"]}),
        "test": pd.DataFrame({"filename": []}),
    }

    with pytest.raises(FileNotFoundError):
        copy_split_videos(source, output, splits)

    assert not output.exists()


def test_split_script_exits_before_scanning_when_output_non_empty(tmp_path, monkeypatch):
    source = tmp_path / "renamed"
    output = tmp_path / "splits"
    source.mkdir()
    output.mkdir()
    (source / "wing_rgb_sky_001.mp4").write_bytes(b"src")
    (output / "stale.mp4").write_bytes(b"old")

    monkeypatch.setenv("RENAMED_DIR", str(source))
    monkeypatch.setenv("SPLIT_OUTPUT_DIR", str(output))

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "04_split_data.py"
    spec = importlib.util.spec_from_file_location("split_script_under_test", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "RENAMED_DIR", source)
    monkeypatch.setattr(module, "SPLIT_OUTPUT_DIR", output)
    monkeypatch.setattr(
        module,
        "get_frame_count",
        lambda *_: pytest.fail("should not scan videos when output is non-empty"),
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
    assert (output / "stale.mp4").read_bytes() == b"old"