import importlib.util
from pathlib import Path

import dotenv
import pytest


@pytest.fixture
def local_config(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("LOCAL_BASE_DIR", raising=False)
    monkeypatch.delenv("EXTRACTED_DATA_PATH", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: False)

    path = Path(__file__).resolve().parents[1] / "src" / "config.py"
    spec = importlib.util.spec_from_file_location("isolated_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_constants_without_local_paths(local_config):
    assert local_config.SPLITS == ["train", "val", "test"]

    with pytest.raises(ValueError, match="LOCAL_BASE_DIR"):
        _ = local_config.FRAME_BASELINE_DIR


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_extracted_path(local_config, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("EXTRACTED_DATA_PATH", value)

    with pytest.raises(ValueError, match="EXTRACTED_DATA_PATH"):
        _ = local_config.EXTRACTED_DATA_DIR


def test_configured_paths(local_config, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("EXTRACTED_DATA_PATH", str(tmp_path / "frames"))

    assert local_config.FRAME_BASELINE_DIR == tmp_path / "frame_baseline"
    assert local_config.EXTRACTED_DATA_DIR == tmp_path / "frames"