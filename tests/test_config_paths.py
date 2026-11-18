import importlib.util
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import dotenv
import pytest


@pytest.fixture
def local_config(monkeypatch):
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("LOCAL_BASE_DIR", raising=False)
    monkeypatch.delenv("EXTRACTED_DATA_PATH", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
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
    assert local_config.WORKING_DIR == tmp_path / "working"
    assert local_config.OUTPUT_DIR == local_config.WORKING_DIR / "runs" / "detect"
    assert local_config.EXTRACTED_DATA_DIR == tmp_path / "frames"

    
@pytest.mark.parametrize("attached", [True, False])
def test_kaggle_selects_configured_dataset(monkeypatch, tmp_path, attached):
    owner = tmp_path / "datasets" / "hannusiaknysh"
    other = owner / "another-dataset"
    expected = owner / "uav-detection-dataset" / "extracted_dataset"
    other.mkdir(parents=True)
    if attached:
        expected.mkdir(parents=True)

    config_file = Path(__file__).resolve().parents[1] / "src" / "config.py"
    kaggle_owner = Path("/kaggle/input/datasets/hannusiaknysh")
    real_exists = Path.exists
    real_iterdir = Path.iterdir

    def exists(path):
        return True if path == kaggle_owner else real_exists(path)

    def iterdir(path):
        return owner.iterdir() if path == kaggle_owner else real_iterdir(path)

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.setitem(
        sys.modules, "kaggle_secrets", SimpleNamespace(UserSecretsClient=Mock())
    )
    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(Path, "iterdir", iterdir)

    config = runpy.run_path(str(config_file))

    assert config["EXTRACTED_DATA_DIR"] == kaggle_owner / expected.relative_to(owner)


def test_wandb_secret_defaults_and_kaggle_fallback(local_config, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    assert local_config.WANDB_PROJECT == "drone-detection"

    monkeypatch.setenv("WANDB_API_KEY", "  wb-key  ")
    monkeypatch.setenv("WANDB_PROJECT", " custom-proj ")
    config_file = Path(__file__).resolve().parents[1] / "src" / "config.py"
    local_loaded = runpy.run_path(str(config_file))
    assert local_loaded["WANDB_API_KEY"] == "wb-key"
    assert local_loaded["WANDB_PROJECT"] == "custom-proj"

    def get_secret(name):
        if name.startswith("WANDB"):
            raise RuntimeError("missing secret")
        return f"secret-{name}"

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "kaggle_secrets",
        SimpleNamespace(UserSecretsClient=Mock(return_value=SimpleNamespace(get_secret=get_secret))),
    )
    kaggle_loaded = runpy.run_path(str(config_file))
    assert kaggle_loaded["REPO_ID"] == "secret-REPO_ID"
    assert kaggle_loaded["HF_TOKEN"] == "secret-HF_TOKEN"
    assert kaggle_loaded["WANDB_API_KEY"] is None
    assert kaggle_loaded["WANDB_PROJECT"] == "drone-detection"


def test_colab_config_uses_userdata_and_content_paths(monkeypatch):
    config_file = Path(__file__).resolve().parents[1] / "src" / "config.py"
    secrets = {
        "REPO_ID": "user/repo",
        "HF_TOKEN": "hf_colab",
        "WANDB_API_KEY": "wb_colab",
        "KAGGLE_USERNAME": "anna",
        "KAGGLE_KEY": "k_secret",
    }

    def get_secret(name):
        if name not in secrets:
            raise LookupError(name)
        return secrets[name]

    fake_colab = SimpleNamespace(userdata=SimpleNamespace(get=get_secret))
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-test")
    monkeypatch.setenv("KAGGLE_USERNAME", "")
    monkeypatch.setenv("KAGGLE_KEY", "")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(colab=fake_colab))
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)

    colab_loaded = runpy.run_path(str(config_file))

    assert colab_loaded["IS_KAGGLE"] is False
    assert colab_loaded["IS_COLAB"] is True
    assert colab_loaded["WORKING_DIR"] == Path("/content")
    assert colab_loaded["OUTPUT_DIR"] == Path("/content/runs/detect")
    assert colab_loaded["REPO_ID"] == "user/repo"
    assert colab_loaded["HF_TOKEN"] == "hf_colab"
    assert colab_loaded["WANDB_API_KEY"] == "wb_colab"
    assert colab_loaded["WANDB_PROJECT"] == "drone-detection"
    assert colab_loaded["os"].environ["KAGGLE_USERNAME"] == "anna"
    assert colab_loaded["os"].environ["KAGGLE_KEY"] == "k_secret"