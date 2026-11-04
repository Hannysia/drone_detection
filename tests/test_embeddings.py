import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from src.processing import embeddings
from src.processing.manifests import write_manifest
import json
from types import SimpleNamespace

class TinyModel(torch.nn.Module):
    def forward(self, images):
        assert not self.training
        return images.mean(dim=(2, 3)) + 5.0


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "dataset"
    rows = [
        ("a.jpg", "train", "positive", ""),
        ("b.jpg", "train", "negative", ""),
        ("excluded.jpg", "train", "positive", "embeddings"),
        ("validation.jpg", "val", "positive", ""),
        ("testing.jpg", "test", "positive", ""),
    ]
    manifest = tmp_path / "manifest.csv"

    write_manifest([
        dict(zip(
            ("filename", "split", "category", "excluded_at"),
            row,
        ))
        for row in rows
    ], manifest)

    for category, name in [
        ("positive", "a.jpg"),
        ("negative", "b.jpg"),
    ]:
        folder = root / "train" / embeddings.CATEGORY_DIRS[category]
        folder.mkdir(parents=True)
        Image.new("RGB", (128, 128), "red").save(folder / name)

    (root / "train/RAW_POSITIVES/a.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n",
    )
    Image.new("RGB", (128, 128)).save(
        root / "train/RAW_POSITIVES/extra.jpg",
    )

    return root, manifest, tmp_path / "output.pt"


@pytest.mark.parametrize("category, expected", [
    ("positive", "a.jpg"),
    ("negative", "b.jpg"),
])
def test_generation_uses_only_active_train_rows(bundle, category, expected):
    root, manifest, output = bundle

    embeddings.generate_and_save_embeddings(
        root,
        manifest,
        TinyModel(),
        category,
        output,
        num_workers=0,
    )

    result = torch.load(output, weights_only=False)

    assert set(result) == {expected}

    vector = result[expected]

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (6,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)

    if category == "negative":
        np.testing.assert_array_equal(vector[3:], 0)
    else:
        ratio = np.linalg.norm(vector[:3]) / np.linalg.norm(vector[3:])
        assert ratio == pytest.approx(0.01 / 0.99)


@pytest.mark.parametrize("filename", ["a.jpg", "a.txt"])
def test_missing_positive_file_fails(bundle, filename):
    (bundle[0] / "train/RAW_POSITIVES" / filename).unlink()

    dataset = embeddings.DroneEmbeddingDataset(
        bundle[0], bundle[1], "positive",
    )

    with pytest.raises(FileNotFoundError):
        dataset[0]


@pytest.mark.parametrize("content", [
    "",
    "0 0.5 0.5",
    "0 0.5 0.5 0 0.1",
    "0 nan 0.5 0.1 0.1",
    "0 0.5 0.5 1.1 0.1",
    "0 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1",
])
def test_invalid_positive_annotation_fails(bundle, content):
    (bundle[0] / "train/RAW_POSITIVES/a.txt").write_text(content)

    dataset = embeddings.DroneEmbeddingDataset(
        bundle[0], bundle[1], "positive",
    )

    with pytest.raises(ValueError):
        dataset[0]


def test_corrupt_image_is_not_replaced_with_black_image(bundle):
    (bundle[0] / "train/RAW_POSITIVES/a.jpg").write_bytes(b"broken")

    with pytest.raises(UnidentifiedImageError):
        embeddings.generate_and_save_embeddings(
            bundle[0],
            bundle[1],
            TinyModel(),
            "positive",
            bundle[2],
            num_workers=0,
        )

    assert not bundle[2].exists()


def test_existing_result_is_not_overwritten(bundle):
    bundle[2].write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        embeddings.generate_and_save_embeddings(
            bundle[0],
            bundle[1],
            TinyModel(),
            "positive",
            bundle[2],
            num_workers=0,
        )

    assert bundle[2].read_bytes() == b"existing"


def test_empty_selection_fails(bundle):
    bundle[1].write_text(
        "filename,split,category\nonly.jpg,val,negative\n",
    )

    with pytest.raises(ValueError, match="No active train"):
        embeddings.DroneEmbeddingDataset(
            bundle[0], bundle[1], "positive",
        )


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_invalid_model_output_is_not_saved(bundle, value):
    class InvalidModel(torch.nn.Module):
        def forward(self, images):
            return torch.full((len(images), 3), value)

    with pytest.raises(ValueError, match="invalid embeddings"):
        embeddings.generate_and_save_embeddings(
            bundle[0],
            bundle[1],
            InvalidModel(),
            "positive",
            bundle[2],
            num_workers=0,
        )

    assert not bundle[2].exists()


def test_failed_save_cleans_temporary_file_and_allows_retry(
    bundle,
    monkeypatch,
):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(embeddings.torch, "save", fail)

        with pytest.raises(OSError, match="disk full"):
            embeddings.generate_and_save_embeddings(
                bundle[0],
                bundle[1],
                TinyModel(),
                "positive",
                bundle[2],
                num_workers=0,
            )

    assert not bundle[2].exists()
    assert not list(bundle[2].parent.glob(".embeddings-*.tmp"))

    embeddings.generate_and_save_embeddings(
        bundle[0],
        bundle[1],
        TinyModel(),
        "positive",
        bundle[2],
        num_workers=0,
    )


@pytest.fixture
def cli_bundle(bundle, tmp_path, monkeypatch):
    root, manifest, _ = bundle
    source = {
        "repo_id": "test/dataset",
        "repo_type": "dataset",
        "revision": "a" * 40,
        "annotation_version": "v1",
    }

    manifests = tmp_path / "manifests"
    (manifests / "sources").mkdir(parents=True)
    (manifests / "baseline/v1").mkdir(parents=True)
    (manifests / "sources/hf.json").write_text(json.dumps(source))
    (root / "source.json").write_text(json.dumps(source))
    (root / "manifest.csv").write_bytes(manifest.read_bytes())
    (manifests / "baseline/v1/main.csv").write_bytes(
        manifest.read_bytes(),
    )

    output = tmp_path / "embeddings"
    monkeypatch.setattr(embeddings, "MANIFESTS_DIR", manifests)
    monkeypatch.setattr(embeddings, "EMBEDDINGS_DIR", output)

    return root, output


def test_cli_check_only_does_not_load_model_or_write_outputs(
    cli_bundle,
    monkeypatch,
):
    def fail(*args, **kwargs):
        pytest.fail("Model should not be loaded")

    monkeypatch.setattr(embeddings.torch.hub, "load", fail)

    embeddings.main([
        "--dataset-root", str(cli_bundle[0]),
        "--category", "positive",
        "--check-only",
    ])

    assert not cli_bundle[1].exists()


@pytest.mark.parametrize("change", ["source", "manifest"])
def test_cli_rejects_wrong_input_before_model_loading(
    cli_bundle,
    monkeypatch,
    change,
):
    root, output = cli_bundle

    if change == "source":
        (root / "source.json").write_text("{}")
    else:
        (root / "manifest.csv").write_text(
            "filename,split,category\nother.jpg,train,positive\n",
        )

    with pytest.raises(ValueError):
        embeddings.main([
            "--dataset-root", str(root),
            "--category", "positive",
            "--check-only",
        ])

    assert not output.exists()


def test_cli_runs_categories_separately_and_records_actual_commit(
    cli_bundle,
    monkeypatch,
):
    root, output = cli_bundle
    calls = []

    monkeypatch.setattr(
        embeddings.torch.cuda, "is_available", lambda: True,
    )
    monkeypatch.setattr(
        embeddings.torch.cuda, "empty_cache", lambda: None,
    )
    monkeypatch.setattr(
        embeddings.torch.hub,
        "load",
        lambda *args, **kwargs: SimpleNamespace(
            to=lambda device: TinyModel(),
        ),
    )
    monkeypatch.setattr(
        embeddings.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="b" * 40 + "\n",
        ),
    )

    def generate(**kwargs):
        calls.append(kwargs["category"])
        kwargs["output_path"].write_bytes(b"test output")

    monkeypatch.setattr(
        embeddings, "generate_and_save_embeddings", generate,
    )

    for category in ("positive", "negative"):
        embeddings.main([
            "--dataset-root", str(root),
            "--category", category,
        ])

        assert (
            output / "v1" / f"train_{category}s_embeddings.pt"
        ).exists()

    assert calls == ["positive", "negative"]

    metadata = json.loads((output / "v1/run.json").read_text())

    assert metadata["code_revision"] == "b" * 40
    assert metadata["input_counts"] == {
        "positive": 1,
        "negative": 1,
    }

    with pytest.raises(FileExistsError):
        embeddings.main([
            "--dataset-root", str(root),
            "--category", "positive",
        ])

    assert calls == ["positive", "negative"]


def test_run_metadata_cannot_be_rebound(tmp_path):
    output = tmp_path / "output"

    embeddings._bind_run(output, {"model": "first"})
    embeddings._bind_run(output, {"model": "first"})

    with pytest.raises(ValueError, match="different run"):
        embeddings._bind_run(output, {"model": "second"})

    assert json.loads((output / "run.json").read_text()) == {
        "model": "first",
    }


def test_failed_metadata_write_cleans_up_and_allows_retry(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output"

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(embeddings.os, "link", fail)

        with pytest.raises(OSError, match="disk full"):
            embeddings._bind_run(output, {"model": "first"})

    assert list(output.iterdir()) == []

    embeddings._bind_run(output, {"model": "first"})


def test_dataset_selection_does_not_access_frame_directory(bundle, tmp_path):
    dataset = embeddings.DroneEmbeddingDataset(
        tmp_path / "unavailable_frames",
        bundle[1],
        "positive",
    )

    assert len(dataset) == 1

    with pytest.raises(FileNotFoundError):
        dataset[0]