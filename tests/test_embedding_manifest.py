import copy
import json

import numpy as np
import pytest

from src.processing.embedding_manifest import (
    build_embedding_manifest,
    save_embedding_selection,
)
from src.processing.manifests import read_manifest


def sample():
    names = [
        "wing_rgb_sky_001_f0010.jpg",
        "wing_rgb_sky_001_f0002.jpg",
        "wing_rgb_sky_001_f0001.jpg",
    ]
    rows = [
        {"filename": name, "split": "train", "category": "positive"}
        for name in names
    ]
    rows += [
        {
            "filename": "wing_rgb_sky_001_f0003.jpg",
            "split": "train",
            "category": "positive",
            "excluded_at": "iou",
        },
        {
            "filename": "wing_rgb_sky_002_f0001.jpg",
            "split": "train",
            "category": "positive",
        },
        {
            "filename": "no_drone_rgb_sky_003_f0001.jpg",
            "split": "train",
            "category": "negative",
        },
        {
            "filename": "wing_rgb_sky_004_f0001.jpg",
            "split": "val",
            "category": "positive",
        },
        {
            "filename": "wing_rgb_sky_005_f0001.jpg",
            "split": "test",
            "category": "positive",
        },
    ]
    embeddings = {
        "positive": {
            name: np.array([1.0, 0.0])
            for name in names + [rows[4]["filename"]]
        },
        "negative": {
            rows[5]["filename"]: np.array([1.0, 0.0]),
        },
    }
    return rows, embeddings, {"positive": 0.09, "negative": 0.02}


def test_preserves_exclusions_splits_and_video_boundaries():
    rows, embeddings, thresholds = sample()
    original = copy.deepcopy(rows)

    result = build_embedding_manifest(rows, embeddings, thresholds)

    assert rows == original
    assert [row["excluded_at"] for row in result] == [
        "embeddings", "embeddings", "", "iou", "", "", "", "",
    ]


def test_repeat_is_identical_with_different_input_order(tmp_path):
    rows, embeddings, thresholds = sample()

    first = build_embedding_manifest(rows, embeddings, thresholds)
    second = build_embedding_manifest(
        list(reversed(rows)), embeddings, thresholds,
    )

    save_embedding_selection(
        first, {"thresholds": thresholds}, tmp_path / "first",
    )
    save_embedding_selection(
        second, {"thresholds": thresholds}, tmp_path / "second",
    )

    for filename in ("main.csv", "run.json"):
        assert (
            (tmp_path / "first" / filename).read_bytes()
            == (tmp_path / "second" / filename).read_bytes()
        )

    assert len(read_manifest(tmp_path / "first/main.csv")) == len(rows)


def test_missing_embedding_fails():
    rows, embeddings, thresholds = sample()
    embeddings["positive"].pop(rows[0]["filename"])

    with pytest.raises(ValueError, match="filenames mismatch"):
        build_embedding_manifest(rows, embeddings, thresholds)


def test_invalid_vector_fails():
    rows, embeddings, thresholds = sample()
    embeddings["positive"][rows[0]["filename"]] = np.array([2.0, 0.0])

    with pytest.raises(ValueError, match="normalized embeddings"):
        build_embedding_manifest(rows, embeddings, thresholds)


def test_existing_result_is_preserved(tmp_path):
    rows, embeddings, thresholds = sample()
    result = build_embedding_manifest(rows, embeddings, thresholds)
    output = tmp_path / "selection"

    save_embedding_selection(result, {}, output)
    original = (output / "main.csv").read_bytes()

    with pytest.raises(FileExistsError):
        save_embedding_selection(result, {}, output)

    assert (output / "main.csv").read_bytes() == original


def test_metadata_failure_leaves_no_partial_result(tmp_path):
    rows, embeddings, thresholds = sample()
    result = build_embedding_manifest(rows, embeddings, thresholds)

    with pytest.raises(TypeError):
        save_embedding_selection(
            result,
            {"invalid": object()},
            tmp_path / "selection",
        )

    assert not list(tmp_path.iterdir())


def test_main_writes_selection_with_provenance(tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    import src.config as config
    from src.processing import embedding_manifest
    from src.processing.manifests import write_manifest

    rows, embeddings, thresholds = sample()
    source = {
        "annotation_version": "v1",
        "repo_id": "test/dataset",
        "revision": "a" * 40,
    }

    manifests_dir = tmp_path / "manifests"
    (manifests_dir / "sources").mkdir(parents=True)
    (manifests_dir / "sources/hf.json").write_text(json.dumps(source))
    write_manifest(rows, manifests_dir / "baseline/v1/main.csv")

    embedding_dir = tmp_path / "embeddings/v1"
    embedding_dir.mkdir(parents=True)
    paths = []

    for category in embeddings:
        path = embedding_dir / f"train_{category}s_embeddings.pt"
        path.write_bytes(b"test fixture")
        paths.append(path)

    monkeypatch.setattr(config, "MANIFESTS_DIR", manifests_dir)
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", embedding_dir.parent)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            load=lambda path, **kwargs: embeddings[
                "negative" if "negatives" in path.name else "positive"
            ],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "src.processing.embeddings",
        SimpleNamespace(validate_embeddings=lambda *args, **kwargs: paths),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embedding_manifest",
            "--positive-theta", "0.09",
            "--negative-theta", "0.02",
        ],
    )

    embedding_manifest.main()

    output = manifests_dir / "embeddings/v1/within_video"
    record = json.loads((output / "run.json").read_text())

    assert len(read_manifest(output / "main.csv")) == len(rows)
    assert record["thresholds"] == thresholds
    assert record["source"] == source
    assert record["manifest_sha256"] == embedding_manifest.file_sha256(
        output / "main.csv"
    )
    assert "Manifest:" in capsys.readouterr().out