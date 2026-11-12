import copy
import json
import sys

import pytest

from src.processing import temporal_manifest
from src.processing.embedding_manifest import (
    file_sha256,
    save_embedding_selection,
)
from src.processing.manifests import read_manifest, write_manifest


def frame(
    number,
    split="val",
    category="positive",
    excluded_at="",
):
    return {
        "filename": f"wing_rgb_sky_001_f{number:04d}.jpg",
        "split": split,
        "category": category,
        "excluded_at": excluded_at,
    }


def sample_rows():
    return [
        frame(1, "train"),
        frame(2, "train", excluded_at="embeddings"),
        *[frame(number) for number in range(1, 7)],
        frame(28),
        *[
            frame(number, category="negative")
            for number in range(7, 12)
        ],
        frame(15, excluded_at="iou"),
        frame(1, "test"),
        frame(2, "test"),
        frame(20, "test", "negative"),
        frame(57, "test", "negative"),
    ]


def test_selection_preserves_train_exclusions_and_row_identity():
    rows = sample_rows()
    original = copy.deepcopy(rows)

    result = temporal_manifest.build_temporal_manifest(rows, 5)

    assert rows == original
    assert len(result) == len(rows)

    for before, after in zip(rows, result):
        for key in ("filename", "split", "category"):
            assert after[key] == before[key]

        if before["split"] == "train" or before["excluded_at"]:
            assert before == after

    retained = {
        (row["split"], row["filename"])
        for row in result
        if not row["excluded_at"]
    }
    expected = [
        frame(1, "train"),
        frame(3),
        frame(6),
        frame(28),
        frame(9, category="negative"),
        frame(1, "test"),
        frame(20, "test", "negative"),
        frame(57, "test", "negative"),
    ]

    assert retained == {
        (row["split"], row["filename"])
        for row in expected
    }


def test_repeat_and_input_order_produce_identical_files(tmp_path):
    rows = sample_rows()
    first = temporal_manifest.build_temporal_manifest(rows, 5)
    second = temporal_manifest.build_temporal_manifest(rows[::-1], 5)

    save_embedding_selection(
        first,
        {"block_size": 5},
        tmp_path / "first",
    )
    save_embedding_selection(
        second,
        {"block_size": 5},
        tmp_path / "second",
    )

    for filename in ("main.csv", "run.json"):
        assert (
            (tmp_path / "first" / filename).read_bytes()
            == (tmp_path / "second" / filename).read_bytes()
        )

    assert any(
        row["excluded_at"] == "temporal"
        for row in read_manifest(tmp_path / "first/main.csv")
    )


@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    import src.config as config

    source = {
        "annotation_version": "v1",
        "repo_id": "test/dataset",
        "revision": "a" * 40,
    }
    manifests_dir = tmp_path / "manifests"
    (manifests_dir / "sources").mkdir(parents=True)
    (manifests_dir / "sources/hf.json").write_text(json.dumps(source))

    parent = manifests_dir / "embeddings/v1/within_video"
    write_manifest(sample_rows(), parent / "main.csv")
    (parent / "run.json").write_text(json.dumps({
        "source": source,
        "manifest_sha256": file_sha256(parent / "main.csv"),
    }))

    monkeypatch.setattr(config, "MANIFESTS_DIR", manifests_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["temporal_manifest", "--block-size", "5"],
    )

    return manifests_dir


def test_cli_replay_reproduces_manifest(
    cli_setup,
    tmp_path,
    monkeypatch,
):
    temporal_manifest.main()
    original = cli_setup / "temporal/v1"
    metadata = json.loads((original / "run.json").read_text())

    assert metadata["block_size"] == 5
    assert metadata["manifest_sha256"] == file_sha256(
        original / "main.csv"
    )

    output = tmp_path / "reproduced"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal_manifest",
            "--from-run", str(original / "run.json"),
            "--output-dir", str(output),
        ],
    )

    temporal_manifest.main()

    assert (
        (original / "main.csv").read_bytes()
        == (output / "main.csv").read_bytes()
    )


@pytest.mark.parametrize(
    "field",
    [
        "source",
        "input_manifest_sha256",
        "input_run_sha256",
        "manifest_sha256",
        "block_size",
    ],
)
def test_replay_mismatch_publishes_no_result(
    cli_setup,
    tmp_path,
    monkeypatch,
    field,
):
    temporal_manifest.main()
    original = cli_setup / "temporal/v1"
    original_csv = (original / "main.csv").read_bytes()

    metadata = json.loads((original / "run.json").read_text())
    metadata[field] = 2 if field == "block_size" else "0" * 64

    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(metadata))
    output = tmp_path / "reproduced"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "temporal_manifest",
            "--from-run", str(changed),
            "--output-dir", str(output),
        ],
    )

    with pytest.raises(ValueError):
        temporal_manifest.main()

    assert not output.exists()
    assert (original / "main.csv").read_bytes() == original_csv


def test_cli_refuses_overwrite(cli_setup):
    temporal_manifest.main()
    path = cli_setup / "temporal/v1/main.csv"
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        temporal_manifest.main()

    assert path.read_bytes() == original


def test_changed_input_manifest_fails(cli_setup):
    path = cli_setup / "embeddings/v1/within_video/main.csv"
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        temporal_manifest.main()

    assert not (cli_setup / "temporal/v1").exists()