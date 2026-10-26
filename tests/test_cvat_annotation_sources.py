
import pytest

from src.integration import cvat


@pytest.mark.parametrize(
    ("source", "directory", "suffix"),
    [
        ("sam3", "cvat_batches", "_mot.zip"),
        ("labels", "labels", ".zip"),
    ],
)
def test_pull_batch_selects_annotation_source(
    monkeypatch,
    tmp_path,
    source,
    directory,
    suffix,
):
    files = [
        "val/clip_a.mp4",
        "val/clip_b.mov",
        "cvat_batches/val/batch_001/clip_a_mot.zip",
        "cvat_batches/val/batch_001/clip_b_mot.zip",
        "labels/val/batch_001/clip_a.zip",
        "labels/val/batch_001/clip_b.zip",
    ]

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: files,
    )
    monkeypatch.setattr(
        cvat,
        "hf_hub_download",
        lambda **kwargs: kwargs["filename"],
    )

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        lambda task_name: [],
    )

    commands = []
    task_ids = iter([101, 102])

    def fake_run(command):
        commands.append(command)

        if " create " in command:
            return f"Task ID: {next(task_ids)}"

        return ""

    monkeypatch.setattr(cvat, "run_cmd", fake_run)

    cvat.pull_batch_to_cvat(
        split="val",
        batch_name="batch_001",
        annotation_source=source,
        version="v1",
    )

    creates = [
        command
        for command in commands
        if " create " in command
    ]

    uploads = [
        command
        for command in commands
        if " upload " in command
    ]

    assert len(creates) == 2
    assert len(uploads) == 2

    for video in ("clip_a", "clip_b"):
        archive = (
            f"{directory}/val/batch_001/"
            f"{video}{suffix}"
        )

        assert any(
            archive in command
            for command in uploads
        )


def test_unknown_annotation_source_fails():
    with pytest.raises(
        ValueError,
        match="Unknown annotation source",
    ):
        cvat.pull_batch_to_cvat(
            "val",
            "batch_001",
            "unknown",
            "v1",
        )
