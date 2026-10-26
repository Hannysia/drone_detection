
from pathlib import Path

import pytest
import requests

from src.integration import cvat


def test_existing_tasks_are_found_across_pages(monkeypatch):
    pages = {
        1: {
            "results": [{"id": 100, "name": "other_train"}],
            "next": "http://localhost:8080/api/tasks?page=2",
        },
        2: {
            "results": [{"id": 695, "name": "video_train"}],
            "next": None,
        },
    }

    requested_pages = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    def fake_get(url, **kwargs):
        assert kwargs["headers"]["Accept"] == "application/vnd.cvat+json"
        assert kwargs["params"]["name"] == "video_train"

        page = kwargs["params"]["page"]
        requested_pages.append(page)

        return FakeResponse(pages[page])

    monkeypatch.setattr(cvat.requests, "get", fake_get)

    result = cvat.find_existing_cvat_task_ids("video_train")

    assert result == [695]
    assert requested_pages == [1, 2]


def test_pull_skips_existing_and_imports_missing(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    files = [
        "cvat_batches/train/batch_001/already_mot.zip",
        "cvat_batches/train/batch_001/fresh_mot.zip",
        "train/already.mp4",
        "train/fresh.mp4",
    ]

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: files,
    )

    checked_names = []

    def fake_find(task_name):
        checked_names.append(task_name)

        if task_name == "already_train":
            return [695]

        return []

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        fake_find,
    )

    downloads = []

    def fake_download(**kwargs):
        downloads.append(kwargs["filename"])
        return str(tmp_path / Path(kwargs["filename"]).name)

    monkeypatch.setattr(
        cvat,
        "hf_hub_download",
        fake_download,
    )

    commands = []

    def fake_run_cmd(command):
        commands.append(command)

        if " create " in command:
            return "Task ID: 700"

        return "Upload successful"

    monkeypatch.setattr(cvat, "run_cmd", fake_run_cmd)

    cvat.pull_batch_to_cvat(
        "train",
        "batch_001",
        annotation_source="sam3",
        version="v1",
    )

    assert checked_names == [
        "already_train",
        "fresh_train",
    ]

    assert downloads == [
        "train/fresh.mp4",
        "cvat_batches/train/batch_001/fresh_mot.zip",
    ]

    assert len(commands) == 2
    assert " create " in commands[0]
    assert "fresh_train" in commands[0]
    assert " upload 700 " in commands[1]


def test_pull_stops_if_cvat_lookup_fails(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: [
            "cvat_batches/train/batch_001/video_mot.zip",
            "train/video.mp4",
        ],
    )

    def fake_find(task_name):
        raise requests.ConnectionError("CVAT unavailable")

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        fake_find,
    )

    def forbidden_command(command):
        pytest.fail("CVAT task must not be created")

    monkeypatch.setattr(
        cvat,
        "run_cmd",
        forbidden_command,
    )

    with pytest.raises(requests.ConnectionError):
        cvat.pull_batch_to_cvat(
            "train",
            "batch_001",
            annotation_source="sam3",
            version="v1",
        )



def test_existing_task_warns_about_unverified_import(
    monkeypatch,
    tmp_path,
    caplog,
):
    import logging

    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    monkeypatch.setattr(
        cvat.api,
        "list_repo_files",
        lambda **kwargs: [
            "labels/test/batch_004/wing_thermal_sky_540.zip",
        ],
    )

    monkeypatch.setattr(
        cvat,
        "find_existing_cvat_task_ids",
        lambda task_name: [695],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Existing CVAT task must not be modified")

    monkeypatch.setattr(cvat, "hf_hub_download", forbidden)
    monkeypatch.setattr(cvat, "run_cmd", forbidden)

    with caplog.at_level(logging.WARNING, logger=cvat.log.name):
        cvat.pull_batch_to_cvat(
            split="test",
            batch_name="batch_004",
            annotation_source="labels",
            version="v1",
        )

    assert "Annotation import status is NOT VERIFIED" in caplog.text
    assert (
        f"http://{cvat.CVAT_HOST}:{cvat.CVAT_PORT}/tasks/695"
        in caplog.text
    )
    assert "Existing annotations were not modified" in caplog.text
