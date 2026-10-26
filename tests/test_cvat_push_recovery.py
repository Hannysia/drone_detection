
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.integration import cvat


ARGS = ("695", "test", "batch_004", "v1")
REPO_ZIP = "labels_curated/v1/test/batch_004/video.zip"
REPO_REVIEW = "labels_curated/v1/test/batch_004/video.review.json"


@pytest.fixture
def push_env(monkeypatch, tmp_path):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    class FakeResponse:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_get(url, **kwargs):
        if url.endswith("/data/meta"):
            return FakeResponse({
                "size": 77,
                "start_frame": 0,
                "stop_frame": 76,
                "frame_filter": "",
                "included_frames": None,
                "deleted_frames": [1, 3, 5],
            })

        assert url.endswith("/api/tasks/695")
        return FakeResponse({"name": "video_test"})

    monkeypatch.setattr(cvat.requests, "get", fake_get)

    class FakeApi:
        def __init__(self):
            self.published = set()
            self.commits = []
            self.mode = "success"

        def dataset_info(self, **kwargs):
            return SimpleNamespace(sha="a" * 40)

        def file_exists(self, **kwargs):
            return kwargs["filename"] in self.published

        def create_commit(self, **kwargs):
            self.commits.append(kwargs)

            if self.mode == "offline":
                raise requests.ConnectionError("Connection lost")

            self.published.update(
                operation.path_in_repo
                for operation in kwargs["operations"]
            )

            if self.mode == "lost_response":
                raise requests.ConnectionError("Response lost")

            return SimpleNamespace(oid="b" * 40)

    fake_api = FakeApi()
    monkeypatch.setattr(cvat, "api", fake_api)

    dump_calls = []

    def successful_dump(command):
        dump_calls.append(command)

        match = re.search(
            r'dump 695 "([^"]+)" --format "MOT 1\.1"',
            command,
        )

        assert match is not None
        archive_path = Path(match.group(1))

        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "gt/gt.txt",
                "1,1,10,20,30,40,1,1,1\n",
            )

        return ""

    monkeypatch.setattr(cvat, "run_cmd", successful_dump)

    export_dir = (
        tmp_path / "exports" / "v1" / "test" / "batch_004"
    )

    return SimpleNamespace(
        api=fake_api,
        dump_calls=dump_calls,
        successful_dump=successful_dump,
        archive_path=export_dir / "video.zip",
        review_path=export_dir / "video.review.json",
    )


def test_interrupted_dump_preserves_existing_local_files_and_retries(
    push_env, monkeypatch
):
    push_env.archive_path.parent.mkdir(parents=True)

    with zipfile.ZipFile(push_env.archive_path, "w") as archive:
        archive.writestr("old.txt", "previous export")

    push_env.review_path.write_text(
        '{"previous": true}',
        encoding="utf-8",
    )

    original_zip = push_env.archive_path.read_bytes()
    original_review = push_env.review_path.read_bytes()
    attempts = 0

    def interrupted_dump(command):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            match = re.search(r'dump 695 "([^"]+)"', command)
            assert match is not None

            Path(match.group(1)).write_bytes(b"incomplete ZIP")
            raise SystemExit(1)

        return push_env.successful_dump(command)

    monkeypatch.setattr(cvat, "run_cmd", interrupted_dump)

    with pytest.raises(SystemExit) as error:
        cvat.push_batch_to_hf(*ARGS)

    assert error.value.code == 1
    assert push_env.api.commits == []
    assert push_env.archive_path.read_bytes() == original_zip
    assert push_env.review_path.read_bytes() == original_review

    cvat.push_batch_to_hf(*ARGS)

    assert attempts == 2
    assert len(push_env.api.commits) == 1
    assert zipfile.is_zipfile(push_env.archive_path)

    review = json.loads(
        push_env.review_path.read_text(encoding="utf-8")
    )

    assert review["deleted_frames"] == [1, 3, 5]


def test_failed_hf_upload_can_retry(push_env):
    push_env.api.mode = "offline"

    with pytest.raises(requests.ConnectionError):
        cvat.push_batch_to_hf(*ARGS)

    assert zipfile.is_zipfile(push_env.archive_path)
    assert push_env.review_path.exists()
    assert push_env.api.published == set()

    push_env.api.mode = "success"
    cvat.push_batch_to_hf(*ARGS)

    assert len(push_env.api.commits) == 2
    assert len(push_env.dump_calls) == 2
    assert push_env.api.published == {REPO_ZIP, REPO_REVIEW}


def test_completed_hf_export_is_skipped(push_env):
    push_env.api.published.update((REPO_ZIP, REPO_REVIEW))

    cvat.push_batch_to_hf(*ARGS)

    assert push_env.api.commits == []
    assert push_env.dump_calls == []


@pytest.mark.parametrize(
    "published_file",
    [REPO_ZIP, REPO_REVIEW],
)
def test_partial_hf_export_is_rejected(
    push_env,
    published_file,
):
    push_env.api.published.add(published_file)

    with pytest.raises(
        RuntimeError,
        match="Incomplete HF export",
    ):
        cvat.push_batch_to_hf(*ARGS)

    assert push_env.api.commits == []
    assert push_env.dump_calls == []


def test_lost_commit_response_does_not_publish_twice(push_env):
    push_env.api.mode = "lost_response"

    with pytest.raises(requests.ConnectionError):
        cvat.push_batch_to_hf(*ARGS)

    assert push_env.api.published == {REPO_ZIP, REPO_REVIEW}

    push_env.api.mode = "success"
    cvat.push_batch_to_hf(*ARGS)

    assert len(push_env.api.commits) == 1
    assert len(push_env.dump_calls) == 1
