
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import re
import pytest
import requests
from src.integration import cvat


def test_review_metadata_preserves_deleted_frames():
    meta = {
        "size": 77,
        "start_frame": 0,
        "stop_frame": 76,
        "frame_filter": "",
        "included_frames": None,
        "deleted_frames": [15, 1, 11, 3, 7, 5, 13, 9],
    }

    result = cvat.build_review_metadata(
        meta,
        "wing_thermal_sky_540",
        "695",
        "v2",
    )

    assert result["size"] == 77
    assert result["task_id"] == 695
    assert result["deleted_frames"] == [
        1, 3, 5, 7, 9, 11, 13, 15
    ]


def test_review_metadata_accepts_no_deletions():
    meta = {
        "size": 5,
        "start_frame": 0,
        "stop_frame": 4,
        "frame_filter": "",
        "included_frames": None,
        "deleted_frames": [],
    }

    result = cvat.build_review_metadata(
        meta,
        "clip",
        "20",
        "v2",
    )

    assert result["deleted_frames"] == []


def test_review_metadata_requires_deleted_frames():
    meta = {
        "size": 5,
        "start_frame": 0,
        "stop_frame": 4,
        "frame_filter": "",
        "included_frames": None,
    }

    with pytest.raises(
        ValueError,
        match="Missing CVAT metadata fields",
    ):
        cvat.build_review_metadata(
            meta,
            "clip",
            "20",
            "v2",
        )


def test_push_uploads_both_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(cvat, "LABELS_DIR", tmp_path)

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

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

        return FakeResponse({
            "name": "wing_thermal_sky_540_val"
        })

    monkeypatch.setattr(
        cvat.requests,
        "get",
        fake_get,
    )

    class FakeApi:
        def dataset_info(self, **kwargs):
            return SimpleNamespace(sha="a" * 40)

        def file_exists(self, **kwargs):
            return False

        def create_commit(self, **kwargs):
            self.commit_args = kwargs
            return SimpleNamespace(oid="b" * 40)

    fake_api = FakeApi()

    monkeypatch.setattr(cvat, "api", fake_api)

    
    def fake_run(command):
        match = re.search(
            r'dump \d+ "([^"]+)" --format "MOT 1\.1"',
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

    monkeypatch.setattr(cvat, "run_cmd", fake_run)

    cvat.push_batch_to_hf(
        "695",
        "val",
        "batch_001",
        "v2",
    )

    operations = fake_api.commit_args["operations"]

    assert len(operations) == 2
    assert fake_api.commit_args["parent_commit"] == "a" * 40

    assert [op.path_in_repo for op in operations] == [
        "labels_curated/v2/val/batch_001/wing_thermal_sky_540.zip",
        "labels_curated/v2/val/batch_001/wing_thermal_sky_540.review.json",
    ]

    review_path = (
        tmp_path
        / "exports"
        / "v2"
        / "val"
        / "batch_001"
        / "wing_thermal_sky_540.review.json"
    )

    saved_review = json.loads(
        review_path.read_text(encoding="utf-8")
    )

    assert saved_review["deleted_frames"] == [1, 3, 5]



def test_push_recovers_after_failed_upload(
    tmp_path,
    monkeypatch,
):
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

        return FakeResponse({
            "name": "wing_thermal_sky_540_test",
        })

    monkeypatch.setattr(cvat.requests, "get", fake_get)

    class FakeApi:
        def __init__(self):
            self.existing = set()
            self.commit_calls = 0

        def dataset_info(self, **kwargs):
            return SimpleNamespace(sha="a" * 40)

        def file_exists(self, **kwargs):
            return kwargs["filename"] in self.existing

        def create_commit(self, **kwargs):
            self.commit_calls += 1

            if self.commit_calls == 1:
                raise requests.ConnectionError("Connection lost")

            return SimpleNamespace(oid="b" * 40)

    fake_api = FakeApi()
    monkeypatch.setattr(cvat, "api", fake_api)

    export_calls = []

    def fake_run(command):
        match = re.search(
            r'dump \d+ "([^"]+)" --format "MOT 1\.1"',
            command,
        )

        assert match is not None

        archive_path = Path(match.group(1))
        export_calls.append(archive_path)

        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(
                "gt/gt.txt",
                "1,1,10,20,30,40,1,1,1\n",
            )

        return ""

    monkeypatch.setattr(cvat, "run_cmd", fake_run)

    args = ("695", "test", "batch_004", "v1")

    with pytest.raises(requests.ConnectionError):
        cvat.push_batch_to_hf(*args)

    export_dir = (
        tmp_path / "exports" / "v1" / "test" / "batch_004"
    )

    archive_path = export_dir / "wing_thermal_sky_540.zip"
    review_path = (
        export_dir / "wing_thermal_sky_540.review.json"
    )

    assert zipfile.is_zipfile(archive_path)
    assert review_path.exists()

    cvat.push_batch_to_hf(*args)

    assert fake_api.commit_calls == 2
    assert len(export_calls) == 2
    assert zipfile.is_zipfile(archive_path)

    review = json.loads(
        review_path.read_text(encoding="utf-8")
    )

    assert review["deleted_frames"] == [1, 3, 5]

    repo_zip = (
        "labels_curated/v1/test/batch_004/"
        "wing_thermal_sky_540.zip"
    )

    repo_review = (
        "labels_curated/v1/test/batch_004/"
        "wing_thermal_sky_540.review.json"
    )

    fake_api.existing = {repo_zip, repo_review}

    cvat.push_batch_to_hf(*args)

    assert fake_api.commit_calls == 2
    assert len(export_calls) == 2

    fake_api.existing = {repo_zip}

    with pytest.raises(
        RuntimeError,
        match="Incomplete HF export",
    ):
        cvat.push_batch_to_hf(*args)

    assert fake_api.commit_calls == 2
    assert len(export_calls) == 2
