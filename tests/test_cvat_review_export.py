
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        archive_path = (
            tmp_path
            / "exports"
            / "v2"
            / "val"
            / "batch_001"
            / "wing_thermal_sky_540.zip"
        )

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
