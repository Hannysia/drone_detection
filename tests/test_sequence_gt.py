import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest

from src.processing import trackeval_dataset, tracking
from src.processing.manifests import CATEGORY_DIRS, write_manifest

@pytest.fixture
def sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        trackeval_dataset,
        "get_video_metadata",
        lambda _: (640, 480, 25, 3),
    )
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()

    return (
        tmp_path / "example.mp4",
        tmp_path / "example.zip",
        gt_dir,
        tmp_path / "seqinfo.ini",
    )


@pytest.mark.parametrize(
    "contents",
    [b"", b" \r\n", b"1,1,2,3,4,5,1,2,1\n"],
)
def test_sequence_accepts_present_gt(sequence, contents):
    video, archive, folder, info = sequence

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("gt/gt.txt", contents)

    trackeval_dataset.prepare_trackeval_sequence(video, archive, folder, info)

    expected = "1,1,2.00,3.00,4.00,5.00,1,1,1.00\n" if contents.strip() else ""
    assert (folder / "gt.txt").read_text() == expected
    assert "seqLength=3" in info.read_text()


def test_sequence_rejects_missing_gt(sequence):
    video, archive, folder, info = sequence

    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("not_gt.txt", b"")

    with pytest.raises(ValueError, match="expected exactly one gt.txt, found 0"):
        trackeval_dataset.prepare_trackeval_sequence(video, archive, folder, info)

    assert not (folder / "gt.txt").exists()
    assert not info.exists()




def test_prepare_trackeval_sequence_validates_metadata_and_review_alignment(tmp_path, monkeypatch):
    archive = tmp_path / "seq_001.zip"
    video = tmp_path / "seq_001.mp4"
    gt_dir = tmp_path / "gt"
    info = tmp_path / "seqinfo.ini"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("gt/gt.txt", "1,1,2,3,4,5,1,2,1\n3,1,4,5,6,7,1,2,1\n")

    monkeypatch.setattr(trackeval_dataset, "get_video_metadata", lambda _: (0, 480, 25.0, 5))
    with pytest.raises(ValueError, match="Invalid sequence metadata"):
        trackeval_dataset.prepare_trackeval_sequence(video, archive, gt_dir, info)

    monkeypatch.setattr(trackeval_dataset, "get_video_metadata", lambda _: (640, 480, 25.0, 5))
    review = {
        "video_name": "seq_001",
        "size": 5,
        "start_frame": 0,
        "stop_frame": 4,
        "frame_filter": "",
        "included_frames": None,
        "deleted_frames": [1, 3],
    }
    trackeval_dataset.prepare_trackeval_sequence(video, archive, gt_dir, info, review=review)
    assert "seqLength=3\n" in info.read_text(encoding="utf-8")

    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("gt/gt.txt", "4,1,2,3,4,5,1,2,1\n")
    with pytest.raises(ValueError, match="frame IDs must be integers"):
        trackeval_dataset.prepare_trackeval_sequence(video, archive, gt_dir, info, review=review)


def test_compile_kaggle_mot_dataset_uses_curated_annotations_and_baseline_frames(tmp_path):
    hf_dir = tmp_path / "hf"
    baseline_dir = tmp_path / "baseline"
    mot_dir = tmp_path / "mot"
    seqmaps_dir = mot_dir / "seqmaps"

    decoy_zip = hf_dir / "labels" / "val" / "quad_rgb_sky_001.zip"
    decoy_zip.parent.mkdir(parents=True)
    with zipfile.ZipFile(decoy_zip, "w") as out:
        out.writestr("gt/gt.txt", "99,1,1,1,2,2,1,2,1\n")

    curated_dir = hf_dir / "labels_curated" / "v1" / "val" / "batch_001"
    curated_dir.mkdir(parents=True)
    curated_zip = curated_dir / "quad_rgb_sky_001.zip"
    with zipfile.ZipFile(curated_zip, "w") as out:
        out.writestr("gt/gt.txt", "1,1,10,12,20,24,1,2,1\n2,1,14,16,20,24,1,2,1\n")

    review = {
        "video_name": "quad_rgb_sky_001",
        "size": 3,
        "start_frame": 0,
        "stop_frame": 2,
        "frame_filter": "",
        "included_frames": None,
        "deleted_frames": [1],
    }
    (curated_dir / "quad_rgb_sky_001.review.json").write_text(json.dumps(review), encoding="utf-8")

    rows = [
        {"filename": "quad_rgb_sky_001_f0001.jpg", "split": "val", "category": "positive"},
        {"filename": "quad_rgb_sky_001_f0003.jpg", "split": "val", "category": "positive"},
    ]
    for item in rows:
        folder = baseline_dir / "val" / CATEGORY_DIRS[item["category"]]
        folder.mkdir(parents=True, exist_ok=True)
        img = folder / item["filename"]
        assert cv2.imwrite(str(img), np.zeros((48, 64, 3), dtype=np.uint8))
        img.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest_path = tmp_path / "baseline_main.csv"
    write_manifest(rows, manifest_path, include_excluded=False)

    compiled = trackeval_dataset.compile_kaggle_mot_dataset(
        hf_dir,
        mot_dir,
        seqmaps_dir,
        (".mp4",),
        splits=("val",),
        baseline_dataset_root=baseline_dir,
        manifest_path=manifest_path,
    )

    assert compiled == {"val": ["quad_rgb_sky_001"]}
    assert (seqmaps_dir / "drone-val.txt").read_text(encoding="utf-8").splitlines() == ["name", "quad_rgb_sky_001"]
    seqinfo = (mot_dir / "drone-val" / "quad_rgb_sky_001" / "seqinfo.ini").read_text(encoding="utf-8")
    assert "seqLength=2\n" in seqinfo
    assert "imWidth=64\n" in seqinfo
    gt_lines = (mot_dir / "drone-val" / "quad_rgb_sky_001" / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()
    assert gt_lines == [
        "1,1,10.00,12.00,20.00,24.00,1,1,1.00",
        "2,1,14.00,16.00,20.00,24.00,1,1,1.00",
    ]

    dup_dir = hf_dir / "labels_curated" / "v1" / "val" / "batch_002"
    dup_dir.mkdir(parents=True)
    with zipfile.ZipFile(dup_dir / "quad_rgb_sky_001.zip", "w") as out:
        out.writestr("gt/gt.txt", "")
    with pytest.raises(ValueError, match="Duplicate curated archive"):
        trackeval_dataset.compile_kaggle_mot_dataset(
            hf_dir,
            mot_dir,
            seqmaps_dir,
            (".mp4",),
            splits=("val",),
            baseline_dataset_root=baseline_dir,
            manifest_path=manifest_path,
        )


def test_run_yolo_inference_for_tracking_uses_rect_false_and_curated_mapping(tmp_path, monkeypatch):
    hf_dir = tmp_path / "hf"
    seqmaps = tmp_path / "seqmaps"
    out_dir = tmp_path / "dets"
    seqmaps.mkdir()
    (hf_dir / "val").mkdir(parents=True)
    (hf_dir / "val" / "vid_001.mp4").write_bytes(b"mp4")
    (seqmaps / "drone-val.txt").write_text("name\nvid_001\n", encoding="utf-8")

    review_dir = hf_dir / "labels_curated" / "v1" / "val" / "batch_001"
    review_dir.mkdir(parents=True)
    (review_dir / "vid_001.review.json").write_text(
        json.dumps({
            "video_name": "vid_001",
            "size": 4,
            "start_frame": 0,
            "stop_frame": 3,
            "frame_filter": "",
            "included_frames": None,
            "deleted_frames": [1],
        }),
        encoding="utf-8",
    )

    calls = []

    class _FakeTensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=float)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    def fake_model_call(video, **kwargs):
        calls.append(dict(kwargs))
        return [
            SimpleNamespace(boxes=SimpleNamespace(xyxy=_FakeTensor([[10, 10, 30, 30]]), conf=_FakeTensor([0.9]), cls=_FakeTensor([0]))),
            SimpleNamespace(boxes=SimpleNamespace(xyxy=_FakeTensor([[99, 99, 120, 120]]), conf=_FakeTensor([0.8]), cls=_FakeTensor([0]))),
            SimpleNamespace(boxes=SimpleNamespace(xyxy=_FakeTensor([[20, 20, 40, 40]]), conf=_FakeTensor([0.85]), cls=_FakeTensor([0]))),
            SimpleNamespace(boxes=None),
        ]

    monkeypatch.setattr(tracking, "YOLO", lambda *a, **k: fake_model_call)
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"pt")

    tracking.run_yolo_inference_for_tracking(
        model_path,
        seqmaps,
        hf_dir,
        out_dir,
        (".mp4",),
        imgsz=1280,
        splits=("val",),
    )

    assert len(calls) == 1
    assert calls[0]["rect"] is False
    assert calls[0]["imgsz"] == 1280
    lines = (out_dir / "val" / "vid_001.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1,-1,10.00,10.00,20.00,20.00,0.9000")
    assert lines[1].startswith("2,-1,20.00,20.00,20.00,20.00,0.8500")