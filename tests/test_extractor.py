import pytest
from pathlib import Path
from src.processing.extractor import parse_annotations, select_annotation_zips
from src.processing import extractor



def test_parse_annotations_groups_valid_rows(tmp_path):
    path = tmp_path / "gt.txt"
    path.write_text(
        "1,1,100.0,150.0,50.0,60.0,1,1,1\n"
        "1,2,200.0,250.0,30.0,40.0,1,1,1\n"
        "2,1,105.0,155.0,50.0,60.0,1,1,1\n"
        "   \n",
        encoding="utf-8",
    )

    assert parse_annotations(path) == {
        1: [
            [100.0, 150.0, 50.0, 60.0],
            [200.0, 250.0, 30.0, 40.0],
        ],
        2: [[105.0, 155.0, 50.0, 60.0]],
    }


@pytest.mark.parametrize(
    ("row", "reason"),
    [
        ("3,1,10", "expected at least 6 MOT fields"),
        ("abc,1,10,20,30,40", "invalid frame ID or bounding box number"),
        ("1,1,10,20,bad,40", "invalid frame ID or bounding box number"),
        ("1,1,nan,20,30,40", "invalid frame ID or bounding box dimensions"),
        ("1,1,10,20,0,40", "invalid frame ID or bounding box dimensions"),
    ],
)
def test_parse_annotations_reports_invalid_row(tmp_path, row, reason):
    path = tmp_path / "gt.txt"
    path.write_text(f"1,1,10,20,30,40\n\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        parse_annotations(path)

    assert f"{path}:3: {reason}" in str(error.value)



def test_selects_zip_from_nested_batch():
    files = [
        "labels_curated/v1/train/batch_001/sample.zip",
        "labels_curated/v1/train/batch_002/other.zip",
        "labels_curated/v1/val/batch_001/sample.zip",
        "train/sample.mp4",
    ]

    files += [
        path.removesuffix(".zip") + ".review.json"
        for path in files
        if path.endswith(".zip")
    ]

    result = select_annotation_zips(
        files,
        split="train",
        video_name="sample",
        version="v1"
    )

    assert result == ["labels_curated/v1/train/batch_001/sample.zip"]


def test_missing_annotation_zip_fails():
    with pytest.raises(ValueError, match="found 0"):
        select_annotation_zips(
            [],
            split="train",
            video_name="sample",
            version="v1"
        )


def test_duplicate_annotation_zip_fails():
    files = [
        "labels_curated/v1/train/batch_001/sample.zip",
        "labels_curated/v1/train/batch_002/sample.zip",
    ]

    with pytest.raises(ValueError, match="found 2"):
        select_annotation_zips(
            files,
            split="train",
            video_name="sample",
            version="v1"
        )



def test_extraction_fails_when_jpeg_cannot_be_saved(
    tmp_path,
    monkeypatch,
):
    class FakeCapture:
        def __init__(self):
            self.read_count = 0
            self.released = False

        def get(self, prop):
            return 10

        def isOpened(self):
            return self.read_count == 0

        def read(self):
            self.read_count += 1
            return True, object()

        def release(self):
            self.released = True

    capture = FakeCapture()

    monkeypatch.setattr(
        extractor.cv2,
        "VideoCapture",
        lambda _: capture,
    )

    monkeypatch.setattr(
        extractor.cv2,
        "imwrite",
        lambda *args: False,
    )

    positive_dir = tmp_path / "positives"
    negative_dir = tmp_path / "negatives"

    positive_dir.mkdir()
    negative_dir.mkdir()

    annotations = {1: [[1, 1, 2, 2]]}

    with pytest.raises(OSError, match="Could not save JPEG"):
        extractor.extract_frames_and_labels(
            video_path=Path("dummy.mp4"),
            annotations=annotations,
            video_base_name="demo",
            pos_dir=positive_dir,
            neg_dir=negative_dir,
            review={
                "video_name": "demo",
                "size": 1,
                "start_frame": 0,
                "stop_frame": 0,
                "frame_filter": "",
                "included_frames": None,
                "deleted_frames": [],
            },
        )

    assert capture.released
    assert not (positive_dir / "demo_f0001.txt").exists()
