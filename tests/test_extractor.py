import pytest
from pathlib import Path
from src.processing.extractor import parse_annotations, select_annotation_zips
from src.processing import extractor

def test_parse_annotations_standard_and_invalid_lines(tmp_path):
    mock_gt_file = tmp_path / "gt_mock.txt"
    
    gt_content = (
        "1,1,100.0,150.0,50.0,60.0,1,1,1\n"
        "1,2,200.0,250.0,30.0,40.0,1,1,1\n"
        "2,1,105.0,155.0,50.0,60.0,1,1,1\n"
        "3,1,10.0\n"
    )
    mock_gt_file.write_text(gt_content, encoding="utf-8")

    result = parse_annotations(mock_gt_file)

    assert len(result) == 2
    assert 1 in result
    assert 2 in result
    assert 3 not in result
    
    assert len(result[1]) == 2
    assert result[1][0] == [100.0, 150.0, 50.0, 60.0]
    assert result[1][1] == [200.0, 250.0, 30.0, 40.0]
    
    assert len(result[2]) == 1
    assert result[2][0] == [105.0, 155.0, 50.0, 60.0]



def test_selects_zip_from_nested_batch():
    files = [
        "labels/train/batch_001/sample.zip",
        "labels/train/batch_002/other.zip",
        "labels/val/batch_001/sample.zip",
        "train/sample.mp4",
    ]

    result = select_annotation_zips(
        files,
        split="train",
        video_name="sample",
    )

    assert result == ["labels/train/batch_001/sample.zip"]


def test_missing_annotation_zip_fails():
    with pytest.raises(ValueError, match="found 0"):
        select_annotation_zips(
            [],
            split="train",
            video_name="sample",
        )


def test_duplicate_annotation_zip_fails():
    files = [
        "labels/train/batch_001/sample.zip",
        "labels/train/batch_002/sample.zip",
    ]

    with pytest.raises(ValueError, match="found 2"):
        select_annotation_zips(
            files,
            split="train",
            video_name="sample",
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
        )

    assert capture.released
    assert not (positive_dir / "demo_f0001.txt").exists()
