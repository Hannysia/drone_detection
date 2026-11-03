import cv2
import numpy as np
import pytest

from src.processing.extractor import extract_frames_and_labels, select_annotation_zips


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "demo.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5,
        (32, 32),
    )
    assert writer.isOpened()

    try:
        for value in (20, 40, 60, 80, 100, 120, 140):
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    positive = tmp_path / "positive"
    negative = tmp_path / "negative"
    positive.mkdir()
    negative.mkdir()

    review = {
        "video_name": "demo",
        "size": 7,
        "start_frame": 0,
        "stop_frame": 6,
        "frame_filter": "",
        "included_frames": None,
        "deleted_frames": [0, 3, 6],
    }

    return video, positive, negative, review


def test_saved_images_and_boxes_follow_original_frames(clip):
    video, positive, negative, review = clip
    annotations = {
        1: [[2, 4, 8, 12]],
        3: [[8, 8, 16, 16], [0, 0, 8, 8]],
    }

    extract_frames_and_labels(
        video, annotations, "demo", positive, negative, review,
    )

    assert sorted(p.name for p in positive.iterdir()) == [
        "demo_f0002.jpg",
        "demo_f0002.txt",
        "demo_f0005.jpg",
        "demo_f0005.txt",
    ]
    assert sorted(p.name for p in negative.iterdir()) == [
        "demo_f0003.jpg",
        "demo_f0006.jpg",
    ]

    for folder, index, value in [
        (positive, 2, 40),
        (positive, 5, 100),
        (negative, 3, 60),
        (negative, 6, 120),
    ]:
        image = cv2.imread(str(folder / f"demo_f{index:04d}.jpg"))
        assert image is not None
        assert float(image.mean()) == pytest.approx(value, abs=5)

    assert (positive / "demo_f0002.txt").read_text() == (
        "0 0.187500 0.312500 0.250000 0.375000\n"
    )
    assert (positive / "demo_f0005.txt").read_text() == (
        "0 0.500000 0.500000 0.500000 0.500000\n"
        "0 0.125000 0.125000 0.250000 0.250000\n"
    )


def test_empty_gt_produces_only_retained_negatives(clip):
    video, positive, negative, review = clip

    extract_frames_and_labels(
        video, {}, "demo", positive, negative, review,
    )

    assert list(positive.iterdir()) == []
    assert sorted(p.name for p in negative.iterdir()) == [
        "demo_f0002.jpg",
        "demo_f0003.jpg",
        "demo_f0005.jpg",
        "demo_f0006.jpg",
    ]


def test_all_deleted_frames_produce_no_files(clip):
    video, positive, negative, review = clip
    review["deleted_frames"] = list(range(7))

    extract_frames_and_labels(
        video, {}, "demo", positive, negative, review,
    )

    assert list(positive.iterdir()) == []
    assert list(negative.iterdir()) == []


@pytest.mark.parametrize("size", [6, 8])
def test_wrong_video_length_fails(clip, size):
    video, positive, negative, review = clip
    review.update(size=size, stop_frame=size - 1, deleted_frames=[])

    with pytest.raises(
        ValueError,
        match="more frames|Could not read source frame",
    ):
        extract_frames_and_labels(
            video, {}, "demo", positive, negative, review,
        )


@pytest.mark.parametrize("frame", [0, 5, True])
def test_out_of_range_mot_frame_fails_before_writing(clip, frame):
    video, positive, negative, review = clip

    with pytest.raises(ValueError, match="Invalid MOT"):
        extract_frames_and_labels(
            video,
            {frame: [[0, 0, 8, 8]]},
            "demo",
            positive,
            negative,
            review,
        )

    assert list(positive.iterdir()) == []
    assert list(negative.iterdir()) == []


def test_wrong_review_video_fails(clip):
    video, positive, negative, review = clip
    review["video_name"] = "other"

    with pytest.raises(ValueError, match="video_name"):
        extract_frames_and_labels(
            video, {}, "demo", positive, negative, review,
        )


def test_curated_selection_requires_review_and_never_uses_old_labels():
    old = "labels/train/batch_001/demo.zip"
    curated = "labels_curated/v1/train/batch_001/demo.zip"

    with pytest.raises(ValueError, match="No curated"):
        select_annotation_zips([old], "train", "v1")

    with pytest.raises(ValueError, match="Missing review"):
        select_annotation_zips([old, curated], "train", "v1")

    review = curated.removesuffix(".zip") + ".review.json"

    assert select_annotation_zips(
        [old, curated, review], "train", "v1",
    ) == [curated]