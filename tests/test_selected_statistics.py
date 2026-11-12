import numpy as np
import pytest
from PIL import Image

from src.processing.statistics import (
    calculate_dataset_local_contrast,
    gather_all_splits_geometry,
    gather_dataset_frame_metadata_df,
    gather_dataset_geometry,
    gather_spatial_coordinates,
)


@pytest.mark.parametrize("mode", ["all", "selected", "empty"])
def test_statistics_respect_manifest_selection(tmp_path, mode):
    name = "wing_rgb_sky_001_f0001.jpg"
    other = "wing_rgb_sky_001_f0002.jpg"
    negative = "no_drone_rgb_sky_002_f0001.jpg"

    for split in ("train", "val"):
        for category in ("RAW_POSITIVES", "RAW_NEGATIVES"):
            (tmp_path / split / category).mkdir(parents=True)

    pixels = np.full((20, 30, 3), 100, dtype=np.uint8)

    for split, filename in (
        ("train", name),
        ("train", other),
        ("val", name),
    ):
        path = tmp_path / split / "RAW_POSITIVES" / filename
        Image.fromarray(pixels).save(path)
        path.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n")

    Image.fromarray(pixels).save(
        tmp_path / "train" / "RAW_NEGATIVES" / negative
    )

    selections = {
        "all": None,
        "selected": {("train", name), ("train", negative)},
        "empty": set(),
    }
    selected = selections[mode]
    expected_train = {"all": 2, "selected": 1, "empty": 0}[mode]
    expected_positive = {"all": 3, "selected": 1, "empty": 0}[mode]
    expected_frames = {"all": 4, "selected": 2, "empty": 0}[mode]
    splits = ["train", "val"]

    images, boxes = gather_dataset_geometry(
        tmp_path / "train" / "RAW_POSITIVES",
        selected_frames=selected,
    )
    assert len(images) == len(boxes) == expected_train

    _, _, total_boxes = gather_spatial_coordinates(
        tmp_path,
        splits,
        selected_frames=selected,
    )
    assert total_boxes == expected_positive

    geometry = gather_all_splits_geometry(
        tmp_path,
        splits,
        selected_frames=selected,
    )
    assert len(geometry) == expected_positive

    contrast = calculate_dataset_local_contrast(
        tmp_path,
        splits,
        selected_frames=selected,
    )
    assert len(contrast) == expected_positive

    frames = gather_dataset_frame_metadata_df(
        tmp_path,
        splits,
        selected_frames=selected,
    )
    assert len(frames) == expected_frames

    if mode == "selected":
        assert set(zip(frames["split"], frames["filename"])) == selected