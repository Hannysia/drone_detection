from pathlib import Path
import pytest

from src.utils.io import parse_frame_name, parse_video_id


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("wing_rgb_sky_001_f0001.jpg", "wing_rgb_sky_001"),
        (Path("dir/no_drone_rgb_sky_002_f0042.txt"), "no_drone_rgb_sky_002"),
        ("no_drone_rgb_sky_001.mp4", "no_drone_rgb_sky_001"),
        ("frame_1.jpg", "frame_1"),
    ],
)
def test_parse_video_id_strips_only_frame_suffix(raw, expected):
    assert parse_video_id(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("wing_rgb_sky_001_f0001.jpg", ("wing_rgb_sky_001", 1)),
        (Path("sub/no_drone_rgb_sky_002_f0042.jpg"), ("no_drone_rgb_sky_002", 42)),
    ],
)
def test_parse_frame_name_extracts_video_and_number(raw, expected):
    assert parse_frame_name(raw) == expected


@pytest.mark.parametrize(
    "bad_name",
    [
        "wing_rgb_sky_001.jpg",
        "_f0001.jpg",
        "wing_f0000.jpg",
        "wing_fabc.jpg",
        "wing_f.jpg",
    ],
)
def test_parse_frame_name_rejects_invalid_names(bad_name):
    with pytest.raises(ValueError, match="Invalid frame filename"):
        parse_frame_name(bad_name)