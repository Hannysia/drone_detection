import pytest

from src.processing.dataset import (
    extract_metadata_from_filename,
    parse_filename,
    parse_frame_filename_logic,
    rename_dataset_videos_safely,
)
from src.processing.dataset_statistics import get_splits_metadata_df


@pytest.mark.parametrize(
    ("filename", "expected_tuple", "expected_prefix"),
    [
        ("wing_rgb_sky_001.mp4", ("wing", "rgb", "sky"), "wing_rgb_sky"),
        (
            "delta_thermal_field_folder_clip.v2.mov",
            ("delta", "thermal", "field"),
            "delta_thermal_field",
        ),
        (
            "no_drone_rgb_sky_folder_clip.mp4",
            ("no_drone", "rgb", "sky"),
            "no_drone_rgb_sky",
        ),
        (
            "no_drone_thermal_complex_601.mp4",
            ("no_drone", "thermal", "complex"),
            "no_drone_thermal_complex",
        ),
    ],
)
def test_valid_filenames_preserve_shape_camera_background(
    filename, expected_tuple, expected_prefix
):
    assert extract_metadata_from_filename(filename) == expected_tuple
    assert parse_filename(filename) == expected_prefix


@pytest.mark.parametrize(
    "filename",
    [
        "",
        "invalid.mp4",
        "no_drone_f0001.jpg",
        "no_drone_rgb_224.mp4",
        "wing_rgb_001.mp4",
        "001_rgb_sky_002.mp4",
        "wing__sky_001.mp4",
    ],
)
def test_malformed_or_truncated_filenames_return_none(filename):
    assert extract_metadata_from_filename(filename) is None
    assert parse_filename(filename) is None


def test_rename_dataset_videos_safely_preserves_no_drone_background(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()

    (source / "no_drone_rgb_sky_folder_clip.mp4").write_bytes(b"neg")
    (source / "wing_thermal_field_folder_clip.mov").write_bytes(b"pos")
    (source / "no_drone_rgb_224.mp4").write_bytes(b"legacy")
    (source / "notes.txt").write_text("ignore", encoding="utf-8")

    rename_dataset_videos_safely(source, dest, [".mp4", ".mov"])

    renamed = sorted(p.name for p in dest.iterdir())
    assert renamed == [
        "no_drone_rgb_sky_001.mp4",
        "wing_thermal_field_002.mov",
    ]
    assert (dest / "no_drone_rgb_sky_001.mp4").read_bytes() == b"neg"
    assert (dest / "wing_thermal_field_002.mov").read_bytes() == b"pos"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "wing_rgb_sky_001_f0001.jpg",
            {"shape": "wing", "camera": "rgb", "background": "sky"},
        ),
        (
            "no_drone_rgb_sky_002_f0001.jpg",
            {"shape": "no_drone", "camera": "rgb", "background": "sky"},
        ),
        (
            "no_drone_rgb_forest_595_f0496.jpg",
            {"shape": "no_drone", "camera": "rgb", "background": "complex"},
        ),
        (
            "no_drone_rgb_sky_596",
            {"shape": "no_drone", "camera": "rgb", "background": "sky"},
        ),
        (
            "no_drone_rgb_224_f0001.jpg",
            {"shape": "no_drone", "camera": "rgb", "background": "unknown"},
        ),
        (
            "no_drone_f0001.jpg",
            {"shape": "no_drone", "camera": "unknown", "background": "unknown"},
        ),
        (
            "unknown.jpg",
            {"shape": "unknown", "camera": "unknown", "background": "unknown"},
        ),
    ],
)
def test_parse_frame_filename_logic_handles_no_drone_and_legacy_names(
    filename, expected
):
    assert parse_frame_filename_logic(filename) == expected


def test_get_splits_metadata_df_preserves_no_drone_background(tmp_path):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "no_drone_rgb_sky_001.mp4").write_bytes(b"v1")
    (train_dir / "no_drone_rgb_224.mp4").write_bytes(b"v2")

    df = get_splits_metadata_df(tmp_path, ["train"])

    assert len(df) == 1
    assert df.iloc[0].to_dict() == {
        "Class": "no_drone_rgb_sky",
        "Morphology": "no_drone",
        "Camera": "rgb",
        "Background": "sky",
        "Split": "TRAIN",
    }