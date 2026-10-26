
import pytest

from src.integration.cvat import curated_annotation_path


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (
            "v2",
            "labels_curated/v2/val/batch_001/video_001.zip",
        ),
        (
            "v3",
            "labels_curated/v3/val/batch_001/video_001.zip",
        ),
        (
            "v15",
            "labels_curated/v15/val/batch_001/video_001.zip",
        ),
    ],
)
def test_curated_annotation_path(version, expected):
    result = curated_annotation_path(
        version,
        "val",
        "batch_001",
        "video_001",
    )

    assert result == expected


@pytest.mark.parametrize(
    "version",
    ["v0", "v02", "../v2", "v2/other", "latest"],
)
def test_invalid_version(version):
    with pytest.raises(ValueError, match="Invalid dataset version"):
        curated_annotation_path(
            version,
            "val",
            "batch_001",
            "video_001",
        )


def test_invalid_batch_name():
    with pytest.raises(ValueError, match="Invalid name"):
        curated_annotation_path(
            "v2",
            "val",
            "../batch_001",
            "video_001",
        )


def test_invalid_split():
    with pytest.raises(ValueError, match="Invalid split"):
        curated_annotation_path(
            "v2",
            "validation",
            "batch_001",
            "video_001",
        )
